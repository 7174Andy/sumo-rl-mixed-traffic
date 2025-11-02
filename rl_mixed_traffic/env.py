import time
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from pathlib import Path
import os
import sys

from utils.sumo_utils import get_vehicles_pos_speed, start_traci, compute_ring_length
from config import SumoConfig

if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "share" / "sumo" / "tools"
    sys.path.append(str(tools))
else:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable to your SUMO install path."
    )

import traci


class RingRoadEnv(gym.Env):
    """
    Minimal RL env that controls one agent vehicle's speed on a ring via setSpeed().
    state = concat([v_norm...N], [p_norm...N]) of all vehicles in the network, normalized.
    action = acceleration command (m/s²)

    Notes:
    - The episode runs for a fixed number of SUMO steps or until SUMO finishes.
    - We assume 'car0' exists in routes and departs early.
    """

    def __init__(
        self,
        sumo_config: SumoConfig,
        agent_id: str = "car1",
        gui: bool = False,
        max_accel: float = 3.0,
        min_accel: float = -3.0,
        episode_length: float = 200.0,
        num_vehicles: int = 3,
    ):
        self.sumo_config = sumo_config
        self.agent_id = agent_id
        self.gui = gui or sumo_config.use_gui
        self.max_accel = max_accel
        self.min_accel = min_accel
        self.num_vehicles = num_vehicles

        # TODO: Linear interpolation?

        self.episode_length = episode_length
        self.step_length = sumo_config.step_length
        self.max_steps = int(episode_length / self.step_length)

        self.ring_length = None

        # Meeting notes 10/29
        # TODO: acceleration as action
        # - penalize large gap
        # - HDV being the leader and CAV follower behind it
        # - Reward function design
        #   - speed of the platoon (mean speed of first K followers)
        #   - comfort penalty (acceleration changes)
        #   - safety penalty (time-to-collision with follower)
        #   - fuel consumption?
        # - traditional control methods comparison is good

        # TODO: Deep Q Network implementation
        # What should be the good scenario for the controlling vehicle?

        self.cmd_speed = 0.0
        self.v_max = 30.0

    @property
    def action_space(self):
        # [v1..vN, p1..pN] (both normalized into [0,1]), padded/truncated to max_vehicles
        return Box(
            low=-self.max_accel, high=self.max_accel, shape=(1,), dtype=np.float32
        )

    @property
    def observation_space(self):
        # Three vehicles, each with (velocity, absolute_pos)
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        return Box(low=0.0, high=1.0, shape=(2 * self.num_vehicles,), dtype=np.float32)

    def open_traci(self):
        if not traci.isLoaded():
            start_traci(
                SumoConfig(
                    sumocfg_path=self.sumo_config.sumocfg_path,
                    use_gui=self.gui,
                    delay_ms=0,
                )
            )

        if self.ring_length is None:
            self.ring_length = compute_ring_length(self.agent_id)

    def close_traci(self):
        if traci.isLoaded():
            traci.close(False)

    def get_state(self) -> np.ndarray:
        """Observation = concat([v_norm...N], [p_norm...N]) padded/truncated to max_vehicles.
        v_norm = speed / v_max
        p_norm = position / ring_length
        """
        ids_sorted, speeds_mps_sorted, positions_m_sorted_mod = get_vehicles_pos_speed(
            self.ring_length
        )
        L = float(max(1e-6, self.ring_length or 0.0))
        vmax = max(1e-6, self.v_max)

        # Normalize
        v_norm = np.clip(np.array(speeds_mps_sorted, dtype=np.float32) / vmax, 0.0, 1.0)
        p_norm = np.clip(
            np.array(positions_m_sorted_mod, dtype=np.float32) / L, 0.0, 1.0
        )

        # pad or truncate to the fixed size
        max_vehicles = self.num_vehicles

        def pad_trunc(arr, k=max_vehicles):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size >= k:
                return arr[:k]
            out = np.zeros((k,), dtype=np.float32)
            out[: arr.size] = arr
            return out

        v_final = pad_trunc(v_norm)
        p_final = pad_trunc(p_norm)

        obs = np.concatenate([v_final, p_final], axis=0).astype(np.float32)

        # Sanity check
        assert self.observation_space.contains(obs), f"Invalid observation: {obs}"
        return obs

    def reset(self):
        self.close_traci()
        self.open_traci()
        self.step_count = 0
        self.cmd_speed = 0.0
        self.v_max = 30.0

        # Warm up the simulation
        warmup = 0
        while self.agent_id not in traci.vehicle.getIDList() and warmup < 200:
            traci.simulationStep()
            warmup += 1

        if self.agent_id in traci.vehicle.getIDList():
            self.v_max = traci.vehicle.getMaxSpeed(self.agent_id)
            v_now = traci.vehicle.getSpeed(self.agent_id)
            self.cmd_speed = max(0.5, min(v_now, self.v_max))

        return self.get_state(), {}

    def render(self):
        return None

    def step(self, action: int):
        """Apply the given action and return the new state, reward, and done flag.

        Args:
            action (int): The action to apply.

        Returns:
            Tuple[int, float, bool]: The new state, reward, and done flag.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Apply action
        if isinstance(action, (list, list, np.ndarray)):
            a = float(np.array(action, dtype=np.float32).reshape(-1)[0])
        else:
            a = float(action)
        a = float(np.clip(a, self.min_accel, self.max_accel))

        if self.agent_id in traci.vehicle.getIDList():
            traci.vehicle.setAcceleration(self.agent_id, a)

        traci.simulationStep()
        time.sleep(0.01)  # to avoid busy waiting
        self.step_count += 1

        obs = self.get_state()
        reward = self.compute_reward(obs=obs, action=a)
        done = self.terminal()
        info = {}

        return obs, reward, done, info

    def compute_reward(self, obs: np.ndarray, action: float | None = None) -> float:
        """
        Network-speed heavy reward:
        Platton speed = mean speed of all vehicles
        Car Tracking =
        """
        n = obs.size // 2
        v_norm = obs[:n].astype(np.float32)
        p_norm = obs[n:].astype(np.float32)

        # Platoon speed
        platoon_speed = float(np.clip(np.mean(v_norm), 0.0, 1.0))

        # CAV -> Leader Tracking
        v0 = float(v_norm[1])  # CAV Speed
        v_leader = float(v_norm[0])  # Leader Speed
        s_star_m = 2.0 + v0 * 1.2  # desired gap (m)
        s_star_norm = s_star_m / max(1e-6, self.ring_length)

        g0_norm = float((p_norm[0] - p_norm[1]) % 1.0)  # gap to leader (normalized)

        e_g = float(g0_norm - s_star_norm) / max(1e-6, s_star_norm)
        S_gap = float(np.exp(-e_g * e_g))

        dv = (v_leader - v0) / max(1e-6, self.v_max)
        S_dv = float(np.exp(-dv * dv))

        track = 0.5 * S_gap + 0.5 * S_dv

        # backward safety penalty: protect follower
        v1 = float(v_norm[2]) * self.v_max  # follower speed
        v0 = float(v_norm[1]) * self.v_max  # CAV speed

        # back gap
        g1_norm = float(p_norm[1] - p_norm[2]) % 1.0
        g1 = g1_norm * self.ring_length

        # TTC with follower
        if v1 > v0 + 1e-6:
            ttc1 = g1 / (v1 - v0 + 1e-6)
        else:
            ttc1 = np.inf

        ttc_ref = 2.0  # seconds (can adjust)
        g_min_back = 5.0  # meters (can adjust)
        S_ttc = float(
            np.clip(0.0 if not np.isfinite(ttc1) else ttc1 / ttc_ref, 0.0, 1.0)
        )
        S_gap_back = float(np.clip(g1 / max(1e-6, g_min_back), 0.0, 1.0))
        S_back = 0.5 * S_ttc + 0.5 * S_gap_back

        # Comfort penalty
        a_norm = float((action / max(1e-6, self.max_accel)) ** 2)
        comfort_penalty = a_norm

        reward = (
            1.5 * platoon_speed + 0.5 * track - 0.2 * comfort_penalty + 0.6 * S_back
        )
        return float(reward)

    def terminal(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if traci.simulation.getMinExpectedNumber() == 0:
            return True
        # When the two vehicles collide, SUMO ends the simulation
        if traci.simulation.getCollidingVehiclesNumber() > 0:
            return True
        return False

    def close(self):
        self.close_traci()
