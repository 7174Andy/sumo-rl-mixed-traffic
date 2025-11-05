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
        episode_length: float = 500.0,
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

    def reset(self, seed: int | None = None, options: dict = None):
        """Reset the environment and return the initial observation.
        Returns:
            np.ndarray: The initial observation.
            dict: An empty info dictionary.
        """
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
    
    def apply_acceleration(self, veh_ids, acc, smooth: bool = True):
        """
        Apply acceleration a over one sim step: v_next = clip(v + a*dt, 0, v_max).
        If smooth=True, use slowDown (ramps over ~dt); else setSpeed instantly.

        Args:
            veh_ids: str or list[str]
            acc: float or list[float] (m/s^2), aligned with veh_ids
            smooth: use vehicle.slowDown (True) or setSpeed (False)
        """
        # normalize to lists
        if isinstance(veh_ids, str):
            veh_ids = [veh_ids]
        if isinstance(acc, (int, float, np.floating)):
            acc = [float(acc)]
        assert len(veh_ids) == len(acc), "veh_ids and acc must have same length"

        # duration for slowDown ~ one control tick (TraCI expects ms)
        duration_ms = max(1, int(round(self.step_length * 1000.0)))

        for vid, a in zip(veh_ids, acc):
            if vid not in traci.vehicle.getIDList():
                continue
            if a is None:
                continue

            # clip accel and integrate next speed
            a = float(np.clip(a, self.min_accel, self.max_accel))
            v_now = float(traci.vehicle.getSpeed(vid))
            v_next = float(np.clip(v_now + a * self.step_length, 0.0, self.v_max))

            if smooth:
                # ramp to v_next over ~dt
                traci.vehicle.slowDown(vid, v_next, duration_ms)
            else:
                # jump to v_next immediately
                traci.vehicle.setSpeed(vid, v_next)

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
            self.apply_acceleration(self.agent_id, a, smooth=False)

        traci.simulationStep()
        time.sleep(0.01)  # to avoid busy waiting
        self.step_count += 1

        obs = self.get_state()
        reward = self.compute_reward()
        done = self.terminal()
        info = {}

        return obs, reward, done, info
    

    def get_followers_chain(self, leader_id: str, depth: int = 5):
        """
        Returns a list of (veh_id, gap_meters) for up to `depth` followers
        directly behind `leader_id`, ordered nearest-first.

        - Uses traci.vehicle.getFollower() iteratively.
        - If a follower is missing (e.g., lane change or no vehicle), the chain stops.
        - On a ring with a single lane, this will walk the platoon behind your agent.

        Parameters
        ----------
        leader_id : str
            The vehicle ID to start from (typically the ego agent).
        depth : int
            Max number of followers to return.

        Returns
        -------
        List[Tuple[str, float]]
            [(follower_id, gap_to_front_in_meters), ...]
        """
        chain = []
        current = leader_id
        seen = set([leader_id])  # avoid weird loops if SUMO returns something odd

        for _ in range(max(0, depth)):
            try:
                # SUMO TraCI: returns (vehID, gap) or None/('', -1) when no follower
                nxt = traci.vehicle.getFollower(current)
            except Exception:
                break

            if not nxt:
                break

            follower_id, gap = nxt  # follower directly behind `current`
            if follower_id in (None, "", current) or follower_id in seen:
                break

            # gap can be -1 in edge cases; sanitize
            gap = float(gap) if gap is not None else -1.0
            chain.append((follower_id, gap))
            seen.add(follower_id)
            current = follower_id

        return chain


    def compute_reward(self) -> float:
        if self.agent_id not in traci.vehicle.getIDList():
            return 0.0

        V_MAX = max(1e-6, self.v_max)

        # Followers info (light touch)
        K = 5
        chain = self.get_followers_chain(leader_id=self.agent_id, depth=K)
        f_ids = [fid for fid, _ in chain]
        if f_ids:
            v_followers = [traci.vehicle.getSpeed(i) for i in f_ids]
            mean_v = float(sum(v_followers)) / len(v_followers)
        else:
            mean_v = traci.vehicle.getSpeed(self.agent_id)

        # Ego + leader
        v_ego = traci.vehicle.getSpeed(self.agent_id)
        v_lead = 0.0
        d_gap = 1e9
        lead = traci.vehicle.getLeader(self.agent_id)
        if lead:
            lead_id, d_gap = lead
            v_lead = traci.vehicle.getSpeed(lead_id)

        # -------- Efficiency-forward terms --------
        # 1) progress (pay per distance)
        R_prog = v_ego / V_MAX

        # 2) target-speed tracking (prefer fast cruise but don't ignore leader)
        # v_star = min(0.9 * V_MAX, v_lead + 3.0)
        # R_target = 1.0 - ((v_ego - v_star) / V_MAX) ** 2  # peaks at 1 when v≈v_star

        # 3) platoon mean (small nudge upward)
        R_mean = (mean_v / V_MAX)

        # 4) stronger penalty for lingering at low speed
        v_thresh = 0.5 * V_MAX
        R_low = - max(0.0, (v_thresh - v_ego) / max(1e-6, v_thresh))

        # Safety guardrail (small)
        d_min = 2.0
        rel = max(v_ego - v_lead, 1e-3)
        free_gap = max(d_gap - d_min, 0.0)
        ttc = free_gap / rel if rel > 0 else 1e6
        tau_safe = 0.6  # aggressive but nonzero
        R_ttc = -0.02 * ((tau_safe - ttc) / tau_safe) if ttc < tau_safe else 0.0

        # Fuel consumption penalty
        # mpg = miles_per_gallen()
        # R_fuel = -0.1 * (1.0 / max(mpg, 1e-6))

        # print(f"Step {self.step_count}: R_prog={R_prog:.3f}, R_mean={R_mean:.3f}, R_low={R_low:.3f}, R_ttc={R_ttc:.3f}")

        # Combine (weights emphasize efficiency)
        R = (
            1.0 * R_prog +
            # 1.0 * R_target +
            0.5 * R_mean +
            0.7 * R_low +
            0.15 * R_ttc
            # 0.1 * R_fuel
        )
        return float(np.clip(R, -2.0, 2.0))

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
