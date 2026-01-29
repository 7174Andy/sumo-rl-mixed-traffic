import time
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from pathlib import Path
import os
import sys

from rl_mixed_traffic.utils.sumo_utils import (
    get_vehicles_pos_speed,
    start_traci,
    compute_ring_length,
    close_traci,
)
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.env.head_vehicle_controller import HeadVehicleController

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
        num_vehicles: int = 2,
        head_speed_change_interval: float = 15.0,
        head_speed_min: float = 5.0,
        head_speed_max: float = 20.0,
        head_id: str = "car0",
        head_vehicle_controller: HeadVehicleController = None,
        # DeeP-LCC reward parameters
        v_star: float = 15.0,
        s_star: float = 20.0,
        weight_v: float = 0.8,
        weight_s: float = 0.7,
        weight_u: float = 0.1,
        spacing_min: float = 5.0,
    ):
        self.sumo_config = sumo_config
        self.agent_id = agent_id
        self.gui = gui or sumo_config.use_gui
        self.max_accel = max_accel
        self.min_accel = min_accel
        self.num_vehicles = num_vehicles
        self.prev_accel = 0.0
        self.last_jerk = 0.0

        # DeeP-LCC reward parameters
        self.v_star = v_star
        self.s_star = s_star
        self.weight_v = weight_v
        self.weight_s = weight_s
        self.weight_u = weight_u
        self.spacing_min = spacing_min

        self.episode_length = episode_length
        self.step_length = sumo_config.step_length
        self.max_steps = int(episode_length / self.step_length)

        self.ring_length = None

        self.cmd_speed = 0.0
        self.v_max = 30.0

        self.head_speed_change_interval = head_speed_change_interval
        self.head_speed_change_interval_steps = max(
            1, int(round(head_speed_change_interval / self.step_length))
        )
        self.head_speed_min = head_speed_min
        self.head_speed_max = head_speed_max
        self.head_id = head_id
        self._last_head_update_step = 0

        if head_vehicle_controller is not None:
            self.head_vehicle_controller = head_vehicle_controller
        else:
            self.head_vehicle_controller = HeadVehicleController(
                head_id=self.head_id,
                head_speed_min=self.head_speed_min,
                head_speed_max=self.head_speed_max,
            )

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

    def _update_head_speed(self):
        if (
            self.step_count - self._last_head_update_step
        ) < self.head_speed_change_interval_steps:
            return

        self._last_head_update_step = self.step_count

        # sample a new random cruising speed
        self.head_vehicle_controller.set_random_head_speed()

    def get_state(self) -> np.ndarray:
        """Observation = concat([v_norm...N], [p_norm...N]) padded/truncated to max_vehicles.
        Vehicles are sorted by ID so each index always maps to the same vehicle.
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
        close_traci()
        self.open_traci()
        self.step_count = 0
        self.cmd_speed = 0.0
        self.prev_accel = 0.0
        self.last_jerk = 0.0
        self._last_head_update_step = 0

        # Warm up the simulation
        warmup = 0
        while self.agent_id not in traci.vehicle.getIDList() and warmup < 200:
            traci.simulationStep()
            warmup += 1

        if self.agent_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeedMode(self.agent_id, 95)  # 0b1011111
            traci.vehicle.setMaxSpeed(self.agent_id, self.v_max)
            v_now = traci.vehicle.getSpeed(self.agent_id)
            self.cmd_speed = max(0.5, min(v_now, self.v_max))

        if self.head_id in traci.vehicle.getIDList():
            traci.vehicle.setMaxSpeed(self.head_id, self.v_max)

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

    def step(self, action: np.ndarray | int):
        """Apply the given action and return the new state, reward, and done flag.

        Args:
            action (np.ndarray | int): The action to apply.

        Returns:
            Tuple[int, float, bool]: The new state, reward, and done flag.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Apply action (extract scalar from array if needed)
        if isinstance(action, (list, np.ndarray)):
            a = float(np.array(action, dtype=np.float32).reshape(-1)[0])
        else:
            a = float(action)

        # Clip acceleration to valid range (defensive check)
        a = float(np.clip(a, self.min_accel, self.max_accel))
        
        # Jerk calculation
        dt = self.step_length
        jerk = (a - self.prev_accel) / dt if dt > 0 else 0.0
        self.last_jerk = jerk
        self.prev_accel = a

        if self.agent_id in traci.vehicle.getIDList():
            self.apply_acceleration(self.agent_id, a, smooth=False)

        self._update_head_speed()

        traci.simulationStep()
        time.sleep(0.01)  # to avoid busy waiting
        self.step_count += 1

        obs = self.get_state()
        # reward = self.compute_reward()
        reward = self.compute_lcc_reward()
        done = self.terminal()
        info = {}

        return obs, reward, done, info

    def compute_reward(self) -> float:
        if self.agent_id not in traci.vehicle.getIDList():
            return 0.0

        # Ego and lead info
        v_ego = traci.vehicle.getSpeed(self.agent_id)
        v_lead = 0.0
        d_gap = 1e9
        lead = traci.vehicle.getLeader(self.agent_id)
        if lead:
            lead_id, d_gap = lead
            v_lead = traci.vehicle.getSpeed(lead_id)
        else:
            try:
                ids = list(traci.vehicle.getIDList())
                pos = {vid: traci.vehicle.getDistance(vid) for vid in ids}
                pos_ego = pos[self.agent_id]
                deltas = [
                    (vid, (pos[vid] - pos_ego)) for vid in ids if vid != self.agent_id
                ]
                ahead = [(vid, d) for vid, d in deltas if d > 0]
                if ahead:
                    lead_id, d_gap = min(ahead, key=lambda x: x[1])
                elif deltas:
                    lead_id, d_gap = min(deltas, key=lambda x: abs(x[1]))
                else:
                    lead_id = None

                if lead_id is not None:
                    v_lead = float(traci.vehicle.getSpeed(lead_id))
                else:
                    if self.head_id in traci.vehicle.getIDList():
                        lead_id = self.head_id
                        v_lead = float(traci.vehicle.getSpeed(lead_id))
                        d_gap = float(
                            traci.vehicle.getDistance(lead_id)
                            - traci.vehicle.getDistance(self.agent_id)
                        )
            except traci.TraCIException:
                lead_id = None
                d_gap = 1e3
                v_lead = 0.0

        # print(f"Ego speed: {v_ego:.2f} m/s, Lead speed: {v_lead:.2f} m/s, Gap: {d_gap:.2f} m")
        # print(f"Lead ID: {lead_id}")

        # Safety guardrail (small)
        # TTC -> penalize being too close to the lead vehicle
        d_min = 2.0
        rel = max(v_ego - v_lead, 1e-3)
        free_gap = max(d_gap - d_min, 0.0)
        ttc = free_gap / rel if rel > 0 else 1e6
        tau_safe = 0.6  # aggressive but nonzero
        R_ttc = -0.02 * ((tau_safe - ttc) / tau_safe) if ttc < tau_safe else 0.0

        # headway distance -> penalize being too far
        # print(f"Distance to lead vehicle: {d_gap:.2f} m")
        gap_threshold = 15.0  # desired headway distance (m)
        r_d = -1.0 if d_gap > gap_threshold else 0.0

        # Penalize jerk (acceleration changes): Comfort penalty
        jerk = getattr(self, "last_jerk", 0.0)
        jerk_max = (self.max_accel - self.min_accel) / self.step_length
        norm_jerk = jerk / max(1e-6, jerk_max)
        R_jerk = -(norm_jerk**2)

        # Threshold for inefficiency (seconds)
        # TH = d_gap / max(v_ego, v_eps)
        # TH_threshold = 2.5
        # r_th = -1.0 if TH >= TH_threshold else 0.0

        # Meeting 11/12
        # - second reward: 10~15m distance to the lead vehicle
        # - Metrics for comparison
        # - Actor-critic method

        R = 0.15 * R_ttc + 1.0 * r_d + 0.2 * R_jerk
        return float(R)
    
    def compute_lcc_reward(self) -> float:
        """Compute reward based on DeeP-LCC cost function.

        The DeeP-LCC optimization minimizes ||y||²_Q + ||u||²_R where:
        - y contains velocity and spacing errors from equilibrium
        - u is the control input (acceleration)

        This translates to a reward (negative cost):
        R = -(weight_v * (v - v_star)² + weight_s * (s - s_star)² + weight_u * u²)

        Note: v_star is dynamically set to the head vehicle's current velocity,
        representing the equilibrium velocity the agent should track.

        Returns:
            float: Total reward computed
        """
        if self.agent_id not in traci.vehicle.getIDList():
            return 0.0

        # Get ego velocity
        v_ego = traci.vehicle.getSpeed(self.agent_id)

        # Get v_star from head vehicle's current velocity
        if self.head_id in traci.vehicle.getIDList():
            v_star = traci.vehicle.getSpeed(self.head_id)
        else:
            v_star = self.v_star  # Fallback to default if head vehicle not present
            
        # print(f"Speed of the Head Vehicle {self.head_id}: {v_star}")

        # Calculate s_star using OVM-type spacing policy
        # s_star = acos(1 - v_star/v_max * 2) / pi * (s_go - s_st) + s_st
        # where s_st = 5 (stop spacing), s_go = 35 (free-flow spacing)
        v_ratio = max(0.0, min(15 / self.v_max, 1.0))  # Clamp to [0, 1] for acos domain
        s_star = np.arccos(1 - v_ratio * 2) / np.pi * (35 - 5) + 5

        # Get gap to leader using existing logic
        d_gap = 1e9
        lead = traci.vehicle.getLeader(self.agent_id)
        if lead:
            _, d_gap = lead
        else:
            try:
                ids = list(traci.vehicle.getIDList())
                pos = {vid: traci.vehicle.getDistance(vid) for vid in ids}
                pos_ego = pos[self.agent_id]
                deltas = [
                    (vid, (pos[vid] - pos_ego)) for vid in ids if vid != self.agent_id
                ]
                ahead = [(vid, d) for vid, d in deltas if d > 0]
                if ahead:
                    _, d_gap = min(ahead, key=lambda x: x[1])
                elif deltas:
                    _, d_gap = min(deltas, key=lambda x: abs(x[1]))
            except traci.TraCIException:
                d_gap = 1e3
                
        # print(f"Gap between lead: {d_gap}")

        # Current acceleration (from previous action)
        accel = self.prev_accel

        # Quadratic penalties (negated for reward maximization)
        v_error = v_ego - 15
        s_error = np.clip(d_gap - s_star, -20.0, 20.0)

        R_velocity = -self.weight_v * (v_error**2)
        R_spacing = -self.weight_s * (s_error**2)
        R_control = -self.weight_u * (accel**2)

        R = R_velocity + R_spacing + R_control

        # Safety constraint as hard penalty
        if d_gap < self.spacing_min:
            R -= 100.0

        # Scale reward to keep per-step values in a PPO-friendly range
        # Worst case unscaled ≈ -560; after /100 → ≈ -5.6
        R /= 100.0

        return float(R)

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
        close_traci()
