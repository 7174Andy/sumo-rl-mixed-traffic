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
    RL env that controls one or more agent vehicles' speed on a ring via setSpeed().

    Single-agent mode (num_agents=1, default):
        state = concat([v_norm...N], [p_norm...N]) of all vehicles, normalized.
        action = acceleration command (m/s²), shape (1,)
        step() returns (obs, reward, done, info) where obs is np.ndarray

    Multi-agent mode (num_agents > 1):
        Each agent sees the global state + its normalized agent index.
        action = np.ndarray of shape (num_agents,), one accel per agent.
        step() returns (obs_dict, reward, done, info) where obs_dict maps agent_id -> obs.
        All agents share a single scalar reward (sum of individual DeeP-LCC costs).

    Notes:
    - The episode runs for a fixed number of SUMO steps or until SUMO finishes.
    - 'car0' is the uncontrolled head vehicle. Agents are 'car1', 'car2', etc.
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
        num_agents: int = 1,
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
        self.gui = gui or sumo_config.use_gui
        self.max_accel = max_accel
        self.min_accel = min_accel
        self.num_vehicles = num_vehicles
        self.num_agents = num_agents

        # Multi-agent: derive agent IDs (car1, car2, ..., carN)
        # car0 is always the uncontrolled head vehicle
        if num_agents > 1:
            self.agent_ids = [f"car{i}" for i in range(1, num_agents + 1)]
        else:
            self.agent_ids = [agent_id]
        self.agent_id = self.agent_ids[0]  # backward-compat alias

        # Per-agent acceleration tracking
        self.prev_accels = {aid: 0.0 for aid in self.agent_ids}
        self.prev_accel = 0.0  # backward-compat alias for single-agent
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
        self.cmd_speeds = {aid: 0.0 for aid in self.agent_ids}
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

        # Cache for global state (refreshed each step)
        self._cached_global_state = None

    @property
    def action_space(self):
        if self.num_agents > 1:
            return Box(
                low=-self.max_accel, high=self.max_accel,
                shape=(self.num_agents,), dtype=np.float32,
            )
        return Box(
            low=-self.max_accel, high=self.max_accel, shape=(1,), dtype=np.float32
        )

    @property
    def observation_space(self):
        if self.num_agents > 1:
            # Global state + agent index
            return Box(
                low=0.0, high=1.0,
                shape=(2 * self.num_vehicles + 1,), dtype=np.float32,
            )
        # Single-agent: original behavior
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
        if self.head_vehicle_controller.update_every_step:
            self.head_vehicle_controller.set_random_head_speed()
            return

        if (
            self.step_count - self._last_head_update_step
        ) < self.head_speed_change_interval_steps:
            return

        self._last_head_update_step = self.step_count

        # sample a new random cruising speed
        self.head_vehicle_controller.set_random_head_speed()

    def get_state(self) -> np.ndarray:
        """Global observation = concat([v_norm...N], [p_norm...N]) padded/truncated to num_vehicles.
        Vehicles are sorted by ID so each index always maps to the same vehicle.
        v_norm = speed / v_max
        p_norm = position / ring_length

        Returns shape (2 * num_vehicles,). Does NOT include agent index.
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
        self._cached_global_state = obs
        return obs

    def get_obs_for_agent(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent: global state + normalized agent index.

        Args:
            agent_idx: Index into self.agent_ids (0-based).

        Returns:
            np.ndarray of shape (2 * num_vehicles + 1,)
        """
        if self._cached_global_state is None:
            self.get_state()

        global_state = self._cached_global_state
        # Normalized agent index: agent_idx / max(1, num_agents - 1)
        norm_idx = agent_idx / max(1, self.num_agents - 1)
        obs = np.append(global_state, np.float32(norm_idx))
        return obs

    def _get_gap_to_leader(self, veh_id: str) -> float:
        """Get the gap (meters) from veh_id to its leader on the ring.

        Args:
            veh_id: Vehicle ID to find leader gap for.

        Returns:
            Gap distance in meters (large value if no leader found).
        """
        d_gap = 1e9
        lead = traci.vehicle.getLeader(veh_id)
        if lead:
            _, d_gap = lead
        else:
            try:
                ids = list(traci.vehicle.getIDList())
                pos = {vid: traci.vehicle.getDistance(vid) for vid in ids}
                pos_ego = pos[veh_id]
                deltas = [
                    (vid, (pos[vid] - pos_ego)) for vid in ids if vid != veh_id
                ]
                ahead = [(vid, d) for vid, d in deltas if d > 0]
                if ahead:
                    lead_vid, d_gap = min(ahead, key=lambda x: x[1])
                    lead_length = traci.vehicle.getLength(lead_vid)
                    d_gap = d_gap - lead_length
                elif deltas:
                    lead_vid, d_gap = min(deltas, key=lambda x: abs(x[1]))
                    lead_length = traci.vehicle.getLength(lead_vid)
                    d_gap = d_gap - lead_length
            except traci.TraCIException:
                d_gap = 1e3
        return d_gap

    def reset(self, seed: int | None = None, options: dict = None):
        """Reset the environment and return the initial observation.

        Single-agent mode returns: (np.ndarray, dict)
        Multi-agent mode returns: (dict[str, np.ndarray], dict)
        """
        close_traci()
        self.open_traci()
        self.step_count = 0
        self.cmd_speed = 0.0
        self.prev_accel = 0.0
        self.last_jerk = 0.0
        self._last_head_update_step = 0
        self._cached_global_state = None
        self.head_vehicle_controller.reset()
        self.prev_accels = {aid: 0.0 for aid in self.agent_ids}
        self.cmd_speeds = {aid: 0.0 for aid in self.agent_ids}

        # Warm up the simulation until all agent vehicles have appeared
        warmup = 0
        while warmup < 200:
            all_present = all(
                aid in traci.vehicle.getIDList() for aid in self.agent_ids
            )
            if all_present:
                break
            traci.simulationStep()
            warmup += 1

        # Set speed mode and max speed for all agent vehicles
        for aid in self.agent_ids:
            if aid in traci.vehicle.getIDList():
                traci.vehicle.setSpeedMode(aid, 95)  # 0b1011111
                traci.vehicle.setMaxSpeed(aid, self.v_max)
                v_now = traci.vehicle.getSpeed(aid)
                self.cmd_speeds[aid] = max(0.5, min(v_now, self.v_max))

        # Backward compat: sync single-agent alias
        self.cmd_speed = self.cmd_speeds.get(self.agent_id, 0.0)

        if self.head_id in traci.vehicle.getIDList():
            traci.vehicle.setMaxSpeed(self.head_id, self.v_max)

        if self.num_agents > 1:
            # Multi-agent: return dict of per-agent observations
            self.get_state()  # populate cache
            obs_dict = {
                aid: self.get_obs_for_agent(i)
                for i, aid in enumerate(self.agent_ids)
            }
            return obs_dict, {}
        else:
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

        Single-agent mode:
            action: np.ndarray shape (1,) or int
            returns: (obs, reward, done, info)

        Multi-agent mode:
            action: np.ndarray shape (num_agents,)
            returns: (obs_dict, reward, done, info) where obs_dict maps agent_id -> obs
        """
        if self.num_agents > 1:
            return self._step_multi(action)
        else:
            return self._step_single(action)

    def _step_single(self, action: np.ndarray | int):
        """Single-agent step (backward-compatible with original interface)."""
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
        self.prev_accels[self.agent_id] = a

        if self.agent_id in traci.vehicle.getIDList():
            self.apply_acceleration(self.agent_id, a, smooth=False)

        self._update_head_speed()

        traci.simulationStep()
        self.step_count += 1

        self._cached_global_state = None  # invalidate cache
        obs = self.get_state()
        reward = self.compute_lcc_reward()
        done = self.terminal()
        info = {}

        return obs, reward, done, info

    def _step_multi(self, action: np.ndarray):
        """Multi-agent step: action is shape (num_agents,), one accel per agent.

        Returns:
            (obs_dict, reward, done, info) where:
            - obs_dict: dict mapping agent_id -> np.ndarray of shape (2*num_vehicles+1,)
            - reward: float, shared scalar reward (sum of all agents' costs)
            - done: bool
            - info: dict
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.num_agents)

        active_ids = traci.vehicle.getIDList()

        # Track per-agent accelerations and apply
        accels_to_apply = []
        ids_to_apply = []
        for i, aid in enumerate(self.agent_ids):
            a = float(np.clip(action[i], self.min_accel, self.max_accel))
            self.prev_accels[aid] = a
            if aid in active_ids:
                ids_to_apply.append(aid)
                accels_to_apply.append(a)

        if ids_to_apply:
            self.apply_acceleration(ids_to_apply, accels_to_apply, smooth=False)

        self._update_head_speed()

        traci.simulationStep()
        self.step_count += 1

        self._cached_global_state = None  # invalidate cache
        self.get_state()  # populate cache

        obs_dict = {
            aid: self.get_obs_for_agent(i)
            for i, aid in enumerate(self.agent_ids)
        }

        reward = self.compute_multi_agent_lcc_reward()
        done = self.terminal()
        info = {}

        return obs_dict, reward, done, info

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

        # Safety guardrail (small)
        # TTC -> penalize being too close to the lead vehicle
        d_min = 2.0
        rel = max(v_ego - v_lead, 1e-3)
        free_gap = max(d_gap - d_min, 0.0)
        ttc = free_gap / rel if rel > 0 else 1e6
        tau_safe = 0.6  # aggressive but nonzero
        R_ttc = -0.02 * ((tau_safe - ttc) / tau_safe) if ttc < tau_safe else 0.0

        # headway distance -> penalize being too far
        gap_threshold = 15.0  # desired headway distance (m)
        r_d = -1.0 if d_gap > gap_threshold else 0.0

        # Penalize jerk (acceleration changes): Comfort penalty
        jerk = getattr(self, "last_jerk", 0.0)
        jerk_max = (self.max_accel - self.min_accel) / self.step_length
        norm_jerk = jerk / max(1e-6, jerk_max)
        R_jerk = -(norm_jerk**2)

        R = 0.15 * R_ttc + 1.0 * r_d + 0.2 * R_jerk
        return float(R)

    def compute_lcc_reward(self) -> float:
        """Compute system-level reward based on DeeP-LCC cost function (single-agent).

        Following the DeeP-LCC paper, the cost is a single system-level objective:
            J(y, u) = y^T Q y + u^T R u
        where y = [velocity errors of ALL vehicles; spacing errors of CAVs]
        and u = [accelerations of CAVs].

        The velocity error penalty sums over ALL vehicles (HDVs + CAVs),
        not just the agent. This incentivizes the CAV to smooth traffic flow
        for the entire platoon.

        Returns:
            float: System-level reward (negative cost, scaled)
        """
        active_ids = traci.vehicle.getIDList()

        if self.agent_id not in active_ids:
            return 0.0

        # Equilibrium velocity
        v_eq = self.v_star

        # Calculate s_star using OVM-type spacing policy
        v_ratio = max(0.0, min(v_eq / self.v_max, 1.0))
        s_star = np.arccos(1 - v_ratio * 2) / np.pi * (35 - 5) + 5

        # --- System-level velocity error: ALL vehicles except head ---
        R_velocity = 0.0
        for vid in active_ids:
            if vid == self.head_id:
                continue
            v = traci.vehicle.getSpeed(vid)
            R_velocity -= self.weight_v * (v - v_eq) ** 2

        # --- Spacing error: CAV (agent) only ---
        d_gap = self._get_gap_to_leader(self.agent_id)
        s_error = np.clip(d_gap - s_star, -20.0, 20.0)
        R_spacing = -self.weight_s * (s_error ** 2)

        # --- Control penalty: CAV (agent) only ---
        accel = self.prev_accel
        R_control = -self.weight_u * (accel ** 2)

        R = R_velocity + R_spacing + R_control

        # Safety constraint as hard penalty
        if d_gap < self.spacing_min:
            R -= 100.0

        # Scale reward to keep per-step values in a PPO-friendly range
        R /= 100.0

        return float(R)

    def compute_multi_agent_lcc_reward(self) -> float:
        """Compute single system-level reward following DeeP-LCC (multi-agent).

        The DeeP-LCC cost is a system-level objective:
            J(y, u) = y^T Q y + u^T R u
        where y = [velocity errors of ALL vehicles; spacing errors of CAVs]
        and u = [accelerations of CAVs].

        Components:
        - Velocity error: summed over ALL vehicles (HDVs + CAVs), excluding head
        - Spacing error: summed over CAVs only
        - Control penalty: summed over CAVs only
        - Safety penalty: per CAV if gap < spacing_min

        Returns:
            float: Single shared system-level reward (scaled by /100)
        """
        active_ids = traci.vehicle.getIDList()

        # Equilibrium velocity
        v_eq = self.v_star

        # Compute s_star using OVM-type spacing policy
        v_ratio = max(0.0, min(v_eq / self.v_max, 1.0))
        s_star = np.arccos(1 - v_ratio * 2) / np.pi * (35 - 5) + 5

        # --- System-level velocity error: ALL vehicles except head ---
        R_velocity = 0.0
        for vid in active_ids:
            if vid == self.head_id:
                continue
            v = traci.vehicle.getSpeed(vid)
            R_velocity -= self.weight_v * (v - v_eq) ** 2

        # --- Spacing error + control penalty: CAVs only ---
        R_spacing = 0.0
        R_control = 0.0
        R_safety = 0.0

        for aid in self.agent_ids:
            if aid not in active_ids:
                continue

            d_gap = self._get_gap_to_leader(aid)
            s_error = np.clip(d_gap - s_star, -20.0, 20.0)
            R_spacing -= self.weight_s * (s_error ** 2)

            accel = self.prev_accels[aid]
            R_control -= self.weight_u * (accel ** 2)

            if d_gap < self.spacing_min:
                R_safety -= 100.0

        R_total = R_velocity + R_spacing + R_control + R_safety

        # Scale reward to keep per-step values in a PPO-friendly range
        R_total /= 100.0

        return float(R_total)

    def terminal(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if traci.simulation.getMinExpectedNumber() == 0:
            return True
        # When vehicles collide, SUMO ends the simulation
        if traci.simulation.getCollidingVehiclesNumber() > 0:
            return True
        return False

    def close(self):
        close_traci()
