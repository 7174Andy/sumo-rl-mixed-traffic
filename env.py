import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pathlib import Path
import os
import sys
from typing import Tuple
from dataclasses import dataclass

from utils.sumo_utils import start_traci
from config import SumoConfig

if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "share" / "sumo" / "tools"
    sys.path.append(str(tools))
else:
    raise EnvironmentError("Please set the SUMO_HOME environment variable to your SUMO install path.")

import traci

@dataclass
class Discretizer:
    """Uniform binning for a scalar."""
    low: float
    high: float
    step: float

    @property
    def bins(self) -> np.ndarray:
        # rightmost included by adding +1e-9
        return np.arange(self.low, self.high + 1e-9, self.step)

    def index(self, x: float) -> int:
        x_clipped = np.clip(x, self.low, self.high)
        return int(np.digitize(x_clipped, self.bins) - 1)  # 0-based bin index

class RingRoadEnv(gym.Env):
    """
    Minimal RL env that controls one agent vehicle's speed on a ring via setSpeed().
    State = (gap_to_leader, ego_speed, rel_speed) all discretized.
    Action = delta to commanded speed in { -dv*k, ..., 0, ..., +dv*k }.

    Notes:
    - The episode runs for a fixed number of SUMO steps or until SUMO finishes.
    - We assume 'car0' exists in routes and departs early.
    """
    def __init__(
            self,
            sumo_config: SumoConfig,
            agent_id: str='car0',
            gui: bool=False,
            dv: float = 0.4,
            action_k: int = 3,
            episode_length: float = 200.0,
            safety_distance: float = 100.0
    ):
        self.sumo_config = sumo_config
        self.agent_id = agent_id
        self.gui = gui or sumo_config.use_gui
        self.dv = dv
        self.action_k = action_k
        self.actions = np.arange(-action_k, action_k + 1) * self.dv
        self.safety_distance = safety_distance

        self.episode_length = episode_length
        self.step_length = sumo_config.step_length
        self.max_steps = int(episode_length / self.step_length)

        # Discretizers
        self.gap_disc = Discretizer(low=0.0, high=60.0, step=5.0)
        self.v_disc = Discretizer(low=0.0, high=20.0, step=2.0)
        self.dV_disc = Discretizer(low=-10.0, high=10.0, step=2.0)

        self.action_space = spaces.Discrete(len(self.actions))

        n_gap = len(self.gap_disc.bins) - 1
        n_v   = len(self.v_disc.bins)   - 1
        n_dv  = len(self.dV_disc.bins)  - 1
        self.observation_space = spaces.MultiDiscrete([n_gap, n_v, n_dv])

        self.cmd_speed = 0.0
        self.v_max = 20.0

    def open_traci(self):
        if not traci.isLoaded():
            start_traci(SumoConfig(sumocfg_path=self.sumo_config.sumocfg_path, use_gui=self.gui, delay_ms=0))

    def close_traci(self):
        if traci.isLoaded():
            traci.close(False)

    def get_followers_chain(self, leader_id: str, depth: int = 5):
        """
        Return [(follower_id, gap_back), ...] up to `depth` behind `leader_id`.
        Safe against missing IDs and arrival events.
        """
        chain = []
        if not leader_id or leader_id not in traci.vehicle.getIDList():
            return chain

        current = leader_id
        for _ in range(depth):
            try:
                info = traci.vehicle.getFollower(current)  # -> (vehID, gap) or None
            except traci.TraCIException:
                break

            if not info:
                break

            fid, gap = info
            if not fid or fid not in traci.vehicle.getIDList():
                break

            chain.append((fid, gap))
            current = fid
        return chain
    
    def _get_continuous_state(self) -> Tuple[float, float, float]:
        """
        Returns (gap_to_leader, v_ego, dV) in continuous values.
        If no leader: use a large gap and zero rel speed.
        """
        if self.agent_id not in traci.vehicle.getIDList():
            return 0.0, 0.0, 0.0

        v_ego = traci.vehicle.getSpeed(self.agent_id)
        leader = traci.vehicle.getLeader(self.agent_id)

        if leader is None:
            gap = 60.0
            dV = 0.0
        else:
            leader_id, gap = leader
            v_lead = traci.vehicle.getSpeed(leader_id) if leader_id in traci.vehicle.getIDList() else v_ego
            dV = v_ego - v_lead  # positive if faster than leader

            # clamp unreasonable gaps
            gap = float(np.clip(gap, 0.0, 60.0))

        return gap, v_ego, dV

    def get_discrete_state(self) -> Tuple[int, int, int]:
        gap, v, dV = self._get_continuous_state()
        return (
            self.gap_disc.index(gap),
            self.v_disc.index(v),
            self.dV_disc.index(dV),
        )
    
    def reset(self):
        self.close_traci()
        self.open_traci()
        self.step_count = 0
        self.cmd_speed = 0.0
        self.v_max = 20.0

        # Warm up the simulation
        warmup = 0
        while self.agent_id not in traci.vehicle.getIDList() and warmup < 200:
            traci.simulationStep()
            warmup += 1

        if self.agent_id in traci.vehicle.getIDList():
            self.v_max = traci.vehicle.getMaxSpeed(self.agent_id)
            v_now = traci.vehicle.getSpeed(self.agent_id)
            self.cmd_speed = max(0.5, min(v_now, self.v_max))

        return self.get_discrete_state()
    
    def render(self):
        return None

    def step(self, action: int):
        """Apply the given action and return the new state, reward, and done flag.

        Args:
            action (int): The action to apply.

        Returns:
            Tuple[int, float, bool]: The new state, reward, and done flag.
        """
        # Sanity check
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Apply action
        delta_v = self.actions[action]
        self.cmd_speed = float(np.clip(self.cmd_speed + delta_v, 0.0, self.v_max))

        if self.agent_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(self.agent_id, self.cmd_speed)
        
        traci.simulationStep()
        # time.sleep(0.01)  # to avoid busy waiting
        self.step_count += 1


        discrete_state = self.get_discrete_state()
        reward = self.compute_reward()
        done = self.terminal()

        return discrete_state, reward, done, {}

    def compute_reward(self) -> float:
        """
        Network-speed heavy reward:
        + Main: mean speed of first K followers (fallback to ego speed if none)
        - Small comfort penalty (command mismatch)
        - Light safety penalty (very small gaps or low TTC behind leader)
        """
        if self.agent_id not in traci.vehicle.getIDList():
            return 0.0

        # configurable depth (followers considered)
        followers_depth = 5

        # ---- platoon speed term (dominant) ----
        chain = self.get_followers_chain(leader_id=self.agent_id, depth=followers_depth)
        f_ids = [fid for fid, _ in chain]

        v0 = traci.vehicle.getSpeed(self.agent_id)
        vmax0 = max(1e-6, self.v_max)

        if f_ids:
            speeds = [traci.vehicle.getSpeed(i) for i in f_ids]
            mean_v = float(np.mean(speeds))
        else:
            # no followers yet -> fall back to ego speed
            mean_v = v0

        platoon_speed_term = mean_v / vmax0  # ~[0,1]

        # ---- comfort penalty (very light) ----
        target_diff = abs(self.cmd_speed - v0)  # proxy for jerk
        comfort_pen = (target_diff ** 2)

        # ---- light safety penalty wrt direct follower ----
        safety_pen = 0.0
        try:
            info = traci.vehicle.getFollower(self.agent_id)  # (fid, gap_back) or None
        except traci.TraCIException:
            info = None

        if info:
            fid, gap_back = info
            if fid and fid in traci.vehicle.getIDList():
                v_f = traci.vehicle.getSpeed(fid)
                rel_close = v_f - v0  # follower closing (+) onto leader
                # small-gap penalty (soft)
                if gap_back < 5.0:
                    safety_pen += 0.3
                # TTC penalty if the follower is rapidly closing
                if rel_close > 0.1:
                    ttc = gap_back / max(1e-6, rel_close)
                    if ttc < 1.5:
                        safety_pen += 0.3

        # ---- weights: SPEED >> comfort & safety ----
        w_speed, w_comfort, w_safety = 1.0, 0.02, 0.10
        r = (w_speed * platoon_speed_term) - (w_comfort * comfort_pen) - (w_safety * safety_pen)
        return float(r)


    def terminal(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if traci.simulation.getMinExpectedNumber() == 0:
            return True
        return False

    def close(self):
        self.close_traci()