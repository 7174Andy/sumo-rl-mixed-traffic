import numpy as np
import gymnasium as gym

from pathlib import Path
import os
import sys
from typing import Tuple
from dataclasses import dataclass

from utils import start_traci
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
            episode_length: float = 120.0
    ):
        self.sumo_config = sumo_config
        self.agent_id = agent_id
        self.gui = gui or sumo_config.use_gui
        self.dv = dv
        self.action_k = action_k
        self.actions = np.arange(-action_k, action_k + 1) * self.dv

        self.episode_length = episode_length
        self.step_length = sumo_config.step_length
        self.max_steps = int(episode_length / self.step_length)

        self.cmd_speed = 0.0
        self.v_max = 20.0

        # Discretizers
        self.gap_disc = Discretizer(low=0.0, high=60.0, step=5.0)
        self.v_disc = Discretizer(low=0.0, high=20.0, step=2.0)
        self.dV_disc = Discretizer(low=-10.0, high=10.0, step=2.0)

    def open_traci(self):
        if not traci.isLoaded():
            start_traci(SumoConfig(sumocfg_path=self.sumo_config.sumocfg_path, use_gui=self.gui, delay_ms=0))

    def close_traci(self):
        if traci.isLoaded():
            traci.close(False)
    
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
        # Apply action
        delta_v = self.actions[action]
        self.cmd_speed = float(np.clip(self.cmd_speed + delta_v, 0.0, self.v_max))

        if self.agent_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(self.agent_id, self.cmd_speed)
        
        traci.simulationStep()
        self.step_count += 1


        discrete_state = self.get_discrete_state()
        reward = self.compute_reward()
        done = self.terminal()

        return discrete_state, reward, done, {}
    
    def compute_reward(self) -> float:
        """
        Encourage high speeds with smoothness and safety:
        r = +w1 * (v_ego / v_max)  - w2 * jerk^2  - w3 * penalty_close
        where 'jerk' ~ change in command speed, 'penalty_close' if gap < threshold.
        """
        gap, v, _ = self._get_continuous_state()

        # normalize speed reward
        speed_term = (v / max(1e-6, self.v_max))

        # "jerk" ~ delta command speed this step (approx)
        # we don't store previous cmd explicitly; approximate with how much we nudged
        # keep it simple: penalize big |target - actual|
        target_diff = abs(self.cmd_speed - v)

        penalty_close = 1.0 if gap < 5.0 else 0.0

        w1, w2, w3 = 1.0, 0.03, 0.3
        r = +w1 * speed_term - w2 * (target_diff ** 2) - w3 * penalty_close
        return float(r)

    def terminal(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if traci.simulation.getMinExpectedNumber() == 0:
            return True
        return False

    def close(self):
        self.close_traci()