"""PlatoonNNMPCEnv — SUMO-based gymnasium env for RLMPC training.

Subclasses RingRoadEnv to reuse SUMO lifecycle, leader-finding, multi-agent
reward (`compute_multi_agent_lcc_reward`), and Lagrangian spacing-violation
helper. Adds NNMPC-shaped observation, mode-aware action box, NNMPC inside
step() for residual mode, and Lagrangian-augmented reward with collision = -1.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ----------------------------------------------------------------------
# Observation buffer (testable in isolation, no SUMO)
# ----------------------------------------------------------------------


class ObservationBuilder:
    """Maintains rolling uini/yini/eini buffers for the NNMPC observation.

    Buffers are filled by push_step() from current SUMO state each step.
    build_obs() returns the concatenated 260-dim vector (T_ini × m_ctr +
    T_ini × p_ctr + T_ini); build_normalized_obs() applies the provided
    NNMPC training normalization stats.

    Layout (matches NNMPC training input):
        obs = concat(uini, yini, eini)
            uini.shape = (T_ini * m_ctr,)
            yini.shape = (T_ini * p_ctr,)  where p_ctr = n_followers + m_ctr
            eini.shape = (T_ini,)
    """

    def __init__(self, T_ini: int, n_followers: int, m_ctr: int):
        self.T_ini = T_ini
        self.n_followers = n_followers
        self.m_ctr = m_ctr
        self.p_ctr = n_followers + m_ctr

        self.uini: deque = deque(maxlen=T_ini * m_ctr)
        self.yini: deque = deque(maxlen=T_ini * self.p_ctr)
        self.eini: deque = deque(maxlen=T_ini)

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def total_dim(self) -> int:
        return self.T_ini * (self.m_ctr + self.p_ctr + 1)

    def push_step(
        self,
        uini_step: np.ndarray,
        yini_step: np.ndarray,
        eini_step: np.ndarray,
    ) -> None:
        if uini_step.shape != (self.m_ctr,):
            raise ValueError(f"uini_step shape {uini_step.shape} != ({self.m_ctr},)")
        if yini_step.shape != (self.p_ctr,):
            raise ValueError(f"yini_step shape {yini_step.shape} != ({self.p_ctr},)")
        if eini_step.shape != (1,):
            raise ValueError(f"eini_step shape {eini_step.shape} != (1,)")
        self.uini.extend(uini_step.tolist())
        self.yini.extend(yini_step.tolist())
        self.eini.extend(eini_step.tolist())

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        if mean.shape != (self.total_dim,):
            raise ValueError(f"mean shape {mean.shape} != ({self.total_dim},)")
        if std.shape != (self.total_dim,):
            raise ValueError(f"std shape {std.shape} != ({self.total_dim},)")
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)

    def _padded(self, buf: deque, target_len: int) -> np.ndarray:
        out = np.zeros(target_len, dtype=np.float64)
        if len(buf) > 0:
            arr = np.fromiter(buf, dtype=np.float64, count=len(buf))
            out[-len(arr):] = arr
        return out

    def build_obs(self) -> np.ndarray:
        u = self._padded(self.uini, self.T_ini * self.m_ctr)
        y = self._padded(self.yini, self.T_ini * self.p_ctr)
        e = self._padded(self.eini, self.T_ini)
        return np.concatenate([u, y, e]).astype(np.float32)

    def build_normalized_obs(self) -> np.ndarray:
        obs = self.build_obs()
        if self._mean is None or self._std is None:
            return obs
        return ((obs - self._mean) / np.maximum(self._std, 1e-6)).astype(np.float32)

    def reset(self) -> None:
        self.uini.clear()
        self.yini.clear()
        self.eini.clear()


import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.spaces import Box

from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.deep_lcc.measurement import measure_mixed_traffic
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_head_controller import (
    PerturbMixHeadController,
)
from rl_mixed_traffic.env.ring_env import RingRoadEnv

if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "share" / "sumo" / "tools"
    sys.path.append(str(tools))

import traci  # noqa: E402  (must come after SUMO_HOME path setup)


class PlatoonNNMPCEnv(RingRoadEnv):
    """SUMO-based env for RLMPC training (warm-start RL or RL+NNMPC residual).

    Subclass of RingRoadEnv: reuses SUMO lifecycle, leader/spacing helpers,
    `compute_multi_agent_lcc_reward`, and `get_spacing_violation`. Overrides
    observation_space, action_space, step(), and reset() to produce the
    NNMPC-shaped observation and the mode-aware action.

    Action handling:
        - Policy action ∈ [-1, 1]^2 (tanh-squashed Gaussian sample)
        - Scaled to action_space box (mode-dependent)
        - For residual: total = clip(NNMPC + scaled, [accel_min, accel_max])
        - For warm_start: total = scaled (already physical-range)
    """

    def __init__(self, config: RLMPCConfig):
        sumo_config = SumoConfig(
            sumocfg_path=config.sumocfg_path,
            use_gui=config.use_gui,
            step_length=config.Tstep,
        )
        head_controller = PerturbMixHeadController(
            head_id="car0",
            tstep=config.Tstep,
            episode_length_s=config.episode_length_s,
            v_star=config.v_star,
            seed=config.seed,
        )
        super().__init__(
            sumo_config=sumo_config,
            agent_id="car3",
            gui=config.use_gui,
            max_accel=config.accel_max,
            min_accel=config.accel_min,
            episode_length=config.episode_length_s,
            num_vehicles=config.n_vehicle,
            num_agents=config.m_ctr,
            head_vehicle_controller=head_controller,
            v_star=config.v_star,
            s_star=config.s_star,
            weight_v=config.weight_v,
            weight_s=config.weight_s,
            weight_u=config.weight_u,
            spacing_min=config.spacing_min,
            spacing_max=config.spacing_max,
        )
        self.agent_ids = [f"car{p}" for p in config.cav_positions]
        self.agent_id = self.agent_ids[0]
        self.prev_accels = {aid: 0.0 for aid in self.agent_ids}
        self.cmd_speeds = {aid: 0.0 for aid in self.agent_ids}

        self.config = config
        self._cav_indices_in_platoon = [p - 1 for p in config.cav_positions]
        self._ID = [
            1 if (i + 1) in config.cav_positions else 0
            for i in range(config.n_followers)
        ]
        self._follower_ids = [f"car{i}" for i in range(1, config.n_followers + 1)]

        self._builder = ObservationBuilder(
            T_ini=config.T_ini,
            n_followers=config.n_followers,
            m_ctr=config.m_ctr,
        )

        self._nnmpc, mean, std = self._load_nnmpc(config.nnmpc_path)
        self._builder.set_normalization(mean, std)

        if config.mode == "warm_start":
            low = np.full(config.action_dim, config.accel_min, dtype=np.float32)
            high = np.full(config.action_dim, config.accel_max, dtype=np.float32)
        else:
            low = np.full(config.action_dim, -config.residual_max, dtype=np.float32)
            high = np.full(config.action_dim, +config.residual_max, dtype=np.float32)
        self._action_space = Box(low=low, high=high, shape=(config.action_dim,),
                                  dtype=np.float32)
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(config.obs_dim,), dtype=np.float32,
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @staticmethod
    def _load_nnmpc(path: str) -> tuple[NNMPCNetwork, np.ndarray, np.ndarray]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = NNMPCNetwork(
            input_dim=ckpt["input_dim"],
            output_dim=ckpt["output_dim"],
            hidden_dims=ckpt["config"]["hidden_dims"],
            accel_min=ckpt["config"]["accel_min"],
            accel_max=ckpt["config"]["accel_max"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt["input_mean"], ckpt["input_std"]

    def _compose_action(
        self,
        policy_action: np.ndarray,
        u_nnmpc: np.ndarray,
    ) -> np.ndarray:
        """Combine the policy output with NNMPC's suggestion and clip to physical range.

        policy_action is already in the per-mode action box. For residual, we add
        NNMPC and clip; for warm_start, the policy IS the command (NNMPC bypassed).
        """
        if self.config.mode == "residual":
            total = u_nnmpc + policy_action
        else:
            total = policy_action
        return np.clip(total, self.config.accel_min, self.config.accel_max)

    def _augment_reward(
        self, r_base: float, violation: float, collision: bool
    ) -> float:
        if collision:
            return -1.0
        return float(r_base - self.config.lambda_violation * violation)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options  # gymnasium signature requires it; we don't use it
        from rl_mixed_traffic.utils.sumo_utils import close_traci, start_traci

        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        close_traci()
        start_traci(SumoConfig(
            sumocfg_path=self.config.sumocfg_path,
            use_gui=self.config.use_gui,
            step_length=self.config.Tstep,
        ))

        self.step_count = 0
        self._cached_global_state = None
        self._head_speed_buffer.clear()
        self._builder.reset()
        self.prev_accels = {aid: 0.0 for aid in self.agent_ids}
        self.cmd_speeds = {aid: 0.0 for aid in self.agent_ids}
        self.head_vehicle_controller.reset()

        expected = ["car0"] + self._follower_ids
        warmup = 0
        while warmup < 200:
            present = all(v in traci.vehicle.getIDList() for v in expected)
            if present:
                break
            traci.simulationStep()
            warmup += 1

        hdv_ids = [
            f"car{i}" for i in range(1, self.config.n_followers + 1)
            if i not in self.config.cav_positions
        ]
        for hid in hdv_ids:
            if hid in traci.vehicle.getIDList():
                tau = float(rng.uniform(*self.config.hdv_tau_range))
                accel = float(rng.uniform(*self.config.hdv_accel_range))
                decel = float(rng.uniform(*self.config.hdv_decel_range))
                gap = float(rng.uniform(*self.config.hdv_minGap_range))
                sigma = float(rng.uniform(*self.config.hdv_sigma_range))
                traci.vehicle.setTau(hid, tau)
                traci.vehicle.setAccel(hid, accel)
                traci.vehicle.setDecel(hid, decel)
                traci.vehicle.setMinGap(hid, gap)
                traci.vehicle.setImperfection(hid, sigma)

        for _ in range(self.config.T_ini):
            self._step_for_warmup()

        for aid in self.agent_ids:
            if aid in traci.vehicle.getIDList():
                traci.vehicle.setSpeedMode(aid, 95)
                traci.vehicle.setMaxSpeed(aid, self.config.v_max)
                self.cmd_speeds[aid] = float(traci.vehicle.getSpeed(aid))

        obs = self._builder.build_normalized_obs()
        return obs, {}

    def _step_for_warmup(self) -> None:
        """One simulation step during warm-up; record observation buffers."""
        self.head_vehicle_controller.set_random_head_speed()
        traci.simulationStep()
        self.step_count += 1
        self._head_speed_buffer.append(
            traci.vehicle.getSpeed("car0")
            if "car0" in traci.vehicle.getIDList() else self.config.v_star
        )
        self._record_step_to_buffers()

    def _record_step_to_buffers(self) -> None:
        """Read SUMO state and push (uini, yini, eini) onto the rolling buffers."""
        active = traci.vehicle.getIDList()
        uini = np.array([
            traci.vehicle.getAcceleration(aid) if aid in active else 0.0
            for aid in self.agent_ids
        ], dtype=np.float32)

        n = self.config.n_followers
        vel = np.zeros(n, dtype=np.float32)
        pos = np.zeros(n + 1, dtype=np.float32)
        pos[0] = traci.vehicle.getDistance("car0") if "car0" in active else 0.0
        for i, fid in enumerate(self._follower_ids):
            if fid in active:
                vel[i] = traci.vehicle.getSpeed(fid)
                pos[i + 1] = traci.vehicle.getDistance(fid)
            else:
                vel[i] = 0.0
                pos[i + 1] = pos[i] - self.config.s_star

        v_eq = self.v_eq if self._head_speed_buffer else self.config.v_star
        s_star = self.config.s_star

        yini = measure_mixed_traffic(
            vel=vel, pos=pos, ID=self._ID,
            v_star=v_eq, s_star=s_star, measure_type=3,
        ).astype(np.float32)

        v_head = (
            traci.vehicle.getSpeed("car0")
            if "car0" in active else self.config.v_star
        )
        eini = np.array([v_head - v_eq], dtype=np.float32)

        self._builder.push_step(uini, yini, eini)

    def step(self, action: np.ndarray):  # ty: ignore[invalid-method-override]
        # Returns gymnasium 5-tuple; parent RingRoadEnv returns legacy 4-tuple.
        # FourToFiveTupleWrapper bridges the gap when the env is wrapped at train/eval time.
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if self.config.mode == "residual":
            obs_norm = self._builder.build_normalized_obs()
            with torch.no_grad():
                u_nnmpc = self._nnmpc(
                    torch.from_numpy(obs_norm).unsqueeze(0)
                ).cpu().numpy().ravel()
        else:
            u_nnmpc = np.zeros(self.config.action_dim, dtype=np.float32)

        u_total = self._compose_action(action, u_nnmpc)

        for aid, a in zip(self.agent_ids, u_total):
            if aid in traci.vehicle.getIDList():
                self.apply_acceleration(aid, float(a), smooth=False)
            self.prev_accels[aid] = float(a)

        self.head_vehicle_controller.set_random_head_speed()

        traci.simulationStep()
        self.step_count += 1

        if "car0" in traci.vehicle.getIDList():
            self._head_speed_buffer.append(traci.vehicle.getSpeed("car0"))

        self._cached_global_state = None
        self._record_step_to_buffers()

        obs = self._builder.build_normalized_obs()
        r_base = self.compute_multi_agent_lcc_reward()
        violation = self.get_spacing_violation()
        collision = traci.simulation.getCollidingVehiclesNumber() > 0
        reward = self._augment_reward(r_base, violation, collision)

        terminated = collision
        truncated = (self.step_count >= self.max_steps) or \
                    (traci.simulation.getMinExpectedNumber() == 0)
        info = {
            "r_base": float(r_base),
            "violation": float(violation),
            "collision": bool(collision),
            "u_nnmpc": u_nnmpc.tolist(),
            "u_total": u_total.tolist(),
        }
        return obs, reward, terminated, truncated, info
