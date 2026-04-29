# RLMPC Warm-Start RL + RL+NNMPC Residual — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two RL-augmented controllers (Warm-Start RL and RL+NNMPC residual) on top of the existing NNMPC, training in SUMO with the project's existing PPO + Lagrangian-augmented LCC reward.

**Architecture:** New `PlatoonNNMPCEnv` (subclass of `RingRoadEnv`) exposes a 260-dim NNMPC-shaped observation and a 2-D joint CAV-acceleration action; runs NNMPC inside `step()` for residual mode. New `NNMPCActorCritic` network matches `NNMPCNetwork`'s architecture so warm-starting is a direct `state_dict` copy; plugs into the existing `PPOAgent` unchanged. New SUMO config (`platoon_9`) with 1 head + 8 followers (CAVs at positions 3, 6).

**Tech Stack:** Python 3.13, PyTorch, gymnasium, SUMO/TraCI, numpy. Reuses `rl_mixed_traffic/agents/ppo_agent.py`, `rl_mixed_traffic/ppo/{network,rollout_buffer}.py`, `rl_mixed_traffic/env/ring_env.py`, `rl_mixed_traffic/deep_lcc/{nnmpc_network,measurement,config}.py`.

**Spec:** `docs/superpowers/specs/2026-04-27-rlmpc-warm-rl-residual-design.md`

---

## File map

**New files:**

| Path | Responsibility |
|---|---|
| `configs/ring/platoon_9.rou.xml` | 1 head (`car0`, IDM) + 8 followers (`car1..car8`, IDM, randomizable) |
| `configs/ring/platoon_9.sumocfg` | SUMO config with `step-length=0.05` |
| `rl_mixed_traffic/deep_lcc/rlmpc_config.py` | `RLMPCConfig` dataclass (mode, hyperparams, HDV ranges) |
| `rl_mixed_traffic/deep_lcc/rlmpc_head_controller.py` | `PerturbMixHeadController` — pre-computes head velocity trace from `DeepLCCConfig.perturb_mix` |
| `rl_mixed_traffic/deep_lcc/nnmpc_actor_critic.py` | `NNMPCActorCritic` network — NNMPC-shaped actor + critic; warm-start hook |
| `rl_mixed_traffic/deep_lcc/rlmpc_env.py` | `PlatoonNNMPCEnv` (subclass of `RingRoadEnv`) — 260-dim obs, 2-D action, NNMPC inside step, Lagrangian reward |
| `rl_mixed_traffic/deep_lcc/rlmpc_train.py` | Training script — wires env + network + `PPOAgent` |
| `rl_mixed_traffic/deep_lcc/rlmpc_eval.py` | Eval script — runs QP, NNMPC, Warm-Start RL, RL+NNMPC across scenario × HDV grid |
| `tests/test_rlmpc_head_controller.py` | Tests for head controller perturb-profile generation |
| `tests/test_nnmpc_actor_critic.py` | Tests for warm-start fidelity, residual zero-init, distribution math |
| `tests/test_rlmpc_env.py` | Tests for buffer construction, observation shape, action scaling, reward augmentation (mock traci) |

**Files reused (no edits):** `rl_mixed_traffic/agents/ppo_agent.py`, `rl_mixed_traffic/ppo/{network,rollout_buffer}.py`, `rl_mixed_traffic/env/ring_env.py`, `rl_mixed_traffic/env/wrappers.py`, `rl_mixed_traffic/env/head_vehicle_controller.py`, `rl_mixed_traffic/deep_lcc/nnmpc_network.py`, `rl_mixed_traffic/deep_lcc/nnmpc_eval.py` (eval helpers), `rl_mixed_traffic/deep_lcc/eval_classical.py` (`compute_metrics`, `make_*` head profiles).

---

## Task 1: SUMO route + config for the 9-vehicle platoon

**Files:**
- Create: `configs/ring/platoon_9.rou.xml`
- Create: `configs/ring/platoon_9.sumocfg`

- [ ] **Step 1.1: Create the route file.**

Write `configs/ring/platoon_9.rou.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <route id="r_0" edges="B9B8 B8B7" repeat="200"/>
    <vType id="idm" carFollowModel="IDM" accel="2.0" decel="3.0"
           tau="1.0" sigma="0.5" minGap="2.5" length="5.0" maxSpeed="30"/>
    <!-- car0 = head (uncontrolled), car1..car8 = followers; CAVs at car3, car6 -->
    <vehicle id="car0" depart="0.00" route="r_0" color="0,0,255"     type="idm" departSpeed="10"/>
    <vehicle id="car1" depart="0.00" route="r_0" color="200,200,200" type="idm" departSpeed="10"/>
    <vehicle id="car2" depart="0.00" route="r_0" color="200,200,200" type="idm" departSpeed="10"/>
    <vehicle id="car3" depart="0.00" route="r_0" color="0,200,0"     type="idm" departSpeed="10"/>
    <vehicle id="car4" depart="0.00" route="r_0" color="200,200,200" type="idm" departSpeed="10"/>
    <vehicle id="car5" depart="0.00" route="r_0" color="200,200,200" type="idm" departSpeed="10"/>
    <vehicle id="car6" depart="0.00" route="r_0" color="0,200,0"     type="idm" departSpeed="10"/>
    <vehicle id="car7" depart="0.00" route="r_0" color="200,200,200" type="idm" departSpeed="10"/>
    <vehicle id="car8" depart="0.00" route="r_0" color="200,200,200" type="idm" departSpeed="10"/>
</routes>
```

- [ ] **Step 1.2: Create the sumocfg file.**

Write `configs/ring/platoon_9.sumocfg`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="circle.net.xml"/>
        <route-files value="platoon_9.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="600"/>
        <step-length value="0.05"/>
    </time>
    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>
</sumoConfiguration>
```

- [ ] **Step 1.3: Smoke-test the SUMO config.**

Run: `sumo -c configs/ring/platoon_9.sumocfg --no-step-log --duration-log.disable --end 5`
Expected: completes without error; produces no output beyond standard SUMO banner.

- [ ] **Step 1.4: Commit.**

```bash
git add configs/ring/platoon_9.rou.xml configs/ring/platoon_9.sumocfg
git commit -m "feat: add 9-vehicle platoon SUMO config (1 head + 8 followers, step=0.05)"
```

---

## Task 2: `RLMPCConfig` dataclass

**Files:**
- Create: `rl_mixed_traffic/deep_lcc/rlmpc_config.py`

- [ ] **Step 2.1: Write the config module.**

Write `rl_mixed_traffic/deep_lcc/rlmpc_config.py`:

```python
from dataclasses import dataclass, field
from typing import Literal

from rl_mixed_traffic.configs.ppo_config import PPOConfig


@dataclass
class RLMPCConfig:
    """Configuration for RLMPC training (warm-start RL or RL+NNMPC residual)."""

    # Mode
    mode: Literal["warm_start", "residual"] = "warm_start"

    # Paths
    nnmpc_path: str = "deep_lcc_results/nnmpc.pth"
    sumocfg_path: str = "configs/ring/platoon_9.sumocfg"
    out_dir: str = "deep_lcc_results/rlmpc_{mode}/"

    # SUMO
    use_gui: bool = False

    # Episode
    episode_length_s: float = 30.0
    Tstep: float = 0.05  # must match step-length in sumocfg

    # Platoon
    n_vehicle: int = 9          # 1 head + 8 followers
    n_followers: int = 8
    cav_positions: tuple[int, ...] = (3, 6)  # 1-indexed within platoon
    v_star: float = 15.0
    s_star: float = 20.0
    v_max: float = 30.0
    accel_max: float = 3.0
    accel_min: float = -5.0

    # Cost weights (match DeepLCCConfig)
    weight_v: float = 5.0
    weight_s: float = 0.1
    weight_u: float = 0.1
    spacing_min: float = 5.0
    spacing_max: float = 40.0

    # NNMPC architecture
    T_ini: int = 20
    obs_dim: int = 260   # T_ini * (m_ctr + p_ctr + 1) = 20 * (2 + 10 + 1)
    action_dim: int = 2  # m_ctr (number of CAVs)

    # Action bounds by mode (residual is ±2 m/s² added to NNMPC, then clipped)
    residual_max: float = 2.0

    # Lagrangian penalty weight
    lambda_violation: float = 1.0

    # PPO hyperparams (delegated)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Training loop
    total_steps: int = 1_000_000
    rollout_steps: int = 4096
    save_freq: int = 100_000

    # Warm-start specifics
    log_std_init_warm: float = -2.302585  # log(0.1)
    log_std_init_residual: float = -0.693147  # log(0.5)
    final_layer_gain_warm: float = 1.0
    final_layer_gain_residual: float = 0.01

    # Optional critic-only warm-up updates (paper risk mitigation).
    # Field reserved; implementation deferred to Task 10.1 if divergence is observed.
    critic_warmup_updates: int = 0

    # HDV randomization (per-vehicle, per-episode), applied via traci.vehicle.set*
    hdv_tau_range: tuple[float, float] = (0.8, 1.5)
    hdv_accel_range: tuple[float, float] = (1.5, 2.5)
    hdv_decel_range: tuple[float, float] = (2.5, 3.5)
    hdv_minGap_range: tuple[float, float] = (2.0, 3.0)
    hdv_sigma_range: tuple[float, float] = (0.3, 0.6)

    # Seed
    seed: int = 42

    @property
    def max_steps(self) -> int:
        return int(self.episode_length_s / self.Tstep)

    @property
    def m_ctr(self) -> int:
        return len(self.cav_positions)

    @property
    def p_ctr(self) -> int:
        # measurement_type=3: n_followers velocity errors + m_ctr spacing errors
        return self.n_followers + self.m_ctr
```

- [ ] **Step 2.2: Quick smoke test of the dataclass.**

Run: `uv run python -c "from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig; c = RLMPCConfig(); print(c.mode, c.obs_dim, c.action_dim, c.max_steps, c.m_ctr, c.p_ctr)"`
Expected: `warm_start 260 2 600 2 10`

- [ ] **Step 2.3: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/rlmpc_config.py
git commit -m "feat: add RLMPCConfig dataclass for RLMPC training"
```

---

## Task 3: `PerturbMixHeadController`

The head vehicle's velocity trace is sampled once per `reset()` from `DeepLCCConfig.perturb_mix` and applied via `setSpeed` step-by-step. We pre-compute the whole trace at `reset()` so the trace is deterministic given the seed and SUMO can't re-randomize within an episode.

**Files:**
- Create: `rl_mixed_traffic/deep_lcc/rlmpc_head_controller.py`
- Create: `tests/test_rlmpc_head_controller.py`

- [ ] **Step 3.1: Write a failing test for `sample_perturbation`.**

Write `tests/test_rlmpc_head_controller.py`:

```python
"""Tests for PerturbMixHeadController."""

import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

# Stub SUMO_HOME and traci so the module imports without SUMO installed.
os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")
_fake_traci = types.ModuleType("traci")
_fake_traci.vehicle = types.SimpleNamespace(
    getIDList=lambda: ["car0"],
    setSpeed=lambda vid, speed: None,
)
sys.modules.setdefault("traci", _fake_traci)


from rl_mixed_traffic.deep_lcc.rlmpc_head_controller import (
    PerturbMixHeadController,
)


class TestSamplePerturbation:
    def test_random_perturbation_within_amplitude(self):
        ctrl = PerturbMixHeadController(
            head_id="car0",
            tstep=0.05,
            episode_length_s=10.0,
            v_star=15.0,
            seed=1,
        )
        trace = ctrl.sample_perturbation(("random", 1.0, 1.0), seed=1)
        assert trace.shape == (200,)
        assert (trace >= 14.0 - 1e-6).all()
        assert (trace <= 16.0 + 1e-6).all()

    def test_brake_perturbation_dips_then_recovers(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=30.0,
            v_star=15.0, seed=2,
        )
        trace = ctrl.sample_perturbation(("brake", 0.0, 1.0), seed=2)
        # Trace should reach v_low ≤ 5 m/s during the brake phase
        assert trace.min() <= 5.0
        # And recover toward v_star at the end
        assert abs(trace[-1] - 15.0) < 2.0

    def test_sinusoidal_perturbation_amplitude(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=20.0,
            v_star=15.0, seed=3,
        )
        trace = ctrl.sample_perturbation(("sinusoidal", 5.0, 1.0), seed=3)
        # Amplitude ≈ 5 m/s
        assert abs(trace.max() - 20.0) < 1.5
        assert abs(trace.min() - 10.0) < 1.5

    def test_unknown_type_raises(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=10.0,
            v_star=15.0, seed=1,
        )
        with pytest.raises(ValueError):
            ctrl.sample_perturbation(("garbage", 1.0, 1.0), seed=1)


class TestResetTraceSelection:
    def test_reset_picks_one_type_from_mix(self):
        # Mix with only "random" entries → reset must pick "random".
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=10.0,
            v_star=15.0, seed=42,
            perturb_mix=[("random", 1.0, 0.5), ("random", 3.0, 0.5)],
        )
        ctrl.reset()
        assert ctrl.trace is not None
        assert ctrl.trace.shape == (200,)
        # All values in [12, 18] (worst case amp=3)
        assert (ctrl.trace >= 11.99).all()
        assert (ctrl.trace <= 18.01).all()


class TestSetRandomHeadSpeed:
    def test_calls_setspeed_with_trace_value(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=2.0,
            v_star=15.0, seed=1,
            perturb_mix=[("random", 0.0, 1.0)],   # constant trace at v_star
        )
        ctrl.reset()

        with patch("traci.vehicle.setSpeed") as mock_set:
            ctrl.set_random_head_speed()  # called every step in update_every_step mode
            mock_set.assert_called_once()
            args, _ = mock_set.call_args
            assert args[0] == "car0"
            assert abs(args[1] - 15.0) < 1e-6
```

- [ ] **Step 3.2: Run the test, expect failure (`ModuleNotFoundError`).**

Run: `uv run pytest tests/test_rlmpc_head_controller.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'rl_mixed_traffic.deep_lcc.rlmpc_head_controller'`.

- [ ] **Step 3.3: Implement the controller.**

Write `rl_mixed_traffic/deep_lcc/rlmpc_head_controller.py`:

```python
"""Head-vehicle controller that plays a velocity trace sampled from DeepLCCConfig.perturb_mix.

Each reset() picks one perturbation type from the mix (weighted by fraction)
and pre-computes a velocity trace for the whole episode. set_random_head_speed()
is called every simulation step (via update_every_step=True) and applies the
next sample from the trace via traci.vehicle.setSpeed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "share" / "sumo" / "tools"
    sys.path.append(str(tools))

import traci

from rl_mixed_traffic.env.head_vehicle_controller import HeadVehicleController


PerturbEntry = tuple[str, float, float]  # (type, amplitude, fraction)


class PerturbMixHeadController(HeadVehicleController):
    """Head controller that samples a perturb_mix entry per reset() and plays the trace.

    Args:
        head_id: SUMO vehicle ID for the head ("car0").
        tstep: Simulation step length (seconds). Must match sumocfg.
        episode_length_s: Total episode duration (seconds).
        v_star: Equilibrium head velocity (m/s).
        seed: RNG seed for perturbation type selection and trace generation.
        perturb_mix: List of (type, amplitude, fraction) tuples. Fractions must sum to 1.
            Types: "random" (uniform ±amp around v_star),
                   "brake" (decel-coast-accel profile),
                   "sinusoidal" (sine of given amplitude, period 10 s).
            If None, uses DeepLCCConfig().perturb_mix.
    """

    DEFAULT_MIX: list[PerturbEntry] = [
        ("random", 1.0, 0.30),
        ("random", 3.0, 0.15),
        ("random", 5.0, 0.10),
        ("brake", 0.0, 0.25),
        ("sinusoidal", 5.0, 0.10),
        ("sinusoidal", 3.0, 0.10),
    ]

    def __init__(
        self,
        head_id: str = "car0",
        tstep: float = 0.05,
        episode_length_s: float = 30.0,
        v_star: float = 15.0,
        seed: int = 0,
        perturb_mix: list[PerturbEntry] | None = None,
    ):
        super().__init__(
            head_id=head_id,
            head_speed_min=0,
            head_speed_max=int(v_star * 2),  # generous; setSpeed clamps anyway
        )
        self.tstep = tstep
        self.episode_length_s = episode_length_s
        self.v_star = v_star
        self.perturb_mix = list(perturb_mix) if perturb_mix is not None else list(self.DEFAULT_MIX)
        self._rng = np.random.default_rng(seed)
        self.trace: np.ndarray | None = None
        self._step_idx: int = 0

    @property
    def update_every_step(self) -> bool:
        return True

    @property
    def total_steps(self) -> int:
        return int(round(self.episode_length_s / self.tstep))

    def reset(self) -> None:
        """Pick a perturbation entry and pre-compute the trace for the whole episode."""
        types = [e[0] for e in self.perturb_mix]
        amps = [e[1] for e in self.perturb_mix]
        fracs = np.asarray([e[2] for e in self.perturb_mix], dtype=np.float64)
        fracs = fracs / fracs.sum()
        idx = int(self._rng.choice(len(self.perturb_mix), p=fracs))
        entry = (types[idx], amps[idx], fracs[idx])
        self.trace = self.sample_perturbation(entry, seed=int(self._rng.integers(0, 2**31)))
        self._step_idx = 0

    def sample_perturbation(self, entry: PerturbEntry, seed: int) -> np.ndarray:
        """Build a velocity trace of length self.total_steps for the given perturb entry."""
        kind, amp, _ = entry
        rng = np.random.default_rng(seed)
        n = self.total_steps
        t = np.arange(n) * self.tstep

        if kind == "random":
            return self.v_star + rng.uniform(-amp, amp, size=n)

        if kind == "sinusoidal":
            period = 10.0
            return self.v_star + amp * np.sin(2.0 * np.pi * t / period)

        if kind == "brake":
            # 5s settle → 2s @ -5 m/s² → 5s coast at v_low → 5s @ +2 m/s² → cruise
            v_cruise = self.v_star
            v_low = max(0.0, v_cruise - 10.0)
            trace = np.full(n, v_cruise, dtype=np.float64)
            settle = 5.0
            brake_dur = 2.0
            coast_dur = 5.0
            recover_dur = 5.0
            for k in range(n):
                tk = k * self.tstep
                if tk < settle:
                    trace[k] = v_cruise
                elif tk < settle + brake_dur:
                    frac = (tk - settle) / brake_dur
                    trace[k] = v_cruise + frac * (v_low - v_cruise)
                elif tk < settle + brake_dur + coast_dur:
                    trace[k] = v_low
                elif tk < settle + brake_dur + coast_dur + recover_dur:
                    frac = (tk - settle - brake_dur - coast_dur) / recover_dur
                    trace[k] = v_low + frac * (v_cruise - v_low)
                else:
                    trace[k] = v_cruise
            return trace

        raise ValueError(f"Unknown perturbation type: {kind!r}")

    def set_random_head_speed(self) -> None:
        """Apply the next sample from the pre-computed trace via setSpeed."""
        if self.trace is None:
            return
        if self.head_id not in traci.vehicle.getIDList():
            self._step_idx += 1
            return
        if self._step_idx >= len(self.trace):
            return
        v = float(self.trace[self._step_idx])
        traci.vehicle.setSpeed(self.head_id, v)
        self._step_idx += 1
```

- [ ] **Step 3.4: Run the tests, expect pass.**

Run: `uv run pytest tests/test_rlmpc_head_controller.py -v`
Expected: 5 passing tests.

- [ ] **Step 3.5: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/rlmpc_head_controller.py tests/test_rlmpc_head_controller.py
git commit -m "feat: add PerturbMixHeadController for RLMPC training"
```

---

## Task 4: `NNMPCActorCritic` network

**Files:**
- Create: `rl_mixed_traffic/deep_lcc/nnmpc_actor_critic.py`
- Create: `tests/test_nnmpc_actor_critic.py`

- [ ] **Step 4.1: Write failing tests for the network.**

Write `tests/test_nnmpc_actor_critic.py`:

```python
"""Tests for NNMPCActorCritic — warm-start fidelity, residual zero-init,
and distribution math."""

import math
import numpy as np
import pytest
import torch

from rl_mixed_traffic.deep_lcc.nnmpc_actor_critic import NNMPCActorCritic
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork


@pytest.fixture
def fake_nnmpc_ckpt(tmp_path):
    """Build a small NNMPCNetwork checkpoint to use for warm-start tests."""
    nnmpc = NNMPCNetwork(input_dim=260, output_dim=2, hidden_dims=(256, 128))
    ckpt = {
        "model_state_dict": nnmpc.state_dict(),
        "input_mean": np.zeros(260, dtype=np.float32),
        "input_std": np.ones(260, dtype=np.float32),
        "input_dim": 260,
        "output_dim": 2,
        "config": {
            "hidden_dims": (256, 128),
            "accel_min": -5.0,
            "accel_max": 3.0,
        },
    }
    p = tmp_path / "fake_nnmpc.pth"
    torch.save(ckpt, p)
    return p, nnmpc


class TestForwardShape:
    def test_actor_output_and_value_shapes(self):
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        x = torch.zeros(4, 260)
        actor_out, value = net(x)
        assert actor_out.shape == (4, 2)
        assert value.shape == (4, 1)


class TestWarmStart:
    def test_warm_start_replicates_nnmpc_pre_tanh_output(self, fake_nnmpc_ckpt):
        ckpt_path, nnmpc = fake_nnmpc_ckpt
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        net.warm_start_from_nnmpc(str(ckpt_path))

        x = torch.randn(8, 260)

        # NNMPC output: net(x) returns the post-tanh, scaled action in [-5, 3].
        nn_out = nnmpc(x)

        # NNMPCActorCritic actor body produces the pre-tanh logits;
        # tanh + scale to [-5, 3] should reproduce NNMPC's output.
        with torch.no_grad():
            pre_tanh, _ = net(x)
        accel_min, accel_max = -5.0, 3.0
        scaled = (torch.tanh(pre_tanh) + 1.0) / 2.0 * (accel_max - accel_min) + accel_min

        assert torch.allclose(scaled, nn_out, atol=1e-5)


class TestResidualInit:
    def test_residual_init_outputs_near_zero(self):
        net = NNMPCActorCritic(
            obs_dim=260, action_dim=2,
            log_std_init=math.log(0.5),
            final_layer_gain=0.01,
        )
        x = torch.randn(16, 260)
        with torch.no_grad():
            pre_tanh, _ = net(x)
        # Pre-tanh logits should be close to zero so tanh(.) ≈ 0.
        assert pre_tanh.abs().mean().item() < 0.1


class TestGetActionAndValue:
    def test_signature_and_log_prob_shape(self):
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        x = torch.zeros(5, 260)
        action, log_prob, entropy, value = net.get_action_and_value(x)
        assert action.shape == (5, 2)
        # Action sampled then tanh-squashed to [-1, 1]
        assert (action >= -1.0).all() and (action <= 1.0).all()
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5, 1)

    def test_log_prob_with_provided_action(self):
        torch.manual_seed(0)
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        x = torch.zeros(3, 260)
        action = torch.tensor([[0.0, 0.0], [0.5, -0.5], [-0.9, 0.9]])
        _, log_prob, entropy, value = net.get_action_and_value(x, action=action)
        assert log_prob.shape == (3,)
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()


class TestLogStdInit:
    def test_warm_start_uses_lower_std(self):
        net_warm = NNMPCActorCritic(obs_dim=260, action_dim=2,
                                     log_std_init=math.log(0.1))
        net_resid = NNMPCActorCritic(obs_dim=260, action_dim=2,
                                      log_std_init=math.log(0.5))
        assert net_warm.actor_log_std.mean().item() < net_resid.actor_log_std.mean().item()
```

- [ ] **Step 4.2: Run tests, expect failure (`ModuleNotFoundError`).**

Run: `uv run pytest tests/test_nnmpc_actor_critic.py -v`
Expected: `ModuleNotFoundError: No module named 'rl_mixed_traffic.deep_lcc.nnmpc_actor_critic'`.

- [ ] **Step 4.3: Implement the network.**

Write `rl_mixed_traffic/deep_lcc/nnmpc_actor_critic.py`:

```python
"""NNMPC-shaped actor-critic network for RLMPC PPO training.

Actor architecture matches NNMPCNetwork (260 → 256 → 128 → 2) so warm-starting
from nnmpc.pth is a direct state_dict copy. The actor produces the *pre-tanh*
mean of a diagonal Gaussian; the post-tanh squashed action is in [-1, 1] and
the env scales it to the per-mode action box.

Implements the same get_action_and_value signature as
rl_mixed_traffic/ppo/network.py:ActorCriticNetwork so PPOAgent works unchanged.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import nn


def _ortho_init(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class NNMPCActorCritic(nn.Module):
    """Actor-Critic with NNMPC-shaped actor body and an independent critic head."""

    def __init__(
        self,
        obs_dim: int = 260,
        action_dim: int = 2,
        hidden_dims: tuple[int, ...] = (256, 128),
        log_std_init: float = math.log(0.5),
        final_layer_gain: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)

        # --- Actor body: same shape as NNMPCNetwork's nn.Sequential pre-tanh ---
        actor_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            actor_layers.append(_ortho_init(nn.Linear(prev, h)))
            actor_layers.append(nn.ReLU())
            prev = h
        actor_final = _ortho_init(nn.Linear(prev, action_dim), gain=final_layer_gain)
        actor_layers.append(actor_final)
        self.actor_body = nn.Sequential(*actor_layers)

        # --- Critic head: independent MLP, same hidden dims ---
        critic_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            critic_layers.append(_ortho_init(nn.Linear(prev, h)))
            critic_layers.append(nn.ReLU())
            prev = h
        critic_layers.append(_ortho_init(nn.Linear(prev, 1), gain=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # State-independent learnable log-std
        self.actor_log_std = nn.Parameter(
            torch.full((1, action_dim), float(log_std_init))
        )

    # ----- Public API matching ActorCriticNetwork -----

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_pre_tanh = self.actor_body(state)
        value = self.critic(state)
        return actor_pre_tanh, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_pre_tanh, value = self.forward(state)
        log_std = torch.clamp(self.actor_log_std, min=-2.0, max=0.5)
        std = torch.exp(log_std.expand_as(actor_pre_tanh))
        normal = torch.distributions.Normal(actor_pre_tanh, std)

        if action is None:
            raw = normal.rsample()
            action_squashed = torch.tanh(raw)
        else:
            action_squashed = action
            raw = torch.atanh(torch.clamp(action_squashed, -0.999, 0.999))

        log_prob = normal.log_prob(raw).sum(dim=-1)
        # tanh-squash log-prob correction
        log_prob = log_prob - torch.log(
            1.0 - action_squashed.pow(2) + 1e-6
        ).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)

        return action_squashed, log_prob, entropy, value

    # ----- Warm-start from a trained NNMPCNetwork checkpoint -----

    def warm_start_from_nnmpc(self, ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Copy NNMPCNetwork weights into self.actor_body.

        NNMPCNetwork.net is nn.Sequential of:
            [Linear, ReLU, Linear, ReLU, Linear, Tanh]
        We map the three Linear layers into self.actor_body's three Linear layers.

        Returns:
            (input_mean, input_std) from the checkpoint, for env-side normalization.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        # NNMPCNetwork.net keys look like 'net.0.weight', 'net.0.bias',
        # 'net.2.weight', 'net.2.bias', 'net.4.weight', 'net.4.bias'
        # Our actor_body (same shape) has keys '0.weight', '0.bias', etc.
        nnmpc_indices = [0, 2, 4]
        actor_indices = [0, 2, 4]
        for nnmpc_i, actor_i in zip(nnmpc_indices, actor_indices):
            w_key = f"net.{nnmpc_i}.weight"
            b_key = f"net.{nnmpc_i}.bias"
            if w_key not in sd or b_key not in sd:
                raise KeyError(
                    f"NNMPC checkpoint missing keys {w_key!r}/{b_key!r}"
                )
            target = self.actor_body[actor_i]
            if target.weight.shape != sd[w_key].shape:
                raise ValueError(
                    f"Shape mismatch at layer {actor_i}: "
                    f"actor {tuple(target.weight.shape)} vs nnmpc {tuple(sd[w_key].shape)}"
                )
            with torch.no_grad():
                target.weight.copy_(sd[w_key])
                target.bias.copy_(sd[b_key])

        return ckpt["input_mean"], ckpt["input_std"]
```

- [ ] **Step 4.4: Run tests, expect pass.**

Run: `uv run pytest tests/test_nnmpc_actor_critic.py -v`
Expected: 6 passing tests.

- [ ] **Step 4.5: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/nnmpc_actor_critic.py tests/test_nnmpc_actor_critic.py
git commit -m "feat: add NNMPCActorCritic with warm-start from nnmpc.pth"
```

---

## Task 5: `PlatoonNNMPCEnv` — buffer + observation logic (no SUMO yet)

We isolate the buffer-management logic from SUMO so we can unit-test it. Once it works in isolation, Task 6 wires it up to the full SUMO env.

**Files:**
- Create: `rl_mixed_traffic/deep_lcc/rlmpc_env.py` (initial: just `ObservationBuilder` helper class + skeleton)
- Create: `tests/test_rlmpc_env.py` (tests for `ObservationBuilder`)

- [ ] **Step 5.1: Write failing tests for `ObservationBuilder`.**

Write `tests/test_rlmpc_env.py`:

```python
"""Tests for PlatoonNNMPCEnv pieces that don't require SUMO.

Mocks traci where needed; pure-logic tests for ObservationBuilder.
"""

import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")
_fake_traci = types.ModuleType("traci")
_fake_traci.vehicle = types.SimpleNamespace(
    getIDList=lambda: [],
    getSpeed=lambda vid: 0.0,
    getDistance=lambda vid: 0.0,
    getLength=lambda vid: 5.0,
    getLeader=lambda vid: None,
    getAcceleration=lambda vid: 0.0,
    setSpeed=lambda vid, speed: None,
    setSpeedMode=lambda vid, mode: None,
    setMaxSpeed=lambda vid, mx: None,
    setTau=lambda vid, t: None,
    setAccel=lambda vid, a: None,
    setDecel=lambda vid, d: None,
    setMinGap=lambda vid, g: None,
    setImperfection=lambda vid, s: None,
)
sys.modules.setdefault("traci", _fake_traci)
_fake_sumolib = types.ModuleType("sumolib")
_fake_sumolib.checkBinary = lambda x: x
sys.modules.setdefault("sumolib", _fake_sumolib)


from rl_mixed_traffic.deep_lcc.rlmpc_env import ObservationBuilder


class TestObservationBuilder:
    def test_empty_buffers_have_correct_shapes(self):
        b = ObservationBuilder(T_ini=20, n_followers=8, m_ctr=2)
        # Before any push, buffers are zero-padded
        obs = b.build_obs()
        assert obs.shape == (260,)
        assert obs.dtype == np.float32
        assert (obs == 0.0).all()

    def test_uini_yini_eini_layout(self):
        b = ObservationBuilder(T_ini=2, n_followers=3, m_ctr=1)
        # T_ini=2, n_followers=3, m_ctr=1 → p_ctr = 4
        # uini: 2*1 = 2, yini: 2*4 = 8, eini: 2*1 = 2 → total 12

        # Push step 0
        b.push_step(
            uini_step=np.array([1.0]),
            yini_step=np.array([10.0, 11.0, 12.0, 100.0]),
            eini_step=np.array([0.5]),
        )
        # Push step 1
        b.push_step(
            uini_step=np.array([2.0]),
            yini_step=np.array([20.0, 21.0, 22.0, 200.0]),
            eini_step=np.array([0.6]),
        )

        obs = b.build_obs()
        assert obs.shape == (12,)
        # uini: [1.0, 2.0]
        np.testing.assert_array_equal(obs[:2], [1.0, 2.0])
        # yini: [10, 11, 12, 100, 20, 21, 22, 200]
        np.testing.assert_array_equal(obs[2:10], [10, 11, 12, 100, 20, 21, 22, 200])
        # eini: [0.5, 0.6]
        np.testing.assert_array_equal(obs[10:12], [0.5, 0.6])

    def test_normalize_uses_provided_stats(self):
        b = ObservationBuilder(T_ini=2, n_followers=3, m_ctr=1)
        b.set_normalization(
            mean=np.full(12, 2.0, dtype=np.float32),
            std=np.full(12, 4.0, dtype=np.float32),
        )
        b.push_step(
            uini_step=np.array([10.0]),
            yini_step=np.array([2.0, 2.0, 2.0, 2.0]),
            eini_step=np.array([6.0]),
        )
        b.push_step(
            uini_step=np.array([10.0]),
            yini_step=np.array([2.0, 2.0, 2.0, 2.0]),
            eini_step=np.array([6.0]),
        )
        # mean=2, std=4 → uini values (10) → (10-2)/4 = 2; yini → 0; eini → 1
        norm_obs = b.build_normalized_obs()
        np.testing.assert_array_almost_equal(norm_obs[:2], [2.0, 2.0])
        np.testing.assert_array_almost_equal(norm_obs[2:10], np.zeros(8))
        np.testing.assert_array_almost_equal(norm_obs[10:12], [1.0, 1.0])

    def test_overflow_drops_oldest(self):
        b = ObservationBuilder(T_ini=2, n_followers=1, m_ctr=1)
        # Push 3 steps; only the last 2 survive.
        for k in range(3):
            b.push_step(
                uini_step=np.array([float(k)]),
                yini_step=np.array([float(k), float(k)]),
                eini_step=np.array([float(k)]),
            )
        obs = b.build_obs()
        # uini: [1.0, 2.0]
        np.testing.assert_array_equal(obs[:2], [1.0, 2.0])
```

- [ ] **Step 5.2: Run tests, expect failure.**

Run: `uv run pytest tests/test_rlmpc_env.py -v`
Expected: `ModuleNotFoundError: No module named 'rl_mixed_traffic.deep_lcc.rlmpc_env'`.

- [ ] **Step 5.3: Implement `ObservationBuilder` (just enough for tests).**

Write `rl_mixed_traffic/deep_lcc/rlmpc_env.py`:

```python
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
        out = np.zeros(target_len, dtype=np.float32)
        if len(buf) > 0:
            arr = np.fromiter(buf, dtype=np.float32, count=len(buf))
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
```

- [ ] **Step 5.4: Run tests, expect pass.**

Run: `uv run pytest tests/test_rlmpc_env.py -v`
Expected: 4 passing tests.

- [ ] **Step 5.5: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/rlmpc_env.py tests/test_rlmpc_env.py
git commit -m "feat: add ObservationBuilder for NNMPC-shaped state buffers"
```

---

## Task 6: `PlatoonNNMPCEnv` — full SUMO env

We extend `rlmpc_env.py` with the actual gymnasium env that uses SUMO. Subclass `RingRoadEnv` so reward + spacing-violation logic is inherited.

**Files:**
- Modify: `rl_mixed_traffic/deep_lcc/rlmpc_env.py` (append `PlatoonNNMPCEnv`)
- Modify: `tests/test_rlmpc_env.py` (add tests for env construction with mocked traci)

- [ ] **Step 6.1: Add tests for env construction and obs/action shapes.**

Append to `tests/test_rlmpc_env.py`:

```python
# ----------------------------------------------------------------------
# PlatoonNNMPCEnv tests (mocked traci)
# ----------------------------------------------------------------------

from unittest.mock import patch

from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_env import PlatoonNNMPCEnv


@pytest.fixture
def stub_nnmpc_ckpt(tmp_path):
    """Save a small NNMPC checkpoint to disk for env construction."""
    import torch
    from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork

    nnmpc = NNMPCNetwork(input_dim=260, output_dim=2, hidden_dims=(256, 128))
    p = tmp_path / "nnmpc.pth"
    torch.save({
        "model_state_dict": nnmpc.state_dict(),
        "input_mean": np.zeros(260, dtype=np.float32),
        "input_std": np.ones(260, dtype=np.float32),
        "input_dim": 260,
        "output_dim": 2,
        "config": {
            "hidden_dims": (256, 128),
            "accel_min": -5.0,
            "accel_max": 3.0,
        },
    }, p)
    return str(p)


class TestPlatoonNNMPCEnvConstruction:
    def test_warm_start_action_space(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        # Warm-start: full physical range, asymmetric — action_space exposes a
        # symmetric box [-max_residual, +max_residual] for the policy in the
        # squashed Gaussian view; env scales internally to [accel_min, accel_max].
        assert env.action_space.shape == (2,)
        # Default warm-start: tanh output ∈ [-1, 1] scaled to [-1, +1] then
        # mapped per-mode in apply_action; high/low here describe the scaled box.
        assert env.action_space.high.tolist() == [3.0, 3.0]
        assert env.action_space.low.tolist() == [-5.0, -5.0]

    def test_residual_action_space(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="residual", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        assert env.action_space.shape == (2,)
        assert env.action_space.high.tolist() == [2.0, 2.0]
        assert env.action_space.low.tolist() == [-2.0, -2.0]

    def test_observation_space_shape(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        assert env.observation_space.shape == (260,)

    def test_compose_action_residual_mode_clips_total(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="residual", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)

        # NNMPC suggests full +3 m/s² for both CAVs; residual adds +2 → +5 → clip to +3
        u_total = env._compose_action(
            policy_action=np.array([2.0, 2.0]),
            u_nnmpc=np.array([3.0, 3.0]),
        )
        np.testing.assert_array_equal(u_total, [3.0, 3.0])

        # NNMPC suggests -5; residual subtracts 1 → -6 → clip to -5
        u_total = env._compose_action(
            policy_action=np.array([-1.0, -1.0]),
            u_nnmpc=np.array([-5.0, -5.0]),
        )
        np.testing.assert_array_equal(u_total, [-5.0, -5.0])

        # In-range stays in range
        u_total = env._compose_action(
            policy_action=np.array([0.5, -0.5]),
            u_nnmpc=np.array([1.0, -1.0]),
        )
        np.testing.assert_array_almost_equal(u_total, [1.5, -1.5])

    def test_compose_action_warm_start_uses_policy_directly(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        # In warm-start mode the policy output IS the action (after scaling).
        # NNMPC is bypassed and u_nnmpc is irrelevant (set to 0).
        u_total = env._compose_action(
            policy_action=np.array([2.5, -3.0]),
            u_nnmpc=np.zeros(2),
        )
        np.testing.assert_array_almost_equal(u_total, [2.5, -3.0])


class TestRewardAugmentation:
    def test_collision_overrides_to_minus_one(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        # Direct test of the helper: collision flag forces -1.
        r = env._augment_reward(r_base=0.5, violation=0.2, collision=True)
        assert r == -1.0

    def test_lagrangian_penalty_subtracts(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt,
                          lambda_violation=2.0)
        env = PlatoonNNMPCEnv(cfg)
        r = env._augment_reward(r_base=0.5, violation=0.1, collision=False)
        assert abs(r - (0.5 - 2.0 * 0.1)) < 1e-6
```

- [ ] **Step 6.2: Run new tests, expect failure (`PlatoonNNMPCEnv` not defined).**

Run: `uv run pytest tests/test_rlmpc_env.py -v`
Expected: import error or `AttributeError` for `PlatoonNNMPCEnv`.

- [ ] **Step 6.3: Implement `PlatoonNNMPCEnv`.**

Append to `rl_mixed_traffic/deep_lcc/rlmpc_env.py`:

```python
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
        # Build SumoConfig and parent env
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
        # Tell RingRoadEnv there are 9 vehicles, 2 agents, with cost weights from config.
        super().__init__(
            sumo_config=sumo_config,
            agent_id="car3",                 # backward-compat single-agent alias
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
        # Override agent_ids to be the CAVs only ([car3, car6]).
        self.agent_ids = [f"car{p}" for p in config.cav_positions]
        self.agent_id = self.agent_ids[0]
        self.prev_accels = {aid: 0.0 for aid in self.agent_ids}
        self.cmd_speeds = {aid: 0.0 for aid in self.agent_ids}

        self.config = config
        self._cav_indices_in_platoon = [p - 1 for p in config.cav_positions]   # 0-indexed
        # ID list (1=CAV, 0=HDV) for measure_mixed_traffic
        self._ID = [
            1 if (i + 1) in config.cav_positions else 0
            for i in range(config.n_followers)
        ]
        # All follower IDs in platoon order
        self._follower_ids = [f"car{i}" for i in range(1, config.n_followers + 1)]

        # Observation builder
        self._builder = ObservationBuilder(
            T_ini=config.T_ini,
            n_followers=config.n_followers,
            m_ctr=config.m_ctr,
        )

        # Load NNMPC (always loaded; needed for normalization stats and for residual mode)
        self._nnmpc, mean, std = self._load_nnmpc(config.nnmpc_path)
        self._builder.set_normalization(mean, std)

        # Mode-aware action space
        if config.mode == "warm_start":
            low = np.full(config.action_dim, config.accel_min, dtype=np.float32)
            high = np.full(config.action_dim, config.accel_max, dtype=np.float32)
        else:
            low = np.full(config.action_dim, -config.residual_max, dtype=np.float32)
            high = np.full(config.action_dim, +config.residual_max, dtype=np.float32)
        self._action_space = Box(low=low, high=high, shape=(config.action_dim,),
                                  dtype=np.float32)
        # 260-dim observation
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(config.obs_dim,), dtype=np.float32,
        )

    # Override the property-based spaces from RingRoadEnv with our fixed ones.
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

        policy_action is already in the per-mode action box (warm-start: [-5, 3]^2;
        residual: [-2, 2]^2). For residual, we add NNMPC and clip; for warm_start,
        the policy IS the command (NNMPC bypassed).
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

    # ------------------------------------------------------------------
    # Gymnasium API: reset() and step()
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
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

        # Wait for all expected vehicles to appear
        expected = ["car0"] + self._follower_ids
        warmup = 0
        while warmup < 200:
            present = all(v in traci.vehicle.getIDList() for v in expected)
            if present:
                break
            traci.simulationStep()
            warmup += 1

        # Sample HDV parameters per non-CAV follower
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

        # Warm-up T_ini steps under SUMO IDM control (no policy action yet)
        for _ in range(self.config.T_ini):
            self._step_for_warmup()

        # After warm-up, take CAV control: speed mode 95, max speed v_max
        for aid in self.agent_ids:
            if aid in traci.vehicle.getIDList():
                traci.vehicle.setSpeedMode(aid, 95)
                traci.vehicle.setMaxSpeed(aid, self.config.v_max)
                self.cmd_speeds[aid] = float(traci.vehicle.getSpeed(aid))

        obs = self._builder.build_normalized_obs()
        return obs, {}

    def _step_for_warmup(self) -> None:
        """One simulation step during warm-up; record observation buffers."""
        # Apply head controller (will use trace in update_every_step mode)
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
        # uini: applied accelerations of the CAVs in this step
        uini = np.array([
            traci.vehicle.getAcceleration(aid) if aid in active else 0.0
            for aid in self.agent_ids
        ], dtype=np.float32)

        # Build velocity and position vectors in platoon order:
        #   index 0 = head (car0), index 1..n_followers = followers in order
        n = self.config.n_followers
        vel = np.zeros(n, dtype=np.float32)
        pos = np.zeros(n + 1, dtype=np.float32)
        pos[0] = traci.vehicle.getDistance("car0") if "car0" in active else 0.0
        for i, fid in enumerate(self._follower_ids):
            if fid in active:
                vel[i] = traci.vehicle.getSpeed(fid)
                pos[i + 1] = traci.vehicle.getDistance(fid)
            else:
                # Vehicle missing: fall back to last known distance/velocity = 0
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

    def step(self, action: np.ndarray):
        # Action arrives in the per-mode action box already (env.action_space)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # Compute NNMPC suggestion for residual mode (or zero for warm_start)
        if self.config.mode == "residual":
            obs_norm = self._builder.build_normalized_obs()
            with torch.no_grad():
                u_nnmpc = self._nnmpc(
                    torch.from_numpy(obs_norm).unsqueeze(0)
                ).cpu().numpy().ravel()
        else:
            u_nnmpc = np.zeros(self.config.action_dim, dtype=np.float32)

        u_total = self._compose_action(action, u_nnmpc)

        # Apply CAV accelerations
        for aid, a in zip(self.agent_ids, u_total):
            if aid in traci.vehicle.getIDList():
                self.apply_acceleration(aid, float(a), smooth=False)
            self.prev_accels[aid] = float(a)

        # Apply head controller speed
        self.head_vehicle_controller.set_random_head_speed()

        traci.simulationStep()
        self.step_count += 1

        if "car0" in traci.vehicle.getIDList():
            self._head_speed_buffer.append(traci.vehicle.getSpeed("car0"))

        self._cached_global_state = None  # invalidate parent's cache
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
```

- [ ] **Step 6.4: Run all env tests, expect pass.**

Run: `uv run pytest tests/test_rlmpc_env.py -v`
Expected: 8 passing tests (4 ObservationBuilder + 4 PlatoonNNMPCEnv construction + 2 reward).

Note: SUMO-touching code paths (`reset`, `step`, `_step_for_warmup`) are *not* unit-tested here; they're exercised in the smoke test (Task 9).

- [ ] **Step 6.5: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/rlmpc_env.py tests/test_rlmpc_env.py
git commit -m "feat: add PlatoonNNMPCEnv (SUMO env for RLMPC)"
```

---

## Task 7: `rlmpc_train.py`

Single training script supporting both modes. Mirrors `rl_mixed_traffic/ppo_train.py`'s `_train_single_agent` loop but with the new env, the new actor-critic, and a 2-D joint action.

**Files:**
- Create: `rl_mixed_traffic/deep_lcc/rlmpc_train.py`

- [ ] **Step 7.1: Write the training script.**

Write `rl_mixed_traffic/deep_lcc/rlmpc_train.py`:

```python
"""Train Warm-Start RL or RL+NNMPC residual via PPO on PlatoonNNMPCEnv.

Usage:
    uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode warm_start
    uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode residual

Outputs:
    deep_lcc_results/rlmpc_{mode}/agent.pth       — final model
    deep_lcc_results/rlmpc_{mode}/agent_step_N.pth — per-checkpoint snapshots
    deep_lcc_results/rlmpc_{mode}/returns.csv     — per-episode return + diagnostics
    deep_lcc_results/rlmpc_{mode}/returns.png     — return curve
    deep_lcc_results/rlmpc_{mode}/ppo_metrics.png — policy/value/entropy/clip
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.deep_lcc.nnmpc_actor_critic import NNMPCActorCritic
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_env import PlatoonNNMPCEnv
from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper
from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_ppo_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["warm_start", "residual"], default="warm_start")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def build_network(config: RLMPCConfig) -> NNMPCActorCritic:
    log_std_init = (
        config.log_std_init_warm if config.mode == "warm_start"
        else config.log_std_init_residual
    )
    final_gain = (
        config.final_layer_gain_warm if config.mode == "warm_start"
        else config.final_layer_gain_residual
    )
    network = NNMPCActorCritic(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        hidden_dims=(256, 128),
        log_std_init=log_std_init,
        final_layer_gain=final_gain,
    )
    if config.mode == "warm_start":
        network.warm_start_from_nnmpc(config.nnmpc_path)
    return network


def scale_action(
    action_tanh: np.ndarray, config: RLMPCConfig,
) -> np.ndarray:
    """Map tanh-squashed action ∈ [-1,1]^2 to the per-mode action box."""
    a = np.asarray(action_tanh, dtype=np.float32).reshape(-1)
    if config.mode == "warm_start":
        # Asymmetric: [-5, 3]
        # tanh in [-1, 1] → linear map: 0.5*(t+1)*(amax-amin)+amin
        amin, amax = config.accel_min, config.accel_max
        return 0.5 * (a + 1.0) * (amax - amin) + amin
    # Residual: symmetric ±residual_max
    return a * config.residual_max


def train(config: RLMPCConfig) -> tuple[list[float], Path]:
    out_dir = Path(config.out_dir.format(mode=config.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = PlatoonNNMPCEnv(config)
    env = FourToFiveTupleWrapper(env)

    network = build_network(config)
    agent = PPOAgent(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        config=config.ppo,
        continuous=True,
        total_steps=config.total_steps,
        rollout_steps=config.rollout_steps,
    )
    # Replace the default ActorCriticNetwork with the NNMPC-shaped one.
    agent.network = network.to(agent.device)
    # Re-create optimizer/scheduler around the new network parameters.
    import torch
    agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=config.ppo.lr)
    if config.ppo.anneal_lr and config.total_steps > 0:
        max_updates = max(config.total_steps // config.rollout_steps, 1)
        lr_lambda = lambda update: 1.0 - (update / max_updates)
    else:
        lr_lambda = lambda update: 1.0
    agent.scheduler = torch.optim.lr_scheduler.LambdaLR(
        agent.optimizer, lr_lambda=lr_lambda,
    )

    print(f"[rlmpc_train] mode={config.mode} obs_dim={config.obs_dim} "
          f"action_dim={config.action_dim} total_steps={config.total_steps}")

    returns: list[float] = []
    metrics_history: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "clipfrac": [],
    }

    s, _ = env.reset(seed=config.seed)
    ep_ret, ep_len = 0.0, 0
    ep_collisions = 0
    ep_violations = 0.0
    step_count = 0

    csv_path = out_dir / "returns.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["episode", "return", "length", "collisions", "mean_violation"]
    )

    while step_count < config.total_steps:
        last_done = False
        for _ in range(config.rollout_steps):
            if step_count >= config.total_steps:
                break

            action_tanh, value, log_prob = agent.get_action_and_value(s)
            action_scaled = scale_action(action_tanh, config)
            s_next, r, done, truncated, info = env.step(action_scaled)

            agent.store_transition(
                s, action_tanh, r, value, log_prob, done or truncated,
            )

            s = s_next
            ep_ret += r
            ep_len += 1
            ep_collisions += int(info.get("collision", False))
            ep_violations += float(info.get("violation", 0.0))
            step_count += 1
            episode_done = done or truncated
            last_done = episode_done

            if episode_done:
                returns.append(ep_ret)
                csv_writer.writerow([
                    len(returns), f"{ep_ret:.4f}", ep_len, ep_collisions,
                    f"{ep_violations / max(ep_len, 1):.4f}",
                ])
                csv_file.flush()
                print(f"Step {step_count:>7d} | Ep {len(returns):>5d} | "
                      f"Ret={ep_ret:>8.2f} | Len={ep_len:>4d} | "
                      f"Coll={ep_collisions} | Viol={ep_violations / max(ep_len, 1):.3f}")
                s, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
                ep_collisions = 0
                ep_violations = 0.0

        # Bootstrap last value
        if last_done:
            last_value = 0.0
        else:
            _, last_value, _ = agent.get_action_and_value(s)

        m = agent.learn(last_value=last_value)
        if m:
            for k in metrics_history:
                if k in m:
                    metrics_history[k].append(m[k])
            print(f"  Update {agent.update_count:>4d} | "
                  f"PL={m['policy_loss']:>7.4f} | "
                  f"VL={m['value_loss']:>7.4f} | "
                  f"Ent={m['entropy']:>6.4f} | "
                  f"CF={m['clipfrac']:>5.3f}")

        if step_count % config.save_freq == 0 or step_count >= config.total_steps:
            agent.save(str(out_dir / f"agent_step_{step_count}.pth"))

    csv_file.close()

    agent.save(str(out_dir / "agent.pth"))
    env.close()

    plot_returns(returns, out_path=str(out_dir / "returns.png"),
                 title=f"RLMPC ({config.mode}) Training Returns")
    plot_ppo_metrics(metrics_history, out_dir=str(out_dir))

    print(f"[rlmpc_train] Done. Saved to {out_dir}")
    return returns, out_dir


def main() -> None:
    args = parse_args()
    config = RLMPCConfig(mode=args.mode, seed=args.seed, use_gui=args.gui)
    if args.total_steps is not None:
        config.total_steps = args.total_steps
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    train(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Smoke import (no SUMO call yet).**

Run: `uv run python -c "from rl_mixed_traffic.deep_lcc.rlmpc_train import build_network, scale_action; from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig; cfg = RLMPCConfig(); print('imports ok')"`

Expected: `imports ok`

- [ ] **Step 7.3: Verify `scale_action` on both modes via a quick assertion.**

Run:
```
uv run python -c "
import numpy as np
from rl_mixed_traffic.deep_lcc.rlmpc_train import scale_action
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig

# warm_start: tanh=+1 → +3, tanh=-1 → -5
cfg_w = RLMPCConfig(mode='warm_start')
assert np.allclose(scale_action(np.array([1.0, 1.0]), cfg_w), [3.0, 3.0])
assert np.allclose(scale_action(np.array([-1.0, -1.0]), cfg_w), [-5.0, -5.0])

# residual: tanh=+1 → +2, tanh=-1 → -2
cfg_r = RLMPCConfig(mode='residual')
assert np.allclose(scale_action(np.array([1.0, 1.0]), cfg_r), [2.0, 2.0])
assert np.allclose(scale_action(np.array([-1.0, -1.0]), cfg_r), [-2.0, -2.0])
print('scale_action ok')
"
```
Expected: `scale_action ok`

- [ ] **Step 7.4: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/rlmpc_train.py
git commit -m "feat: add rlmpc_train.py (PPO train script for warm/residual)"
```

---

## Task 8: `rlmpc_eval.py`

Eval script: load each controller, run the eval grid (scenarios × HDV configs × controllers), produce CSV/MD summary + per-scenario plots.

For v1 we'll focus on the three RL-comparable controllers (NNMPC, Warm-Start RL, RL+NNMPC). Adding QP requires building Hankel matrices in OVM-land for each HDV config — defer to Task 8b if QP is needed for the writeup.

**Files:**
- Create: `rl_mixed_traffic/deep_lcc/rlmpc_eval.py`

- [ ] **Step 8.1: Write the eval script.**

Write `rl_mixed_traffic/deep_lcc/rlmpc_eval.py`:

```python
"""Evaluate RLMPC controllers on the SUMO platoon across scenarios + HDV configs.

Compares: NNMPC alone, Warm-Start RL, RL+NNMPC (residual).
Produces per-(scenario, hdv_config, controller) metrics + plots.

Usage:
    uv run rl_mixed_traffic/deep_lcc/rlmpc_eval.py
    SCENARIOS=brake,sinusoidal uv run rl_mixed_traffic/deep_lcc/rlmpc_eval.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.deep_lcc.eval_classical import (
    make_aggressive_sine,
    make_extreme_brake,
    make_nedc,
    make_sinusoidal,
    make_stop_and_go,
    make_varying_sine,
)
from rl_mixed_traffic.deep_lcc.nnmpc_actor_critic import NNMPCActorCritic
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_env import PlatoonNNMPCEnv
from rl_mixed_traffic.deep_lcc.rlmpc_head_controller import (
    PerturbMixHeadController,
)
from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper


# ----------------------------------------------------------------------
# Controller adapters (each takes obs → action in the env's action space)
# ----------------------------------------------------------------------


def make_nnmpc_controller(nnmpc_path: str):
    """NNMPC alone: forward pass returns full action ∈ [-5, 3]."""
    ckpt = torch.load(nnmpc_path, map_location="cpu", weights_only=False)
    nnmpc = NNMPCNetwork(
        input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"],
        hidden_dims=ckpt["config"]["hidden_dims"],
        accel_min=ckpt["config"]["accel_min"],
        accel_max=ckpt["config"]["accel_max"],
    )
    nnmpc.load_state_dict(ckpt["model_state_dict"])
    nnmpc.eval()

    def controller(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = nnmpc(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().ravel()
        return out  # already in [-5, 3]

    # NNMPC's action lives in [-5, 3] (warm_start-style). Eval env is run with
    # mode="warm_start" so the action box matches.
    return controller


def make_rl_controller(agent_path: str, config: RLMPCConfig):
    """Warm-start or residual RL policy: deterministic mean action."""
    network = NNMPCActorCritic(
        obs_dim=config.obs_dim, action_dim=config.action_dim,
        hidden_dims=(256, 128),
        log_std_init=(
            config.log_std_init_warm if config.mode == "warm_start"
            else config.log_std_init_residual
        ),
    )
    agent = PPOAgent(
        obs_dim=config.obs_dim, action_dim=config.action_dim,
        config=config.ppo, continuous=True,
    )
    agent.network = network
    agent.load(agent_path, map_location="cpu")
    agent.network.eval()

    def controller(obs: np.ndarray) -> np.ndarray:
        action_tanh = agent.act(obs, eval_mode=True)
        action_tanh = np.asarray(action_tanh, dtype=np.float32).reshape(-1)
        if config.mode == "warm_start":
            amin, amax = config.accel_min, config.accel_max
            return 0.5 * (action_tanh + 1.0) * (amax - amin) + amin
        return action_tanh * config.residual_max

    return controller


# ----------------------------------------------------------------------
# Eval runner — runs one (controller, scenario, hdv_seed) and returns metrics
# ----------------------------------------------------------------------


def run_eval_episode(
    config: RLMPCConfig,
    controller_fn: Callable[[np.ndarray], np.ndarray],
    head_trace: np.ndarray,
    hdv_seed: int,
) -> dict:
    """Run a single eval episode and collect metrics."""
    # Override episode length to match scenario duration.
    eval_cfg = replace(
        config,
        episode_length_s=len(head_trace) * config.Tstep,
    )
    env = PlatoonNNMPCEnv(eval_cfg)
    env = FourToFiveTupleWrapper(env)

    # Inject the precomputed head trace by replacing the head controller's trace.
    obs, _ = env.reset(seed=hdv_seed)
    head = env.unwrapped.head_vehicle_controller
    head.trace = head_trace.copy()
    head._step_idx = 0

    cum_reward = 0.0
    cum_cost = 0.0
    n_collisions = 0
    n_violations = 0
    cav_velocities = {aid: [] for aid in env.unwrapped.agent_ids}
    cav_actions = {aid: [] for aid in env.unwrapped.agent_ids}
    head_velocities = []
    spacings = {aid: [] for aid in env.unwrapped.agent_ids}
    latencies_us: list[float] = []

    done = False
    truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action = controller_fn(obs)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

        obs, reward, done, truncated, info = env.step(action)
        cum_reward += reward
        cum_cost += (1.0 - info["r_base"]) * env.unwrapped.J_max_multi  # raw J
        if info["collision"]:
            n_collisions += 1
        if info["violation"] > 0:
            n_violations += 1

        import traci
        for j, aid in enumerate(env.unwrapped.agent_ids):
            if aid in traci.vehicle.getIDList():
                cav_velocities[aid].append(traci.vehicle.getSpeed(aid))
                cav_actions[aid].append(env.unwrapped.prev_accels[aid])
                spacings[aid].append(env.unwrapped._get_gap_to_leader(aid))
        head_velocities.append(
            traci.vehicle.getSpeed("car0")
            if "car0" in traci.vehicle.getIDList() else 0.0
        )

    env.close()

    cav_v_arr = {a: np.asarray(v, dtype=np.float32) for a, v in cav_velocities.items()}
    head_v = np.asarray(head_velocities, dtype=np.float32)
    n = min(len(head_v), min(len(v) for v in cav_v_arr.values()))
    msve = {
        a: float(np.mean((cav_v_arr[a][:n] - head_v[:n]) ** 2))
        for a in cav_v_arr
    }
    msve_avg = float(np.mean(list(msve.values())))

    s_arrs = {a: np.asarray(s, dtype=np.float32) for a, s in spacings.items()}
    min_spacing = float(min(s.min() for s in s_arrs.values())) if s_arrs else 0.0
    max_spacing = float(max(s.max() for s in s_arrs.values())) if s_arrs else 0.0

    return {
        "total_cost": cum_cost,
        "cumulative_reward": cum_reward,
        "msve": msve,
        "msve_avg": msve_avg,
        "n_collisions": n_collisions,
        "n_violations": n_violations,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "mean_latency_us": float(np.mean(latencies_us)) if latencies_us else 0.0,
        "head_velocities": head_v,
        "cav_velocities": cav_v_arr,
        "cav_actions": {a: np.asarray(v) for a, v in cav_actions.items()},
    }


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------


def plot_velocity_comparison(
    scenario_name: str, hdv_label: str,
    results_per_controller: dict[str, dict],
    Tstep: float, out_dir: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    head = next(iter(results_per_controller.values()))["head_velocities"]
    t = np.arange(len(head)) * Tstep
    axes[0].plot(t, head, color="gray", linestyle="--", linewidth=1.2,
                 label="head", alpha=0.8)
    cmap = plt.get_cmap("tab10")
    for i, (ctrl_name, res) in enumerate(results_per_controller.items()):
        for j, aid in enumerate(res["cav_velocities"]):
            v = res["cav_velocities"][aid]
            n = min(len(t), len(v))
            axes[0].plot(t[:n], v[:n],
                         color=cmap(i), linestyle=("-" if j == 0 else ":"),
                         linewidth=1.0,
                         label=f"{ctrl_name} {aid}")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title(f"{scenario_name} / {hdv_label} — CAV velocities")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i, (ctrl_name, res) in enumerate(results_per_controller.items()):
        for j, aid in enumerate(res["cav_actions"]):
            a = res["cav_actions"][aid]
            n = min(len(t), len(a))
            axes[1].plot(t[:n], a[:n],
                         color=cmap(i), linestyle=("-" if j == 0 else ":"),
                         linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Applied a (m/s²)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{scenario_name}_{hdv_label}_velocities.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    nnmpc_cfg = RLMPCConfig(mode="warm_start")  # NNMPC eval uses warm_start env (action box [-5,3])
    warm_cfg = RLMPCConfig(mode="warm_start")
    resid_cfg = RLMPCConfig(mode="residual")

    out_dir = Path("deep_lcc_results/rlmpc/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scenarios
    Tstep = warm_cfg.Tstep
    v_star = warm_cfg.v_star
    scen_specs = {
        "brake": (lambda: make_extreme_brake(int(30.0 / Tstep), Tstep, v_star)),
        "sinusoidal": (lambda: make_sinusoidal(int(40.0 / Tstep), Tstep, v_star, amplitude=2.0)),
        "varying_sine": (lambda: make_varying_sine(200.0, Tstep, v_star)),
        "aggressive_sine": (lambda: make_aggressive_sine(200.0, Tstep, v_star)),
        "stop_and_go": (lambda: make_stop_and_go(200.0, Tstep, v_star)),
        "NEDC": (lambda: make_nedc(Tstep)),
    }
    if "SCENARIOS" in os.environ:
        wanted = {s.strip() for s in os.environ["SCENARIOS"].split(",")}
        scen_specs = {k: v for k, v in scen_specs.items() if k in wanted}

    hdv_configs = {
        "nominal": [42],
        "hetero_fixed": [999],   # deterministic single sample (spec's hetero_fixed)
        "hetero_random": [101, 202, 303, 404, 505],
    }

    # Build controllers
    nnmpc_ctrl = make_nnmpc_controller(nnmpc_cfg.nnmpc_path)
    warm_path = Path(warm_cfg.out_dir.format(mode="warm_start")) / "agent.pth"
    resid_path = Path(resid_cfg.out_dir.format(mode="residual")) / "agent.pth"
    warm_ctrl = make_rl_controller(str(warm_path), warm_cfg) if warm_path.exists() else None
    resid_ctrl = make_rl_controller(str(resid_path), resid_cfg) if resid_path.exists() else None

    controllers = {"nnmpc": (nnmpc_ctrl, nnmpc_cfg)}
    if warm_ctrl is not None:
        controllers["warm_rl"] = (warm_ctrl, warm_cfg)
    if resid_ctrl is not None:
        controllers["rl_residual"] = (resid_ctrl, resid_cfg)

    csv_path = out_dir / "summary.csv"
    rows = []
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "hdv_config", "seed", "controller",
            "total_cost", "cum_reward", "msve_avg",
            "n_collisions", "n_violations",
            "min_spacing", "max_spacing", "mean_latency_us",
        ])

        for scen_name, scen_fn in scen_specs.items():
            head_trace = scen_fn()
            for hdv_label, seeds in hdv_configs.items():
                for seed in seeds:
                    results_per_ctrl = {}
                    for ctrl_name, (ctrl_fn, ctrl_cfg) in controllers.items():
                        print(f"[{scen_name}/{hdv_label}/seed={seed}] running {ctrl_name}")
                        res = run_eval_episode(ctrl_cfg, ctrl_fn, head_trace, seed)
                        results_per_ctrl[ctrl_name] = res
                        writer.writerow([
                            scen_name, hdv_label, seed, ctrl_name,
                            f"{res['total_cost']:.4f}",
                            f"{res['cumulative_reward']:.4f}",
                            f"{res['msve_avg']:.4f}",
                            res['n_collisions'], res['n_violations'],
                            f"{res['min_spacing']:.3f}", f"{res['max_spacing']:.3f}",
                            f"{res['mean_latency_us']:.1f}",
                        ])
                        f.flush()
                        rows.append((scen_name, hdv_label, seed, ctrl_name, res))
                    # One combined plot per (scenario, hdv_label, seed)
                    plot_velocity_comparison(
                        f"{scen_name}", f"{hdv_label}_seed{seed}",
                        results_per_ctrl, Tstep, out_dir,
                    )

    # Render a markdown summary grouped by scenario, ranked by total cost.
    md_path = out_dir / "summary.md"
    with md_path.open("w") as md:
        md.write("# RLMPC Eval Summary\n\n")
        # Group rows by (scenario, hdv_config), then sort by mean cost across seeds.
        from collections import defaultdict
        grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for scen, hdv, seed, ctrl, res in rows:
            grouped[(scen, hdv)][ctrl].append(res["total_cost"])
        for (scen, hdv), per_ctrl in grouped.items():
            md.write(f"## {scen} / {hdv}\n\n")
            md.write("| controller | mean cost | n seeds |\n")
            md.write("|---|---:|---:|\n")
            ranked = sorted(
                per_ctrl.items(), key=lambda kv: float(np.mean(kv[1]))
            )
            for ctrl, costs in ranked:
                md.write(f"| {ctrl} | {float(np.mean(costs)):.2f} | {len(costs)} |\n")
            md.write("\n")
    print(f"[rlmpc_eval] Done. Summary → {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Smoke import.**

Run: `uv run python -c "from rl_mixed_traffic.deep_lcc.rlmpc_eval import main, run_eval_episode, make_nnmpc_controller, make_rl_controller; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 8.3: Commit.**

```bash
git add rl_mixed_traffic/deep_lcc/rlmpc_eval.py
git commit -m "feat: add rlmpc_eval.py (compare NNMPC, warm-RL, RL+NNMPC across scenarios)"
```

---

## Task 9: Smoke test the full training loop

Verify that training produces non-NaN rollouts for ~5000 steps in each mode, and that checkpoints round-trip through `PPOAgent.load`. We don't expect convergence — just absence of crashes / NaNs.

**Files:**
- (no new files; runs against existing scripts)

- [ ] **Step 9.1: Verify NNMPC checkpoint exists.**

Run: `ls -la deep_lcc_results/nnmpc.pth`
Expected: file exists; if not, `uv run rl_mixed_traffic/deep_lcc/nnmpc_train.py` first (per CLAUDE.md).

- [ ] **Step 9.2: Smoke-train warm-start for 5000 steps (no GUI).**

Run:
```
uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode warm_start \
    --total-steps 5000 --rollout-steps 2048 --seed 42
```

Expected: prints per-episode lines; no NaN losses; produces:
- `deep_lcc_results/rlmpc_warm_start/agent.pth`
- `deep_lcc_results/rlmpc_warm_start/returns.csv` with at least a few rows
- `deep_lcc_results/rlmpc_warm_start/returns.png`
- `deep_lcc_results/rlmpc_warm_start/ppo_metrics.png`

If any of these fail or NaN appears in metrics, fix the env/network before continuing. Most likely culprits:
- NNMPC normalization drift (input_mean/std unsuitable for SUMO data) → relax `actor_log_std` clamp to `[-3, 0.5]` or set `log_std_init_warm = log(0.05)`.
- HDV randomization too aggressive → narrow `hdv_decel_range` to `(2.8, 3.2)`.
- Spacing-violation Lagrangian dominates → set `lambda_violation = 0.1`.

- [ ] **Step 9.3: Smoke-train residual for 5000 steps (no GUI).**

Run:
```
uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode residual \
    --total-steps 5000 --rollout-steps 2048 --seed 42
```

Expected: same artifacts under `deep_lcc_results/rlmpc_residual/`.

- [ ] **Step 9.4: Verify checkpoint round-trip.**

Run:
```
uv run python -c "
import numpy as np
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_eval import make_rl_controller

cfg_w = RLMPCConfig(mode='warm_start')
ctrl_w = make_rl_controller('deep_lcc_results/rlmpc_warm_start/agent.pth', cfg_w)
out = ctrl_w(np.zeros(260, dtype=np.float32))
assert out.shape == (2,)
assert (-5.0 <= out).all() and (out <= 3.0).all()

cfg_r = RLMPCConfig(mode='residual')
ctrl_r = make_rl_controller('deep_lcc_results/rlmpc_residual/agent.pth', cfg_r)
out = ctrl_r(np.zeros(260, dtype=np.float32))
assert out.shape == (2,)
assert (-2.0 <= out).all() and (out <= 2.0).all()
print('checkpoint round-trip ok')
"
```
Expected: `checkpoint round-trip ok`

- [ ] **Step 9.5: Smoke-eval.**

Run: `SCENARIOS=brake uv run rl_mixed_traffic/deep_lcc/rlmpc_eval.py`

Expected:
- `deep_lcc_results/rlmpc/summary.csv` with rows for `brake / nominal / 42 / {nnmpc, warm_rl, rl_residual}`.
- `deep_lcc_results/rlmpc/brake_nominal_seed42_velocities.png`.
- No crashes; metrics finite.

- [ ] **Step 9.6: Commit smoke artifacts as a marker (optional — usually we don't commit artifacts).**

Skip committing the artifact files; just confirm the pipeline runs end-to-end. If everything passes, commit any minor fixes you made during smoke testing:

```bash
git status
# If there are uncommitted fixes:
git add <whatever was fixed>
git commit -m "fix: address smoke-test issues in RLMPC pipeline"
```

---

## Task 10: Run full training and full eval

Once the smoke test passes, kick off the full 1M-step training runs. This is single-machine, sequential — expect 20–40 hours total.

- [ ] **Step 10.1: Full warm-start training.**

Run:
```
uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode warm_start \
    --total-steps 1000000 --rollout-steps 4096 --seed 42
```

Watch:
- Returns trend upward after the first ~50 episodes.
- Mean violation per step → 0 over time.
- No collisions in the last few hundred episodes.

If divergence in the first ~100 episodes, set `RLMPCConfig.critic_warmup_updates = 5` and re-run from a checkpoint.

- [ ] **Step 10.2: Full residual training.**

```
uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode residual \
    --total-steps 1000000 --rollout-steps 4096 --seed 42
```

- [ ] **Step 10.3: Full eval grid.**

```
uv run rl_mixed_traffic/deep_lcc/rlmpc_eval.py
```

Inspect `deep_lcc_results/rlmpc/summary.csv` and `deep_lcc_results/rlmpc/*.png`.

- [ ] **Step 10.4: Commit results (.csv + .png artifacts) as a checkpoint.**

```bash
git add deep_lcc_results/rlmpc/summary.csv deep_lcc_results/rlmpc/*.png \
        deep_lcc_results/rlmpc_warm_start/returns.csv deep_lcc_results/rlmpc_warm_start/*.png \
        deep_lcc_results/rlmpc_residual/returns.csv deep_lcc_results/rlmpc_residual/*.png
git commit -m "results: RLMPC training + eval (warm-start, residual)"
```

---

## Acceptance check

Per spec Section "Acceptance criteria":

1. ✅ Both training runs complete (Tasks 10.1–10.2 produce `agent.pth` for both modes).
2. ✅ Return curves show upward trend after the first ~100 episodes (`returns.png` per mode).
3. ✅ Eval grid produces complete `summary.csv` and per-scenario plots.
4. **Goal-A/B test**: at least one of {warm-start, residual} achieves lower total cost than NNMPC averaged across `hetero_random` (compute from `summary.csv`).
5. **Safety regression test**: collision count for RL controllers ≤ NNMPC's across all eval cells.
6. **Latency check**: RL controllers' `mean_latency_us` ≤ 2× NNMPC's.

If 4 or 5 fails, investigate:
- 4 fails: try lower `lambda_violation` (RL might be over-cautious), more total steps, or check if NNMPC normalization is fighting the policy. Document as a negative result if root cause is benign.
- 5 fails: spec calls this a regression; tighten HDV ranges, increase `lambda_violation`, or add stronger collision-prevention shaping. Re-train.

---

## Glossary of project conventions referenced

- **Speed mode 95** = `0b1011111` — disables most SUMO safety checks except the front-collision braking; lets us issue acceleration commands directly.
- **`apply_acceleration(vid, a, smooth=False)`** = inherited from `RingRoadEnv`; clips `a` to `[min_accel, max_accel]`, integrates `v_next = clip(v + a*dt, 0, v_max)`, calls `setSpeed(vid, v_next)`.
- **`measure_mixed_traffic(..., measure_type=3)`** returns `[v_1−v★, …, v_8−v★, s_3−s★, s_6−s★]` (length 10).
- **`v_eq`** = 20-second rolling average of head speed; used as the equilibrium in y, e error computations.
- **`compute_multi_agent_lcc_reward()`** (in `RingRoadEnv`) returns `r_base ∈ [0, 1]` from `J_velocity + J_spacing + J_control`.
- **`get_spacing_violation()`** returns the sum across CAVs of `max(0, s_min − gap_i) / s_min`.
