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
