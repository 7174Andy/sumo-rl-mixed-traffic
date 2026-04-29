"""Failure-mode scenarios for DeeP-LCC evaluation.

Provides:
- SEVERITY_RANGES: α/β/s_go distributions for mild / medium / severe non-stationary HDV behavior.
- make_ovm_resampler: factory for a per-step OVMConfig callable.
"""

from __future__ import annotations

SEVERITY_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "mild": {
        "alpha": (0.4, 0.8),
        "beta": (0.6, 1.0),
        "s_go": (30.0, 40.0),
    },
    "medium": {
        "alpha": (0.3, 0.9),
        "beta": (0.5, 1.2),
        "s_go": (25.0, 45.0),
    },
    "severe": {
        "alpha": (0.2, 1.0),
        "beta": (0.4, 1.5),
        "s_go": (20.0, 50.0),
    },
}


from typing import Callable

import numpy as np

from rl_mixed_traffic.deep_lcc.config import OVMConfig

# CAV nominal values (matches the reference hdv_ovm_2.mat for positions 3 and 6)
_CAV_ALPHA = 0.6
_CAV_BETA = 0.9
_CAV_S_GO = 35.0

# CAV positions (0-indexed) in the default 8-vehicle platoon
# ID = [0, 0, 1, 0, 0, 1, 0, 0]
_CAV_IDX = (2, 5)
_N_VEHICLE = 8

# Resampling cadence range (seconds, uniform per-HDV per-sample)
_RESAMPLE_MIN = 15.0
_RESAMPLE_MAX = 25.0


def make_ovm_resampler(severity: str, seed: int) -> Callable[[float], OVMConfig]:
    """Return a callable `ovm_at(t)` producing a time-varying OVMConfig.

    Each of the 6 non-CAV HDVs carries independent (alpha, beta, s_go) that are
    resampled every U[15, 25] s (per-HDV clock). CAV positions (indices
    2 and 5, i.e. 1-indexed positions 3 and 6) always hold nominal values
    (0.6, 0.9, 35).

    Args:
        severity: one of "mild", "medium", "severe". Controls the distribution
            bounds for alpha, beta, s_go.
        seed: RNG seed for reproducibility. Drives both resampling cadence
            and parameter draws.

    Returns:
        Callable mapping time t (seconds) to an OVMConfig reflecting the
        instantaneous HDV parameters.
    """
    if severity not in SEVERITY_RANGES:
        raise ValueError(
            f"Unknown severity {severity!r}. Expected one of "
            f"{sorted(SEVERITY_RANGES)}."
        )
    ranges = SEVERITY_RANGES[severity]
    rng = np.random.default_rng(seed)

    # Per-HDV current params and next-resample-time.
    # Only positions NOT in _CAV_IDX are resampled; CAV positions stay nominal.
    alpha_arr = [_CAV_ALPHA] * _N_VEHICLE
    beta_arr = [_CAV_BETA] * _N_VEHICLE
    s_go_arr = [_CAV_S_GO] * _N_VEHICLE
    next_resample = [0.0] * _N_VEHICLE  # HDVs resample at t=0 on first call

    def _sample_for(i: int) -> None:
        alpha_arr[i] = float(rng.uniform(*ranges["alpha"]))
        beta_arr[i] = float(rng.uniform(*ranges["beta"]))
        s_go_arr[i] = float(rng.uniform(*ranges["s_go"]))

    def _schedule_next(i: int, t_now: float) -> None:
        next_resample[i] = t_now + float(rng.uniform(_RESAMPLE_MIN, _RESAMPLE_MAX))

    def ovm_at(t: float) -> OVMConfig:
        for i in range(_N_VEHICLE):
            if i in _CAV_IDX:
                continue
            if t >= next_resample[i]:
                _sample_for(i)
                _schedule_next(i, t)
        return OVMConfig(
            alpha=list(alpha_arr),
            beta=list(beta_arr),
            s_go=list(s_go_arr),
        )

    return ovm_at
