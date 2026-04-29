"""Intelligent Driver Model (IDM) car-following dynamics.

Drop-in alternative to `rl_mixed_traffic.deep_lcc.ovm.hdv_dynamics` with an
identical `(S, config) -> acel` contract so it can be plugged into
`run_with_state` via its `hdv_dynamics_fn` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IDMConfig:
    """IDM parameters. Kept structurally similar to OVMConfig so downstream
    code (state init, constraint limits, CAV ID mask) can treat both alike.
    """

    # IDM-specific
    v0: float = 30.0      # desired speed (m/s)
    T: float = 1.0        # safe time headway (s)
    a: float = 1.0        # max acceleration (m/s^2)
    b: float = 1.5        # comfortable deceleration (m/s^2)
    s0: float = 5.0       # standstill gap (m)
    delta: float = 4.0    # acceleration exponent

    # Shared with OVMConfig (used by run_with_state / measurement code)
    n_vehicle: int = 8
    ID: list[int] = field(default_factory=lambda: [0, 0, 1, 0, 0, 1, 0, 0])
    acel_max: float = 2.0
    dcel_max: float = -5.0
    s_st: float = 5.0
    s_go: float = 35.0
    v_max: float = 30.0


def get_default_idm_config() -> IDMConfig:
    """Return the tuned IDM parameters defined in the design spec.

    Equilibrium at v=15 m/s gives s_eq ≈ 20.66 m, close to s_star=20 m.
    """
    return IDMConfig()


def idm_dynamics(S: np.ndarray, config: IDMConfig) -> np.ndarray:
    """Compute IDM accelerations for all following vehicles.

    Args:
        S: shape (n_vehicle + 1, 3). Axis 1 is [position, velocity, acceleration].
            Index 0 is the head vehicle.
        config: IDM parameters.

    Returns:
        acel: shape (n_vehicle,). Accelerations for followers 1..n_vehicle.
    """
    # Leader − follower → positive gap, positive approach velocity
    D_diff = S[:-1, 0] - S[1:, 0]
    V_leader = S[:-1, 1]
    V_follow = S[1:, 1]
    Delta_v = V_follow - V_leader

    s_star_dyn = config.s0 + np.maximum(
        0.0,
        V_follow * config.T
        + V_follow * Delta_v / (2.0 * np.sqrt(config.a * config.b)),
    )

    safe_gap = np.maximum(D_diff, 1e-3)

    acel = config.a * (
        1.0 - (V_follow / config.v0) ** config.delta - (s_star_dyn / safe_gap) ** 2
    )

    acel = np.clip(acel, config.dcel_max, config.acel_max)

    # ADAS safety clamp (matches ovm.hdv_dynamics)
    with np.errstate(divide="ignore", invalid="ignore"):
        acel_sd = (V_follow**2 - V_leader**2) / (2.0 * D_diff)
    acel[acel_sd > abs(config.dcel_max)] = config.dcel_max

    return acel
