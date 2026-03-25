import numpy as np

from rl_mixed_traffic.deep_lcc.config import OVMConfig


def hdv_dynamics(S: np.ndarray, config: OVMConfig) -> np.ndarray:
    """Compute accelerations for all following vehicles using the OVM model.

    Port of HDV_dynamics.m from the DeeP-LCC MATLAB implementation.

    Args:
        S: State array of shape (n_vehicle + 1, 3).
            Axis 1: [position, velocity, acceleration].
            Index 0 is the head vehicle; indices 1..n are followers.
        config: OVM parameters.

    Returns:
        acel: Accelerations for vehicles 1..n, shape (n_vehicle,).
    """
    n_vehicle = S.shape[0] - 1

    # Velocity difference: leader - follower  (positive when leader is faster)
    V_diff = S[:-1, 1] - S[1:, 1]  # shape (n_vehicle,)

    # Spacing: leader_pos - follower_pos  (positive gap)
    D_diff = S[:-1, 0] - S[1:, 0]  # shape (n_vehicle,)

    # Clamp spacing for OVM desired-velocity calculation
    s_go = config.s_go
    if np.isscalar(s_go):
        s_go = np.full(n_vehicle, s_go)
    else:
        s_go = np.asarray(s_go).ravel()[:n_vehicle]

    cal_D = np.clip(D_diff, config.s_st, s_go)

    # Desired velocity: V_d = v_max/2 * (1 - cos(pi * (s - s_st) / (s_go - s_st)))
    V_d = config.v_max / 2.0 * (
        1.0 - np.cos(np.pi * (cal_D - config.s_st) / (s_go - config.s_st))
    )

    # OVM acceleration: a = alpha * (V_d - v) + beta * (v_leader - v)
    v_followers = S[1:, 1]
    acel = config.alpha * (V_d - v_followers) + config.beta * V_diff

    # Acceleration saturation
    acel = np.clip(acel, config.dcel_max, config.acel_max)

    # Safety braking (SD / ADAS): prevent crashes
    # If deceleration needed exceeds |dcel_max|, apply emergency braking
    with np.errstate(divide="ignore", invalid="ignore"):
        acel_sd = (S[1:, 1] ** 2 - S[:-1, 1] ** 2) / (2.0 * D_diff)
    acel[acel_sd > abs(config.dcel_max)] = config.dcel_max

    return acel
