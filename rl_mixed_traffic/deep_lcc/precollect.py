import numpy as np

from rl_mixed_traffic.deep_lcc.config import DeepLCCConfig, OVMConfig
from rl_mixed_traffic.deep_lcc.hankel import build_hankel_matrices
from rl_mixed_traffic.deep_lcc.measurement import measure_mixed_traffic
from rl_mixed_traffic.deep_lcc.ovm import hdv_dynamics


def precollect(
    config: DeepLCCConfig,
    ovm_config: OVMConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run OVM simulation with persistently exciting inputs and build Hankel matrices.

    Port of data_trajectoryDataCollection.m from the DeeP-LCC MATLAB implementation.

    Args:
        config: DeeP-LCC parameters (T, T_ini, N, v_star, etc.).
        ovm_config: OVM car-following model parameters.
        seed: Random seed for reproducibility.

    Returns:
        (Up, Uf, Ep, Ef, Yp, Yf) — Hankel sub-matrices for DeeP-LCC.
    """
    rng = np.random.default_rng(seed)

    n_vehicle = ovm_config.n_vehicle
    ID = ovm_config.ID
    pos_cav = np.where(np.array(ID) == 1)[0]  # 0-indexed positions of CAVs
    m_ctr = len(pos_cav)  # number of CAV control inputs

    # Determine output dimension
    if config.measure_type == 1:
        p_ctr = n_vehicle
    elif config.measure_type == 2:
        p_ctr = 2 * n_vehicle
    elif config.measure_type == 3:
        p_ctr = n_vehicle + m_ctr
    else:
        raise ValueError(f"Unknown measure_type: {config.measure_type}")

    T = config.T

    # State array: (n_vehicle + 1, 3) — [position, velocity, acceleration]
    # Index 0 = head vehicle, 1..n_vehicle = followers
    S = np.zeros((n_vehicle + 1, 3))

    # Initialize at equilibrium: uniform spacing, all at v_star
    S[0, 0] = 0.0
    for i in range(1, n_vehicle + 1):
        S[i, 0] = S[i - 1, 0] - config.s_star
    S[:, 1] = config.v_star

    # Persistently exciting inputs
    ud = -1.0 + 2.0 * rng.random((m_ctr, T))  # CAV accelerations
    ed = -1.0 + 2.0 * rng.random((1, T))  # head vehicle disturbance
    yd = np.zeros((p_ctr, T))

    # Simulation loop
    for k in range(T - 1):
        # HDV accelerations from OVM + noise, clamped to acceleration limits
        acel = hdv_dynamics(S, ovm_config)
        noise = -config.acel_noise + 2.0 * config.acel_noise * rng.random(n_vehicle)
        acel = np.clip(acel + noise, ovm_config.dcel_max, ovm_config.acel_max)

        # Set accelerations
        S[0, 2] = 0.0  # head vehicle (acceleration not used directly)
        S[1:, 2] = acel  # all followers use HDV model
        # Override CAV positions with PE input
        for j, cav_idx in enumerate(pos_cav):
            S[cav_idx + 1, 2] = ud[j, k]  # +1 because S includes head at index 0

        # Euler integration
        S_new = S.copy()
        S_new[:, 1] = S[:, 1] + config.Tstep * S[:, 2]  # velocity update
        S_new[0, 1] = ed[0, k] + config.v_star  # head vehicle velocity from PE
        S_new[:, 0] = S[:, 0] + config.Tstep * S[:, 1]  # position update

        # Measure output (vel of followers, pos of all)
        yd[:, k] = measure_mixed_traffic(
            S[1:, 1], S[:, 0], ID, config.v_star, config.s_star, config.measure_type
        )

        S = S_new

    # Final measurement
    yd[:, T - 1] = measure_mixed_traffic(
        S[1:, 1], S[:, 0], ID, config.v_star, config.s_star, config.measure_type
    )

    # Build Hankel matrices
    Up, Uf, Ep, Ef, Yp, Yf = build_hankel_matrices(ud, ed, yd, config.T_ini, config.N)

    return Up, Uf, Ep, Ef, Yp, Yf
