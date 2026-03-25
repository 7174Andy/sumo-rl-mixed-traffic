"""Generate (state, solution) dataset for NNMPC training.

Runs DeeP-LCC in closed loop with OVM traffic simulation, recording
(uini, yini, eini) → u_opt pairs at each step.

Usage:
    uv run rl_mixed_traffic/deep_lcc/generate_dataset.py
"""

import os
from pathlib import Path

import numpy as np
from tqdm import trange

from rl_mixed_traffic.deep_lcc.config import DeepLCCConfig, OVMConfig
from rl_mixed_traffic.deep_lcc.measurement import measure_mixed_traffic
from rl_mixed_traffic.deep_lcc.ovm import hdv_dynamics
from rl_mixed_traffic.deep_lcc.precollect import precollect
from rl_mixed_traffic.deep_lcc.qp_solver import CachedDeepLCCSolver


def _assign_episode_amplitudes(
    num_episodes: int,
    perturb_mix: list[tuple[float, float]],
) -> list[float]:
    """Assign a perturbation amplitude to each episode based on the mix ratios.

    Episodes are assigned in contiguous blocks so the mix is exact
    (up to rounding). Rounding residuals go to the last tier.
    """
    amplitudes: list[float] = []
    remaining = num_episodes
    for i, (amp, frac) in enumerate(perturb_mix):
        if i == len(perturb_mix) - 1:
            count = remaining
        else:
            count = round(num_episodes * frac)
            remaining -= count
        amplitudes.extend([amp] * count)
    return amplitudes


def _build_weight_matrices(
    config: DeepLCCConfig, n_vehicle: int, m_ctr: int, p_ctr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build Q and R weight matrices for the DeeP-LCC cost."""
    # Q = block_diag(weight_v * I_{n_vehicle}, weight_s * I_{m_ctr})
    Q_v = config.weight_v * np.eye(n_vehicle)
    if config.measure_type == 1:
        Q = Q_v
    elif config.measure_type == 2:
        Q_s = config.weight_s * np.eye(n_vehicle)
        Q = np.block([
            [Q_v, np.zeros((n_vehicle, n_vehicle))],
            [np.zeros((n_vehicle, n_vehicle)), Q_s],
        ])
    elif config.measure_type == 3:
        Q_s = config.weight_s * np.eye(m_ctr)
        Q = np.block([
            [Q_v, np.zeros((n_vehicle, m_ctr))],
            [np.zeros((m_ctr, n_vehicle)), Q_s],
        ])
    else:
        raise ValueError(f"Unknown measure_type: {config.measure_type}")

    R = config.weight_u * np.eye(m_ctr)
    return Q, R


def run_deep_lcc_episode(
    Up: np.ndarray,
    Uf: np.ndarray,
    Ep: np.ndarray,
    Ef: np.ndarray,
    Yp: np.ndarray,
    Yf: np.ndarray,
    config: DeepLCCConfig,
    ovm_config: OVMConfig,
    Q: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    perturb_amplitude: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run one episode of DeeP-LCC control and collect (state, solution) pairs.

    Args:
        perturb_amplitude: Head vehicle velocity perturbation range (±amplitude m/s).

    Returns lists of (uini, yini, eini, u_opt) arrays for each valid step.
    """
    n_vehicle = ovm_config.n_vehicle
    ID = ovm_config.ID
    pos_cav = np.where(np.array(ID) == 1)[0]
    m_ctr = len(pos_cav)

    if config.measure_type == 3:
        p_ctr = n_vehicle + m_ctr
    elif config.measure_type == 2:
        p_ctr = 2 * n_vehicle
    else:
        p_ctr = n_vehicle

    total_steps = int(config.total_time / config.Tstep)
    T_ini = config.T_ini
    N = config.N

    # Constraint limits
    u_limit = (config.dcel_max, config.acel_max)
    s_limit = (config.spacing_min - config.s_star, config.spacing_max - config.s_star)

    # Build cached parametric solver once for the entire episode
    solver = CachedDeepLCCSolver(
        Up, Yp, Uf, Yf, Ep, Ef, Q, R,
        config.lambda_g, config.lambda_y,
        u_limit=u_limit, s_limit=s_limit,
    )

    # Initialize vehicles at equilibrium
    S = np.zeros((n_vehicle + 1, 3))
    S[0, 0] = 0.0
    for i in range(1, n_vehicle + 1):
        S[i, 0] = S[i - 1, 0] - config.s_star
    S[:, 1] = config.v_star

    # Rolling buffers for past data
    u_buffer = np.zeros((m_ctr, T_ini))
    y_buffer = np.zeros((p_ctr, T_ini))
    e_buffer = np.zeros(T_ini)

    # Head vehicle disturbance for this episode (random perturbation)
    ed_episode = -perturb_amplitude + 2.0 * perturb_amplitude * rng.random(total_steps)

    # Collected data
    uini_list = []
    yini_list = []
    eini_list = []
    u_opt_list = []

    for k in range(total_steps):
        # Measure current output
        y_k = measure_mixed_traffic(
            S[1:, 1], S[:, 0], ID, config.v_star, config.s_star, config.measure_type
        )
        e_k = S[0, 1] - config.v_star  # head vehicle disturbance

        # Update rolling buffers (shift left, append new)
        u_buffer = np.roll(u_buffer, -1, axis=1)
        y_buffer = np.roll(y_buffer, -1, axis=1)
        e_buffer = np.roll(e_buffer, -1)
        y_buffer[:, -1] = y_k
        e_buffer[-1] = e_k

        if k >= T_ini:
            # Flatten buffers column-major (Fortran order) to match MATLAB
            uini = u_buffer.flatten(order="F")
            yini = y_buffer.flatten(order="F")
            eini = e_buffer.copy()

            # Solve DeeP-LCC QP (reuses compiled problem)
            u_opt, y_opt, status = solver(uini, yini, eini)

            if status in ("optimal", "optimal_inaccurate"):
                # Record (state, solution) pair
                uini_list.append(uini.copy())
                yini_list.append(yini.copy())
                eini_list.append(eini.copy())
                # First control action only (receding horizon)
                u_opt_list.append(u_opt[:m_ctr].copy())

                # Apply optimal control to CAVs
                cav_accel = u_opt[:m_ctr]
            else:
                # Fallback: zero acceleration
                cav_accel = np.zeros(m_ctr)
        else:
            # Warm-up: use small random acceleration
            cav_accel = 0.1 * rng.standard_normal(m_ctr)

        # HDV dynamics + noise, clamped to acceleration limits
        acel = hdv_dynamics(S, ovm_config)
        noise = -config.acel_noise + 2.0 * config.acel_noise * rng.random(n_vehicle)
        acel = np.clip(acel + noise, ovm_config.dcel_max, ovm_config.acel_max)

        # Set accelerations
        S[0, 2] = 0.0
        S[1:, 2] = acel
        for j, cav_idx in enumerate(pos_cav):
            S[cav_idx + 1, 2] = float(cav_accel[j])

        # Update control buffer with applied action
        u_buffer[:, -1] = cav_accel

        # Euler integration
        S_new = S.copy()
        S_new[:, 1] = S[:, 1] + config.Tstep * S[:, 2]
        S_new[0, 1] = ed_episode[k] + config.v_star
        S_new[:, 0] = S[:, 0] + config.Tstep * S[:, 1]

        S = S_new

    return uini_list, yini_list, eini_list, u_opt_list


def generate_dataset(
    config: DeepLCCConfig | None = None,
    ovm_config: OVMConfig | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Full pipeline: pre-collect, then run DeeP-LCC episodes to generate dataset.

    Args:
        config: DeeP-LCC parameters (uses defaults if None).
        ovm_config: OVM parameters (uses defaults if None).
        seed: Base random seed.

    Returns:
        Dictionary with keys: uini, yini, eini, u_opt, metadata.
    """
    if config is None:
        config = DeepLCCConfig()
    if ovm_config is None:
        ovm_config = OVMConfig()

    n_vehicle = ovm_config.n_vehicle
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    if config.measure_type == 3:
        p_ctr = n_vehicle + m_ctr
    elif config.measure_type == 2:
        p_ctr = 2 * n_vehicle
    else:
        p_ctr = n_vehicle

    Q, R = _build_weight_matrices(config, n_vehicle, m_ctr, p_ctr)
    episode_amplitudes = _assign_episode_amplitudes(
        config.num_episodes, config.perturb_mix,
    )

    all_uini = []
    all_yini = []
    all_eini = []
    all_u_opt = []

    for episode in trange(config.num_episodes, desc="Generating dataset"):
        ep_seed = seed + episode
        amp = episode_amplitudes[episode]

        # Phase 1: Pre-collect trajectories and build Hankel matrices
        Up, Uf, Ep, Ef, Yp, Yf = precollect(config, ovm_config, seed=ep_seed)

        # Phase 2: Run DeeP-LCC in closed loop
        rng = np.random.default_rng(ep_seed + 10000)
        uini_list, yini_list, eini_list, u_opt_list = run_deep_lcc_episode(
            Up, Uf, Ep, Ef, Yp, Yf,
            config, ovm_config, Q, R, rng,
            perturb_amplitude=amp,
        )

        all_uini.extend(uini_list)
        all_yini.extend(yini_list)
        all_eini.extend(eini_list)
        all_u_opt.extend(u_opt_list)

    dataset = {
        "uini": np.array(all_uini),
        "yini": np.array(all_yini),
        "eini": np.array(all_eini),
        "u_opt": np.array(all_u_opt),
        "metadata": np.array([
            config.v_star, config.s_star, config.T_ini, config.N,
            config.lambda_g, config.lambda_y,
        ]),
    }

    return dataset


def main():
    config = DeepLCCConfig()
    ovm_config = OVMConfig()

    dataset = generate_dataset(config, ovm_config)

    # Save
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **dataset)

    n_samples = dataset["uini"].shape[0]
    print(f"Dataset saved to {out_path}")
    print(f"  Samples: {n_samples}")
    print(f"  uini shape: {dataset['uini'].shape}")
    print(f"  yini shape: {dataset['yini'].shape}")
    print(f"  eini shape: {dataset['eini'].shape}")
    print(f"  u_opt shape: {dataset['u_opt'].shape}")


if __name__ == "__main__":
    main()
