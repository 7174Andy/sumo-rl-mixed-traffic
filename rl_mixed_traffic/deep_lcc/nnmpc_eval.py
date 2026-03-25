"""Evaluate NNMPC against the DeeP-LCC QP solver.

Two modes:
  1. Offline accuracy on held-out test data
  2. Closed-loop simulation comparison (NN vs QP)

Usage:
    uv run rl_mixed_traffic/deep_lcc/nnmpc_eval.py
"""

from pathlib import Path

import numpy as np
import torch

from rl_mixed_traffic.deep_lcc.config import DeepLCCConfig, OVMConfig
from rl_mixed_traffic.deep_lcc.measurement import measure_mixed_traffic
from rl_mixed_traffic.deep_lcc.nnmpc_config import NNMPCConfig
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork
from rl_mixed_traffic.deep_lcc.ovm import hdv_dynamics
from rl_mixed_traffic.deep_lcc.precollect import precollect
from rl_mixed_traffic.deep_lcc.qp_solver import CachedDeepLCCSolver


def load_model(
    model_path: str, device: torch.device
) -> tuple[NNMPCNetwork, np.ndarray, np.ndarray]:
    """Load trained NNMPC model and normalization stats."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = NNMPCNetwork(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dims=ckpt["config"]["hidden_dims"],
        accel_min=ckpt["config"]["accel_min"],
        accel_max=ckpt["config"]["accel_max"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["input_mean"], ckpt["input_std"]


def nn_predict(
    model: NNMPCNetwork,
    uini: np.ndarray,
    yini: np.ndarray,
    eini: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run NN inference for a single step."""
    x = np.concatenate([uini, yini, eini]).astype(np.float32)
    x = (x - input_mean) / input_std
    with torch.no_grad():
        pred = model(torch.from_numpy(x).unsqueeze(0).to(device))
    return pred.cpu().numpy().ravel()


# ── Offline evaluation ────────────────────────────────────────────────


def eval_offline(config: NNMPCConfig) -> None:
    """Compare NN predictions against QP solutions on held-out data."""
    device = torch.device(config.device)
    model, input_mean, input_std = load_model(config.model_path, device)

    data = np.load(config.dataset_path)
    X = np.hstack([data["uini"], data["yini"], data["eini"]]).astype(np.float32)
    y = data["u_opt"].astype(np.float32)

    # Use same shuffle as training to get the val split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_val = int(len(X) * config.val_split)
    X_val, y_val = X[:n_val], y[:n_val]

    # Normalize
    X_val_norm = (X_val - input_mean) / input_std

    # Predict
    with torch.no_grad():
        X_t = torch.from_numpy(X_val_norm).to(device)
        preds = model(X_t).cpu().numpy()

    errors = preds - y_val
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))

    print("=== Offline Evaluation (Val Set) ===")
    print(f"  Samples:   {len(y_val)}")
    print(f"  MSE:       {mse:.6f}")
    print(f"  MAE:       {mae:.6f}")
    print(f"  Max error: {max_err:.4f}")
    print(f"  Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  True range: [{y_val.min():.3f}, {y_val.max():.3f}]")


# ── Closed-loop evaluation ────────────────────────────────────────────


def run_closed_loop(
    controller_fn,
    config: DeepLCCConfig,
    ovm_config: OVMConfig,
    Q: np.ndarray,
    R: np.ndarray,
    ed_episode: np.ndarray,
) -> tuple[float, int]:
    """Run one closed-loop episode with a given controller.

    Args:
        controller_fn: Callable(uini, yini, eini) → u_cav (m_ctr,)

    Returns:
        (total_cost, n_steps)
    """
    n_vehicle = ovm_config.n_vehicle
    ID = ovm_config.ID
    pos_cav = np.where(np.array(ID) == 1)[0]
    m_ctr = len(pos_cav)
    p_ctr = n_vehicle + m_ctr  # measure_type=3

    total_steps = int(config.total_time / config.Tstep)
    T_ini = config.T_ini

    # Initialize vehicles at equilibrium
    S = np.zeros((n_vehicle + 1, 3))
    S[0, 0] = 0.0
    for i in range(1, n_vehicle + 1):
        S[i, 0] = S[i - 1, 0] - config.s_star
    S[:, 1] = config.v_star

    u_buffer = np.zeros((m_ctr, T_ini))
    y_buffer = np.zeros((p_ctr, T_ini))
    e_buffer = np.zeros(T_ini)

    total_cost = 0.0
    n_control_steps = 0

    for k in range(total_steps):
        y_k = measure_mixed_traffic(
            S[1:, 1], S[:, 0], ID, config.v_star, config.s_star, config.measure_type
        )
        e_k = S[0, 1] - config.v_star

        u_buffer = np.roll(u_buffer, -1, axis=1)
        y_buffer = np.roll(y_buffer, -1, axis=1)
        e_buffer = np.roll(e_buffer, -1)
        y_buffer[:, -1] = y_k
        e_buffer[-1] = e_k

        if k >= T_ini:
            uini = u_buffer.flatten(order="F")
            yini = y_buffer.flatten(order="F")
            eini = e_buffer.copy()

            cav_accel = controller_fn(uini, yini, eini)

            # Accumulate cost: y_k' Q y_k + u_k' R u_k (single step)
            total_cost += float(y_k @ Q @ y_k + cav_accel @ R[:m_ctr, :m_ctr] @ cav_accel)
            n_control_steps += 1
        else:
            cav_accel = np.zeros(m_ctr)

        # HDV dynamics
        acel = hdv_dynamics(S, ovm_config)
        noise_rng = np.random.default_rng(k)
        noise = -config.acel_noise + 2.0 * config.acel_noise * noise_rng.random(n_vehicle)
        acel = np.clip(acel + noise, ovm_config.dcel_max, ovm_config.acel_max)

        S[0, 2] = 0.0
        S[1:, 2] = acel
        for j, cav_idx in enumerate(pos_cav):
            S[cav_idx + 1, 2] = float(cav_accel[j])

        u_buffer[:, -1] = cav_accel

        S_new = S.copy()
        S_new[:, 1] = S[:, 1] + config.Tstep * S[:, 2]
        S_new[0, 1] = ed_episode[k] + config.v_star
        S_new[:, 0] = S[:, 0] + config.Tstep * S[:, 1]
        S = S_new

    return total_cost, n_control_steps


def eval_closed_loop(nnmpc_config: NNMPCConfig) -> None:
    """Compare NN vs QP in closed-loop on test scenarios."""
    device = torch.device(nnmpc_config.device)
    model, input_mean, input_std = load_model(nnmpc_config.model_path, device)

    config = DeepLCCConfig()
    ovm_config = OVMConfig()

    n_vehicle = ovm_config.n_vehicle
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    Q_v = config.weight_v * np.eye(n_vehicle)
    Q_s = config.weight_s * np.eye(m_ctr)
    Q = np.block([
        [Q_v, np.zeros((n_vehicle, m_ctr))],
        [np.zeros((m_ctr, n_vehicle)), Q_s],
    ])
    R = config.weight_u * np.eye(m_ctr)

    total_steps = int(config.total_time / config.Tstep)

    # Test scenarios
    scenarios = {
        "random_±1": lambda rng: -1.0 + 2.0 * rng.random(total_steps),
        "random_±5": lambda rng: -5.0 + 10.0 * rng.random(total_steps),
        "brake": lambda _: _make_brake_perturbation(total_steps, config.Tstep),
        "sinusoidal": lambda _: _make_sinusoidal_perturbation(total_steps, config.Tstep),
        "NEDC": lambda _: _make_nedc_perturbation(total_steps, config.Tstep),
    }

    print("\n=== Closed-Loop Evaluation ===")
    print(f"{'Scenario':<15} {'QP Cost':>10} {'NN Cost':>10} {'Diff %':>8}")
    print("-" * 48)

    for name, ed_fn in scenarios.items():
        rng = np.random.default_rng(123)
        ed_episode = ed_fn(rng)

        # QP controller
        Up, Uf, Ep, Ef, Yp, Yf = precollect(config, ovm_config, seed=999)
        solver = CachedDeepLCCSolver(
            Up, Yp, Uf, Yf, Ep, Ef, Q, R,
            config.lambda_g, config.lambda_y,
            u_limit=(config.dcel_max, config.acel_max),
            s_limit=(config.spacing_min - config.s_star, config.spacing_max - config.s_star),
        )

        def qp_controller(uini, yini, eini):
            u_opt, _, status = solver(uini, yini, eini)
            if status in ("optimal", "optimal_inaccurate"):
                return u_opt[:m_ctr]
            return np.zeros(m_ctr)

        qp_cost, _ = run_closed_loop(
            qp_controller, config, ovm_config, Q, R, ed_episode
        )

        # NN controller
        def nn_controller(uini, yini, eini):
            return nn_predict(model, uini, yini, eini, input_mean, input_std, device)

        nn_cost, _ = run_closed_loop(
            nn_controller, config, ovm_config, Q, R, ed_episode
        )

        diff_pct = (nn_cost - qp_cost) / max(abs(qp_cost), 1e-8) * 100
        print(f"{name:<15} {qp_cost:10.2f} {nn_cost:10.2f} {diff_pct:+7.2f}%")


def _make_brake_perturbation(total_steps: int, tstep: float) -> np.ndarray:
    """Brake scenario matching the DeeP-LCC reference (per_type=2).

    Head vehicle: brake at -5 m/s² for 2s, coast 5s, accelerate at +2 m/s² for 5s.
    Perturbation is velocity deviation from v_star, computed by integrating acceleration.
    """
    ed = np.zeros(total_steps)
    v_delta = 0.0  # velocity deviation from v_star
    for k in range(total_steps):
        t = k * tstep
        if t < 2.0:
            a = -5.0
        elif t < 7.0:
            a = 0.0
        elif t < 12.0:
            a = 2.0
        else:
            a = 0.0
        v_delta += a * tstep
        # Clamp so head vehicle stays positive (v_star + ed > 0)
        v_delta = max(v_delta, -14.0)
        ed[k] = v_delta
    return ed


def _make_sinusoidal_perturbation(total_steps: int, tstep: float) -> np.ndarray:
    """Sinusoidal perturbation matching the DeeP-LCC reference (per_type=1).

    Amplitude 5 m/s, period 10s (same as reference: sine_amp=5, period=10/Tstep).
    """
    t = np.arange(total_steps) * tstep
    return 5.0 * np.sin(2.0 * np.pi / 10.0 * t)


def _make_nedc_perturbation(total_steps: int, tstep: float) -> np.ndarray:
    """NEDC (New European Driving Cycle) extra-urban profile.

    Compressed from 319s to fit simulation duration and scaled to
    perturbations around v_star. The NEDC profile (70-120 km/h) is
    mapped to [-3, +5] m/s perturbation range.

    Reference: soc-ucsd/DeeP-LCC nedc_trajectory.py
    """
    # Original NEDC breakpoints: (cumulative_time_s, velocity_kmh)
    nedc_breakpoints = [
        (0, 70), (50, 70),       # cruise 70
        (58, 50),                 # decel to 50
        (127, 50),               # cruise 50
        (140, 70),               # accel to 70
        (190, 70),               # cruise 70
        (225, 100),              # accel to 100
        (255, 100),              # cruise 100
        (275, 120),              # accel to 120
        (285, 120),              # cruise 120
        (299, 70),               # decel to 70
        (319, 70),               # cruise 70
    ]

    total_time = total_steps * tstep
    nedc_total = nedc_breakpoints[-1][0]

    # Interpolate NEDC velocity at each simulation step (compressed time)
    ed = np.zeros(total_steps)
    for k in range(total_steps):
        # Map simulation time to NEDC time
        t_nedc = (k * tstep / total_time) * nedc_total

        # Find surrounding breakpoints
        for i in range(len(nedc_breakpoints) - 1):
            t0, v0 = nedc_breakpoints[i]
            t1, v1 = nedc_breakpoints[i + 1]
            if t0 <= t_nedc <= t1:
                # Linear interpolation
                frac = (t_nedc - t0) / (t1 - t0) if t1 > t0 else 0.0
                vel_kmh = v0 + frac * (v1 - v0)
                break
        else:
            vel_kmh = nedc_breakpoints[-1][1]

        # Map 70-120 km/h → perturbation around v_star
        # 70 km/h → -3 m/s, 120 km/h → +5 m/s (linear mapping)
        ed[k] = (vel_kmh - 70.0) / (120.0 - 70.0) * 8.0 - 3.0

    return ed


def main():
    config = NNMPCConfig()

    if not Path(config.model_path).exists():
        print(f"Model not found at {config.model_path}. Run nnmpc_train.py first.")
        return

    eval_offline(config)
    eval_closed_loop(config)


if __name__ == "__main__":
    main()
