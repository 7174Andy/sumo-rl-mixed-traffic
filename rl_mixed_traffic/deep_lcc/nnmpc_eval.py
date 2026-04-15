"""Evaluate NNMPC against the DeeP-LCC QP solver.

Two modes:
  1. Offline accuracy on held-out test data
  2. Closed-loop simulation comparison (NN vs QP)

Usage:
    uv run rl_mixed_traffic/deep_lcc/nnmpc_eval.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_mixed_traffic.deep_lcc.config import (
    DeepLCCConfig,
    OVMConfig,
    get_heterogeneous_ovm_config,
)
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
    head_vel_abs: np.ndarray,
) -> tuple[float, int, dict[str, np.ndarray]]:
    """Run one closed-loop episode with a given controller.

    Matches the reference implementation (soc-ucsd/DeeP-LCC) with:
    - Dynamic equilibrium update (v_star, s_star) each step
    - Re-computation of yini with updated equilibrium
    - Full state history for past measurement recomputation
    - AEB override when HDV dynamics commands emergency braking

    Args:
        controller_fn: Callable(uini, yini, eini) → u_cav (m_ctr,)
        head_vel_abs: Absolute head vehicle velocity at each step (m/s).

    Returns:
        (total_cost, n_steps, velocities)
    """
    n_vehicle = ovm_config.n_vehicle
    ID = ovm_config.ID
    pos_cav = np.where(np.array(ID) == 1)[0]
    m_ctr = len(pos_cav)
    p_ctr = n_vehicle + m_ctr  # measure_type=3

    total_steps = len(head_vel_abs)
    T_ini = config.T_ini

    v_star = config.v_star
    s_star = config.s_star

    # State: S[k] = (n_vehicle+1, 3) with [pos, vel, accel]
    S = np.zeros((total_steps, n_vehicle + 1, 3))
    S[0, 0, 0] = 0.0
    for i in range(1, n_vehicle + 1):
        S[0, i, 0] = S[0, i - 1, 0] - s_star
    S[0, :, 1] = v_star

    # Rolling buffers (matching dataset generator structure)
    u_buf = np.zeros((m_ctr, T_ini))
    y_buf = np.zeros((p_ctr, T_ini))
    e_buf = np.zeros(T_ini)

    total_cost = 0.0
    n_control_steps = 0

    vel_head = np.zeros(total_steps)
    vel_cavs = np.zeros((m_ctr, total_steps))

    for k in range(total_steps - 1):
        # Measure current output (fixed equilibrium, matching dataset generation)
        y_k = measure_mixed_traffic(
            S[k, 1:, 1], S[k, :, 0], ID, v_star, s_star, config.measure_type
        )
        e_k = S[k, 0, 1] - v_star

        # Update rolling buffers
        u_buf = np.roll(u_buf, -1, axis=1)
        y_buf = np.roll(y_buf, -1, axis=1)
        e_buf = np.roll(e_buf, -1)
        y_buf[:, -1] = y_k
        e_buf[-1] = e_k

        if k >= T_ini:
            uini = u_buf.flatten(order="F")
            yini = y_buf.flatten(order="F")
            eini = e_buf.copy()

            cav_accel = controller_fn(uini, yini, eini)

            # Accumulate cost
            total_cost += float(
                y_k @ Q @ y_k + cav_accel @ R[:m_ctr, :m_ctr] @ cav_accel
            )
            n_control_steps += 1
        else:
            cav_accel = np.zeros(m_ctr)

        # HDV dynamics + noise
        acel = hdv_dynamics(S[k], ovm_config)
        noise = -config.acel_noise + 2.0 * config.acel_noise * np.random.rand(
            n_vehicle
        )
        acel = np.clip(acel + noise, ovm_config.dcel_max, ovm_config.acel_max)

        S[k, 0, 2] = 0.0
        S[k, 1:, 2] = acel
        for j, cav_idx in enumerate(pos_cav):
            S[k, cav_idx + 1, 2] = float(cav_accel[j])

        u_buf[:, -1] = cav_accel

        # Integrate dynamics
        S[k + 1, :, 1] = S[k, :, 1] + config.Tstep * S[k, :, 2]
        S[k + 1, 0, 1] = head_vel_abs[k + 1]
        S[k + 1, :, 0] = S[k, :, 0] + config.Tstep * S[k, :, 1]

        vel_head[k] = S[k, 0, 1]
        for j, cav_idx in enumerate(pos_cav):
            vel_cavs[j, k] = S[k, cav_idx + 1, 1]

    # Final step
    k_end = total_steps - 1
    vel_head[k_end] = S[k_end, 0, 1]
    for j, cav_idx in enumerate(pos_cav):
        vel_cavs[j, k_end] = S[k_end, cav_idx + 1, 1]

    velocities = {"head": vel_head}
    for j in range(m_ctr):
        velocities[f"cav_{j}"] = vel_cavs[j]

    return total_cost, n_control_steps, velocities


def eval_closed_loop(nnmpc_config: NNMPCConfig) -> None:
    """Compare NN vs QP in closed-loop on test scenarios."""
    device = torch.device(nnmpc_config.device)
    model, input_mean, input_std = load_model(nnmpc_config.model_path, device)

    config = DeepLCCConfig()
    ovm_config = get_heterogeneous_ovm_config()

    n_vehicle = ovm_config.n_vehicle
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    Q_v = config.weight_v * np.eye(n_vehicle)
    Q_s = config.weight_s * np.eye(m_ctr)
    Q = np.block(
        [
            [Q_v, np.zeros((n_vehicle, m_ctr))],
            [np.zeros((m_ctr, n_vehicle)), Q_s],
        ]
    )
    R = config.weight_u * np.eye(m_ctr)

    # Use 40s for eval scenarios (not the full 100s training duration)
    eval_time = 40.0
    total_steps = int(eval_time / config.Tstep)

    # Test scenarios: all return absolute head vehicle velocity arrays
    v_s = config.v_star
    scenarios = {
        "random_±1": lambda rng: v_s + (-1.0 + 2.0 * rng.random(total_steps)),
        "random_±5": lambda rng: v_s + (-5.0 + 10.0 * rng.random(total_steps)),
        "brake": lambda _: v_s + _make_brake_perturbation(total_steps, config.Tstep),
        "sinusoidal": lambda _: v_s + _make_sinusoidal_perturbation(
            total_steps, config.Tstep
        ),
    }

    print("\n=== Closed-Loop Evaluation ===")
    print(f"{'Scenario':<15} {'QP Cost':>10} {'NN Cost':>10} {'Diff %':>8}")
    print("-" * 48)

    for name, ed_fn in scenarios.items():
        rng = np.random.default_rng(123)
        head_vel_abs = ed_fn(rng)

        # QP controller
        Up, Uf, Ep, Ef, Yp, Yf = precollect(config, ovm_config, seed=999)
        solver = CachedDeepLCCSolver(
            Up,
            Yp,
            Uf,
            Yf,
            Ep,
            Ef,
            Q,
            R,
            config.lambda_g,
            config.lambda_y,
            u_limit=(config.dcel_max, config.acel_max),
            s_limit=(
                config.spacing_min - config.s_star,
                config.spacing_max - config.s_star,
            ),
        )

        def qp_controller(uini, yini, eini):
            u_opt, _, status = solver(uini, yini, eini)
            if status in ("optimal", "optimal_inaccurate"):
                return u_opt[:m_ctr]
            return np.zeros(m_ctr)

        qp_cost, _, qp_vel = run_closed_loop(
            qp_controller, config, ovm_config, Q, R, head_vel_abs
        )

        # NN controller
        def nn_controller(uini, yini, eini):
            return nn_predict(model, uini, yini, eini, input_mean, input_std, device)

        nn_cost, _, nn_vel = run_closed_loop(
            nn_controller, config, ovm_config, Q, R, head_vel_abs
        )

        diff_pct = (nn_cost - qp_cost) / max(abs(qp_cost), 1e-8) * 100
        print(f"{name:<15} {qp_cost:10.2f} {nn_cost:10.2f} {diff_pct:+7.2f}%")

        if name in ("NEDC", "brake", "sinusoidal"):
            plot_scenario_velocities(name, qp_vel, nn_vel, config)


def plot_scenario_velocities(
    scenario_name: str,
    qp_vel: dict[str, np.ndarray],
    nn_vel: dict[str, np.ndarray],
    config: DeepLCCConfig,
) -> None:
    """Plot CAV velocities for QP vs NN on a given scenario."""
    total_steps = len(qp_vel["head"])
    t = np.arange(total_steps) * config.Tstep

    fig, ax = plt.subplots(figsize=(10, 5))

    # Head vehicle (reference)
    ax.plot(t, qp_vel["head"], color="gray", linewidth=1.5, linestyle="--",
            label="Head vehicle", alpha=0.7)

    # CAV 3 (pos_cav index 0)
    ax.plot(t, qp_vel["cav_0"], color="#1f77b4", linewidth=1.5,
            label="CAV 3 (QP)")
    ax.plot(t, nn_vel["cav_0"], color="#1f77b4", linewidth=1.5, linestyle=":",
            label="CAV 3 (NN)")

    # CAV 6 (pos_cav index 1)
    ax.plot(t, qp_vel["cav_1"], color="#d62728", linewidth=1.5,
            label="CAV 6 (QP)")
    ax.plot(t, nn_vel["cav_1"], color="#d62728", linewidth=1.5, linestyle=":",
            label="CAV 6 (NN)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"{scenario_name} Scenario: CAV Velocities — QP vs NN")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    filename = scenario_name.lower().replace("±", "").replace(" ", "_")
    out_path = f"deep_lcc_results/{filename}_cav_velocities.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{scenario_name} velocity plot saved to {out_path}")


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
    """Extract NEDC head vehicle trajectory from reference simulation data.

    Loads the head vehicle velocity from the reference .mat file produced
    by soc-ucsd/DeeP-LCC. Returns absolute velocity (m/s), not perturbation.

    Falls back to a synthetic EUDC-like profile if .mat file is not found.
    """
    mat_path = Path("deep_lcc_dataset/nedc_reference.mat")
    if not mat_path.exists():
        # Try the download location
        mat_path = Path.home() / "Downloads" / (
            "simulation_data2_1_modified_v1_noiseLevel_0.1"
            "_hdvType_1_lambdaG_100_lambdaY_10000.mat"
        )

    if mat_path.exists():
        from scipy.io import loadmat

        data = loadmat(str(mat_path))
        S_ref = data["S"]  # (total_steps, n_vehicle+1, 3)
        head_vel = S_ref[:, 0, 1].ravel()
        # Truncate or pad to match requested total_steps
        if len(head_vel) >= total_steps:
            return head_vel[:total_steps]
        # Pad with last value
        return np.concatenate(
            [head_vel, np.full(total_steps - len(head_vel), head_vel[-1])]
        )

    # Fallback: synthetic EUDC-like profile (highway portion of NEDC)
    # Breakpoints: (time_s, velocity_m/s) — based on EUDC at real timescale
    breakpoints = [
        (0, 15.0), (10, 19.44), (50, 19.44),  # cruise 70 km/h
        (58, 13.89), (127, 13.89),              # decel to 50, cruise
        (140, 19.44), (190, 19.44),             # accel to 70, cruise
        (225, 27.78), (255, 27.78),             # accel to 100, cruise
        (275, 33.33), (285, 33.33),             # accel to 120, cruise
        (299, 19.44), (319, 19.44),             # decel to 70, cruise
    ]

    bp_times = np.array([t for t, _ in breakpoints])
    bp_vals = np.array([v for _, v in breakpoints])

    head_vel = np.full(total_steps, bp_vals[-1])
    for k in range(total_steps):
        t = k * tstep
        if t <= bp_times[0]:
            head_vel[k] = bp_vals[0]
        elif t < bp_times[-1]:
            idx = np.searchsorted(bp_times, t, side="right") - 1
            t0, v0 = bp_times[idx], bp_vals[idx]
            t1, v1 = bp_times[idx + 1], bp_vals[idx + 1]
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            head_vel[k] = v0 + frac * (v1 - v0)

    return head_vel


def main():
    config = NNMPCConfig()

    if not Path(config.model_path).exists():
        print(f"Model not found at {config.model_path}. Run nnmpc_train.py first.")
        return

    eval_offline(config)
    eval_closed_loop(config)


if __name__ == "__main__":
    main()
