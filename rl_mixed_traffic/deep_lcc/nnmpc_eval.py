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
    get_heterogeneous_ovm_config,
)
from rl_mixed_traffic.deep_lcc.eval_classical import (
    compute_metrics,
    make_aggressive_sine,
    make_extreme_brake,
    make_nedc,
    make_sinusoidal,
    make_stop_and_go,
    make_varying_sine,
    plot_scenario,
    run_with_state,
)
from rl_mixed_traffic.deep_lcc.nnmpc_config import NNMPCConfig
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork


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


def eval_closed_loop(nnmpc_config: NNMPCConfig) -> None:
    """Compare NN vs QP in closed-loop using the same simulation as eval_classical."""
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

    eval_steps = int(30.0 / config.Tstep)
    sine_steps = int(40.0 / config.Tstep)

    scenarios = {
        "brake": make_extreme_brake(eval_steps, config.Tstep, config.v_star),
        "sinusoidal": make_sinusoidal(
            sine_steps, config.Tstep, config.v_star, amplitude=2.0
        ),
        "NEDC": make_nedc(config.Tstep),
        "varying_sine": make_varying_sine(200.0, config.Tstep, config.v_star),
        "aggressive_sine": make_aggressive_sine(200.0, config.Tstep, config.v_star),
        "stop_and_go": make_stop_and_go(200.0, config.Tstep, config.v_star),
    }

    print("\n=== Closed-Loop Evaluation (QP vs NN) ===")
    header = (
        f"{'Scenario':<15} {'QP Cost':>10} {'NN Cost':>10} {'Cost Δ%':>8} "
        f"{'QP MSVE avg':>12} {'NN MSVE avg':>12} {'MSVE Δ%':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, head_vel in scenarios.items():
        # QP controller (default when controller_fn=None)
        print(f"\n--- {name} (QP) ---")
        qp_cost, qp_vel, _, _ = run_with_state(
            config, ovm_config, Q, R, head_vel,
            enable_aeb=False, update_s_star=False,
        )

        # NN controller
        def nn_controller(uini, yini, eini):
            return nn_predict(model, uini, yini, eini, input_mean, input_std, device)

        print(f"--- {name} (NN) ---")
        nn_cost, nn_vel, _, _ = run_with_state(
            config, ovm_config, Q, R, head_vel,
            controller_fn=nn_controller,
            enable_aeb=False, update_s_star=False,
        )

        qp_metrics = compute_metrics(qp_vel, head_vel, config)
        nn_metrics = compute_metrics(nn_vel, head_vel, config)

        cost_diff_pct = (nn_cost - qp_cost) / max(abs(qp_cost), 1e-8) * 100
        msve_diff_pct = (
            (nn_metrics["msve_avg"] - qp_metrics["msve_avg"])
            / max(abs(qp_metrics["msve_avg"]), 1e-8) * 100
        )

        print(
            f"\n  QP  MSVE: cav0={qp_metrics['msve_cav0']:.4f}  "
            f"cav1={qp_metrics['msve_cav1']:.4f}  avg={qp_metrics['msve_avg']:.4f}"
        )
        print(
            f"  NN  MSVE: cav0={nn_metrics['msve_cav0']:.4f}  "
            f"cav1={nn_metrics['msve_cav1']:.4f}  avg={nn_metrics['msve_avg']:.4f}"
        )
        rows.append(
            f"{name:<15} {qp_cost:10.2f} {nn_cost:10.2f} {cost_diff_pct:+7.2f}% "
            f"{qp_metrics['msve_avg']:12.4f} {nn_metrics['msve_avg']:12.4f} "
            f"{msve_diff_pct:+7.2f}%"
        )

        # Plot comparison
        _plot_qp_vs_nn(name, qp_vel, nn_vel, config)

    print("\n=== Summary ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(row)


def _plot_qp_vs_nn(
    scenario_name: str,
    qp_vel: dict[str, np.ndarray],
    nn_vel: dict[str, np.ndarray],
    config: DeepLCCConfig,
) -> None:
    """Plot CAV velocities for QP vs NN on a given scenario."""
    total_steps = len(qp_vel["head"])
    t = np.arange(total_steps) * config.Tstep

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, qp_vel["head"], color="gray", linewidth=1.5, linestyle="--",
            label="Head vehicle", alpha=0.7)

    ax.plot(t, qp_vel["cav_0"], color="#1f77b4", linewidth=1.5,
            label="CAV 3 (QP)")
    ax.plot(t, nn_vel["cav_0"], color="#1f77b4", linewidth=1.5, linestyle=":",
            label="CAV 3 (NN)")

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
    print(f"{scenario_name} velocity plot saved to {out_path}")


def main():
    config = NNMPCConfig()

    if not Path(config.model_path).exists():
        print(f"Model not found at {config.model_path}. Run nnmpc_train.py first.")
        return

    eval_offline(config)
    eval_closed_loop(config)


if __name__ == "__main__":
    main()
