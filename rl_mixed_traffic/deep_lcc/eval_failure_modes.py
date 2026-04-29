"""Failure-mode evaluation for DeeP-LCC and its NN surrogate.

Runs a 2D grid of (HDV severity × measurement noise σ) and compares QP vs NN
performance. See docs/superpowers/specs/2026-04-17-deep-lcc-failure-modes-design.md.

Usage:
    uv run rl_mixed_traffic/deep_lcc/eval_failure_modes.py
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_mixed_traffic.deep_lcc.config import (
    DeepLCCConfig,
    OVMConfig,
    get_heterogeneous_ovm_config,
)
from rl_mixed_traffic.deep_lcc.eval_classical import (
    compute_metrics,
    plot_scenario,
    run_with_state,
)
from rl_mixed_traffic.deep_lcc.failure_scenarios import (
    SEVERITY_RANGES,
    make_ovm_resampler,
)
from rl_mixed_traffic.deep_lcc.nnmpc_config import NNMPCConfig
from rl_mixed_traffic.deep_lcc.nnmpc_eval import load_model, nn_predict


SEVERITIES: tuple[str, ...] = ("mild", "medium", "severe")
COMM_DELAYS_MS: tuple[float, ...] = (0.0, 100.0, 200.0, 300.0, 500.0, 1000.0)
SEEDS: tuple[int, ...] = (0, 1, 2)
CONTROLLERS: tuple[str, ...] = ("QP", "NN")
HEAD_PROFILES: tuple[str, ...] = ("flat", "sine")

SINE_AMPLITUDE: float = 2.0   # m/s
SINE_PERIOD: float = 10.0     # s (matches paper's per_type=1 scenario)

REPRESENTATIVE_SEVERITIES: tuple[str, ...] = ("mild", "medium", "severe")


def make_head_velocity(profile: str, config: DeepLCCConfig, total_time: float) -> np.ndarray:
    """Build the head-vehicle velocity trajectory for a given profile name."""
    total_steps = int(total_time / config.Tstep)
    if profile == "flat":
        return np.full(total_steps, config.v_star)
    if profile == "sine":
        t = np.arange(total_steps) * config.Tstep
        return config.v_star + SINE_AMPLITUDE * np.sin(2.0 * np.pi * t / SINE_PERIOD)
    raise ValueError(f"Unknown head profile: {profile!r}. Expected 'flat' or 'sine'.")


@dataclass
class CellResult:
    """One (controller, severity, delay_ms, head_profile, seed) run's output metrics."""

    controller: str
    severity: str
    comm_delay_ms: float
    head_profile: str
    seed: int
    total_cost: float
    msve_avg: float
    msve_cav0: float
    msve_cav1: float
    min_spacing: float
    max_spacing: float
    collision_count: int
    violation_count: int
    aeb_trigger_count: int
    failure_flag: bool
    elapsed_s: float


def _build_weights(
    config: DeepLCCConfig, ovm_config: OVMConfig
) -> tuple[np.ndarray, np.ndarray]:
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
    return Q, R


def run_one_cell(
    controller: str,
    severity: str,
    comm_delay_ms: float,
    seed: int,
    config: DeepLCCConfig,
    base_ovm: OVMConfig,
    Q: np.ndarray,
    R: np.ndarray,
    qp_controller_fn: Callable,
    nn_controller_fn: Callable,
    total_time: float = 100.0,
    return_trace: bool = False,
    head_profile: str = "flat",
) -> tuple[CellResult, dict | None]:
    """Run a single grid cell. Returns metrics and (optionally) velocity trace.

    severity == "static" disables both HDV parameter resampling AND
    the per-step HDV acceleration noise, giving a clean delay-only test.
    """
    head_vel = make_head_velocity(head_profile, config, total_time)

    if severity == "static":
        ovm_resampler = None
        acel_noise_override: float | None = 0.0
    else:
        ovm_resampler = make_ovm_resampler(severity, seed=seed)
        acel_noise_override = None
    ctrl_fn = qp_controller_fn if controller == "QP" else nn_controller_fn

    t0 = time.time()
    cost, vels, state, _ = run_with_state(
        config, base_ovm, Q, R, head_vel,
        controller_fn=ctrl_fn,
        seed=seed,
        noise_seed=seed + 1000,
        enable_aeb=True,
        update_s_star=False,
        ovm_resampler=ovm_resampler,
        comm_delay_ms=comm_delay_ms,
        acel_noise=acel_noise_override,
    )
    elapsed = time.time() - t0

    metrics = compute_metrics(vels, head_vel, config, full_state=state)
    result = CellResult(
        controller=controller,
        severity=severity,
        comm_delay_ms=comm_delay_ms,
        head_profile=head_profile,
        seed=seed,
        total_cost=cost,
        msve_avg=metrics["msve_avg"],
        msve_cav0=metrics["msve_cav0"],
        msve_cav1=metrics["msve_cav1"],
        min_spacing=metrics["min_spacing"],
        max_spacing=metrics["max_spacing"],
        collision_count=int(metrics["collision_count"]),
        violation_count=int(metrics["violation_count"]),
        aeb_trigger_count=int(metrics["aeb_trigger_count"]),
        failure_flag=bool(metrics["failure_flag"]),
        elapsed_s=elapsed,
    )
    trace = {"velocities": vels, "state": state} if return_trace else None
    return result, trace


def _build_qp_controller_fn(
    config: DeepLCCConfig, ovm_config: OVMConfig, Q: np.ndarray, R: np.ndarray
) -> Callable:
    """Build a QP controller_fn. Hankel matrices pre-collected ONCE on the
    static ovm_config so precollection is deterministic across severity cells."""
    from rl_mixed_traffic.deep_lcc.precollect import precollect
    from rl_mixed_traffic.deep_lcc.qp_solver import CachedDeepLCCSolver

    print("  Pre-collecting Hankel matrices (shared across grid)...")
    Up, Uf, Ep, Ef, Yp, Yf = precollect(config, ovm_config, seed=42)
    print("  Building QP solver...")
    solver = CachedDeepLCCSolver(
        Up, Yp, Uf, Yf, Ep, Ef, Q, R,
        config.lambda_g, config.lambda_y,
        u_limit=(config.dcel_max, config.acel_max),
        s_limit=(
            config.spacing_min - config.s_star,
            config.spacing_max - config.s_star,
        ),
    )
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    def qp_ctrl(uini, yini, eini):
        u_opt, _, status = solver(uini, yini, eini)
        if status in ("optimal", "optimal_inaccurate"):
            return u_opt[:m_ctr]
        return np.zeros(m_ctr)

    return qp_ctrl


def _build_nn_controller_fn(nnmpc_config: NNMPCConfig) -> Callable:
    """Build an NN controller_fn using the trained NNMPC checkpoint."""
    device = torch.device(nnmpc_config.device)
    model, input_mean, input_std = load_model(nnmpc_config.model_path, device)

    def nn_ctrl(uini, yini, eini):
        return nn_predict(model, uini, yini, eini, input_mean, input_std, device)

    return nn_ctrl


def run_grid(
    config: DeepLCCConfig,
    ovm_config: OVMConfig,
    nnmpc_config: NNMPCConfig,
    out_dir: Path,
    total_time: float = 100.0,
    severities: tuple[str, ...] = SEVERITIES,
    delays_ms: tuple[float, ...] = COMM_DELAYS_MS,
    seeds: tuple[int, ...] = SEEDS,
    head_profile: str = "flat",
) -> tuple[list[CellResult], dict[tuple[str, str, float, int], dict]]:
    """Run the 2D failure-mode grid (severity × delay) for QP and NN."""
    Q, R = _build_weights(config, ovm_config)
    qp_ctrl = _build_qp_controller_fn(config, ovm_config, Q, R)
    nn_ctrl = _build_nn_controller_fn(nnmpc_config)

    results: list[CellResult] = []
    traces: dict[tuple[str, str, float, int], dict] = {}

    total_cells = len(severities) * len(delays_ms) * len(seeds) * 2
    cell_idx = 0
    for controller in ("QP", "NN"):
        for severity in severities:
            for delay_ms in delays_ms:
                for seed in seeds:
                    cell_idx += 1
                    # Record traces for the first seed at a small set of
                    # representative delays, for any severity being swept.
                    record_trace = (
                        seed == seeds[0]
                        and delay_ms in (0.0, 200.0, 500.0)
                    )
                    print(
                        f"  [{cell_idx:3d}/{total_cells}] {controller} "
                        f"severity={severity} delay={delay_ms:.0f}ms "
                        f"head={head_profile} seed={seed}"
                    )
                    result, trace = run_one_cell(
                        controller, severity, delay_ms, seed,
                        config=config, base_ovm=ovm_config,
                        Q=Q, R=R,
                        qp_controller_fn=qp_ctrl, nn_controller_fn=nn_ctrl,
                        total_time=total_time,
                        return_trace=record_trace,
                        head_profile=head_profile,
                    )
                    results.append(result)
                    if trace is not None:
                        traces[(controller, severity, delay_ms, seed)] = trace

    return results, traces


def write_csv(results: list[CellResult], out_path: Path) -> None:
    """Write a flat CSV with one row per CellResult."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "controller", "severity", "comm_delay_ms", "head_profile", "seed",
        "total_cost", "msve_avg", "msve_cav0", "msve_cav1",
        "min_spacing", "max_spacing",
        "collision_count", "violation_count", "aeb_trigger_count",
        "failure_flag", "elapsed_s",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fields})
    print(f"  Results CSV written to {out_path}")


def _aggregate_by_severity(
    results: list[CellResult],
    metric: str,
    severities: tuple[str, ...],
    reducer=np.mean,
    delay_ms: float = 0.0,
) -> dict[str, np.ndarray]:
    """Aggregate `metric` across seeds at a fixed delay → {controller: (n_severities,) array}."""
    out: dict[str, np.ndarray] = {}
    for ctrl in ("QP", "NN"):
        arr = np.full(len(severities), np.nan)
        for i, sev in enumerate(severities):
            vals = [
                getattr(r, metric)
                for r in results
                if r.controller == ctrl
                and r.severity == sev
                and r.comm_delay_ms == delay_ms
            ]
            if vals:
                arr[i] = reducer(vals)
        out[ctrl] = arr
    return out


def plot_bar(
    results: list[CellResult],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    severities: tuple[str, ...] = SEVERITIES,
    delay_ms: float = 0.0,
    fmt: str = ".0f",
) -> None:
    """Grouped bar chart of `metric` by severity at a fixed delay, QP vs NN."""
    agg = _aggregate_by_severity(results, metric, severities, delay_ms=delay_ms)

    x = np.arange(len(severities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_qp = ax.bar(x - width / 2, agg["QP"], width, label="QP", color="#1f77b4")
    bars_nn = ax.bar(x + width / 2, agg["NN"], width, label="NN", color="#d62728")

    for bars, vals in ((bars_qp, agg["QP"]), (bars_nn, agg["NN"])):
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        format(v, fmt), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(severities)
    ax.set_xlabel("HDV severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved to {out_path}")


def plot_delay_sweep(
    results: list[CellResult],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    severities: tuple[str, ...] = SEVERITIES,
    delays_ms: tuple[float, ...] = COMM_DELAYS_MS,
) -> None:
    """Line plot of `metric` vs comm delay, one line per (controller, severity)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sev_colors = {"mild": "#2ca02c", "medium": "#ff7f0e", "severe": "#d62728"}
    ctrl_styles = {"QP": "-", "NN": "--"}
    delays = np.asarray(delays_ms)

    for ctrl in ("QP", "NN"):
        for sev in severities:
            means = np.full(len(delays_ms), np.nan)
            stds = np.full(len(delays_ms), np.nan)
            for i, d in enumerate(delays_ms):
                vals = [
                    getattr(r, metric) for r in results
                    if r.controller == ctrl
                    and r.severity == sev
                    and r.comm_delay_ms == d
                ]
                if vals:
                    means[i] = float(np.mean(vals))
                    stds[i] = float(np.std(vals))
            ax.plot(
                delays, means,
                color=sev_colors.get(sev, "black"),
                linestyle=ctrl_styles[ctrl],
                marker="o" if ctrl == "QP" else "s",
                label=f"{ctrl} · {sev}",
            )
            ax.fill_between(
                delays, means - stds, means + stds,
                color=sev_colors.get(sev, "black"),
                alpha=0.12,
            )

    ax.set_xlabel("V2V communication delay (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delay sweep plot saved to {out_path}")


def plot_timeseries(
    trace: dict,
    config: DeepLCCConfig,
    controller: str,
    severity: str,
    out_path: Path,
    delay_ms: float = 0.0,
) -> None:
    """Plot velocities/accels/spacing for one representative cell."""
    velocities = trace["velocities"]
    state = trace["state"]
    n_steps = len(velocities["head"])
    t = np.arange(n_steps) * config.Tstep

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax = axes[0]
    ax.plot(t, velocities["head"], color="gray", linestyle="--", label="Head")
    ax.plot(t, velocities["cav_0"], color="#1f77b4", label="CAV 3")
    ax.plot(t, velocities["cav_1"], color="#d62728", label="CAV 6")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"{controller} · severity={severity} · delay={delay_ms:.0f}ms (seed 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, state[:, 3, 2], color="#1f77b4", label="CAV 3")
    ax.plot(t, state[:, 6, 2], color="#d62728", label="CAV 6")
    ax.axhline(config.acel_max, color="black", linestyle=":", alpha=0.4)
    ax.axhline(config.dcel_max, color="black", linestyle=":", alpha=0.4)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    s_cav0 = state[:, 2, 0] - state[:, 3, 0]
    s_cav1 = state[:, 5, 0] - state[:, 6, 0]
    ax.plot(t, s_cav0, color="#1f77b4", label="CAV 3")
    ax.plot(t, s_cav1, color="#d62728", label="CAV 6")
    ax.axhline(config.spacing_min, color="red", linestyle=":", alpha=0.4, label="min")
    ax.axhline(config.spacing_max, color="black", linestyle=":", alpha=0.4, label="max")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spacing (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Timeseries saved to {out_path}")


def run_and_plot(
    config: DeepLCCConfig,
    ovm_config: OVMConfig,
    nnmpc_config: NNMPCConfig,
    out_dir: Path,
    head_profile: str,
    severities: tuple[str, ...] = SEVERITIES,
    seeds: tuple[int, ...] = SEEDS,
) -> None:
    """Run the full grid for one head profile and produce all plots/CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DeeP-LCC Failure-Mode Evaluation — head_profile={head_profile}")
    print("=" * 60)
    print(f"  Severities: {severities}")
    print(f"  Delays ms:  {COMM_DELAYS_MS}")
    print(f"  Seeds:      {seeds}")
    print(f"  Head:       {head_profile}"
          + (f" (amp={SINE_AMPLITUDE}, period={SINE_PERIOD}s)" if head_profile == "sine" else ""))
    print(f"  Output dir: {out_dir}")
    print()

    t0 = time.time()
    results, traces = run_grid(
        config, ovm_config, nnmpc_config,
        out_dir=out_dir,
        total_time=100.0,
        severities=severities,
        seeds=seeds,
        head_profile=head_profile,
    )
    print(f"\nGrid complete in {time.time() - t0:.1f}s")

    write_csv(results, out_dir / "results.csv")

    # 1D bar charts at delay=0
    plot_bar(
        results, "total_cost",
        f"Total Cost at delay=0ms — head={head_profile}",
        "Total cost", out_dir / "bar_cost.png", fmt=".0f", delay_ms=0.0,
    )
    plot_bar(
        results, "aeb_trigger_count",
        f"AEB Triggers at delay=0ms — head={head_profile}",
        "AEB triggers", out_dir / "bar_aeb.png", fmt=".1f", delay_ms=0.0,
    )
    plot_bar(
        results, "msve_avg",
        f"MSVE at delay=0ms — head={head_profile}",
        "MSVE", out_dir / "bar_msve.png", fmt=".3f", delay_ms=0.0,
    )

    # Delay-sweep line plots
    plot_delay_sweep(
        results, "total_cost",
        f"Total Cost vs V2V delay — head={head_profile}",
        "Total cost",
        out_dir / "delay_sweep_cost.png",
        severities=severities,
    )
    plot_delay_sweep(
        results, "msve_avg",
        f"MSVE vs V2V delay — head={head_profile}",
        "MSVE",
        out_dir / "delay_sweep_msve.png",
        severities=severities,
    )
    plot_delay_sweep(
        results, "aeb_trigger_count",
        f"AEB Triggers vs V2V delay — head={head_profile}",
        "AEB triggers",
        out_dir / "delay_sweep_aeb.png",
        severities=severities,
    )
    plot_delay_sweep(
        results, "min_spacing",
        f"Min CAV Spacing vs V2V delay — head={head_profile}",
        "Min spacing (m)",
        out_dir / "delay_sweep_min_spacing.png",
        severities=severities,
    )

    # Timeseries for representative (controller × severity × delay) cells at seeds[0]
    ts_dir = out_dir / "timeseries"
    for ctrl in ("QP", "NN"):
        for sev in severities:
            for d in (0.0, 200.0, 500.0):
                key = (ctrl, sev, d, seeds[0])
                if key not in traces:
                    continue
                filename = f"{sev}_delay{int(d):04d}ms_{ctrl.lower()}.png"
                plot_timeseries(
                    traces[key], config, ctrl, sev,
                    ts_dir / filename, delay_ms=d,
                )

    print()
    print("=" * 60)
    print(f"Evaluation complete ({head_profile}). Artifacts in {out_dir}")
    print("=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DeeP-LCC failure-mode evaluation (V2V delay sweep)."
    )
    parser.add_argument(
        "--head", choices=("flat", "sine", "both"), default="flat",
        help="Head-vehicle velocity profile: 'flat' (constant v_star), "
             "'sine' (±2 m/s, 10s period), or 'both' (runs flat then sine).",
    )
    parser.add_argument(
        "--severity",
        choices=("static", "mild", "medium", "severe", "all"),
        default="static",
        help="HDV configuration: 'static' (no resampling, no HDV accel noise "
             "— cleanest delay-only test; default), 'mild'/'medium'/'severe' "
             "(resampled HDVs at that intensity), or 'all' (2D grid over "
             "mild/medium/severe).",
    )
    args = parser.parse_args()

    severities: tuple[str, ...] = (
        SEVERITIES if args.severity == "all" else (args.severity,)
    )
    # static mode is deterministic (no HDV resampling, no process noise) — 1 seed is enough.
    seeds: tuple[int, ...] = (0,) if args.severity == "static" else SEEDS

    config = DeepLCCConfig()
    ovm_config = get_heterogeneous_ovm_config()
    nnmpc_config = NNMPCConfig()

    base_dir = Path("deep_lcc_results/failure_modes")
    profiles = ("flat", "sine") if args.head == "both" else (args.head,)
    for profile in profiles:
        out_dir = base_dir / profile
        run_and_plot(
            config, ovm_config, nnmpc_config, out_dir, profile,
            severities=severities, seeds=seeds,
        )


if __name__ == "__main__":
    main()
