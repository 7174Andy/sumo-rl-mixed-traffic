"""Shared helpers for the IDM evaluation studies.

Used by eval_idm_baseline.py (delay-less QP-vs-NN comparison) and
eval_idm_delay_sweep.py (delay sweep). Holds QP/NN controller wiring,
per-cell runner, CSV writer, and plotting dispatchers.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_mixed_traffic.deep_lcc.config import (
    DeepLCCConfig,
    get_heterogeneous_ovm_config,
)
from rl_mixed_traffic.deep_lcc.eval_classical import compute_metrics, run_with_state
from rl_mixed_traffic.deep_lcc.eval_failure_modes import (
    make_head_velocity,
    plot_delay_sweep,
)
from rl_mixed_traffic.deep_lcc.idm import (
    get_default_idm_config,
    idm_dynamics,
)
from rl_mixed_traffic.deep_lcc.nnmpc_config import NNMPCConfig
from rl_mixed_traffic.deep_lcc.nnmpc_eval import load_model, nn_predict
from rl_mixed_traffic.deep_lcc.precollect import precollect
from rl_mixed_traffic.deep_lcc.qp_solver import CachedDeepLCCSolver


HEAD_PROFILES: tuple[str, ...] = ("flat", "sine")
DELAYS_MS: tuple[float, ...] = (0.0, 100.0, 200.0, 300.0, 500.0, 1000.0)
SEEDS: tuple[int, ...] = (0, 1, 2)
CONTROLLERS: tuple[str, ...] = ("QP", "NN")


@dataclass
class CellResult:
    head_profile: str
    controller: str
    comm_delay_ms: float
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


def build_weights(config: DeepLCCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Construct (Q, R) for the 8-vehicle, 2-CAV platoon used by the spec."""
    n_vehicle = 8
    m_ctr = 2
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


def build_controllers(
    config: DeepLCCConfig,
    Q: np.ndarray,
    R: np.ndarray,
    nnmpc_config: NNMPCConfig,
) -> tuple[Callable, Callable, object]:
    """Pre-collect Hankel matrices on OVM and build (qp_ctrl, nn_ctrl, eval_hdv_config).

    Hankel matrices and NNMPC checkpoint are both OVM-trained (existing artifacts).
    The returned eval_hdv_config is IDM — evaluation swaps HDV dynamics to IDM.
    """
    eval_hdv_config = get_default_idm_config()
    hankel_hdv_config = get_heterogeneous_ovm_config()

    print("  Pre-collecting Hankel matrices (OVM)...")
    Up, Uf, Ep, Ef, Yp, Yf = precollect(config, hankel_hdv_config, seed=42)
    solver = CachedDeepLCCSolver(
        Up, Yp, Uf, Yf, Ep, Ef, Q, R,
        config.lambda_g, config.lambda_y,
        u_limit=(config.dcel_max, config.acel_max),
        s_limit=(
            config.spacing_min - config.s_star,
            config.spacing_max - config.s_star,
        ),
    )
    pos_cav = np.where(np.array(eval_hdv_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    def qp_ctrl(uini, yini, eini):
        u_opt, _, status = solver(uini, yini, eini)
        if status in ("optimal", "optimal_inaccurate"):
            return u_opt[:m_ctr]
        return np.zeros(m_ctr)

    print(f"  Loading NNMPC from {nnmpc_config.model_path} ...")
    device = torch.device(nnmpc_config.device)
    model, input_mean, input_std = load_model(nnmpc_config.model_path, device)

    def nn_ctrl(uini, yini, eini):
        return nn_predict(model, uini, yini, eini, input_mean, input_std, device)

    return qp_ctrl, nn_ctrl, eval_hdv_config


def run_one_cell(
    controller: str,
    head_profile: str,
    comm_delay_ms: float,
    seed: int,
    config: DeepLCCConfig,
    Q: np.ndarray,
    R: np.ndarray,
    qp_ctrl: Callable,
    nn_ctrl: Callable,
    eval_hdv_config,
    total_time: float = 100.0,
    return_trace: bool = False,
) -> tuple[CellResult, dict | None]:
    head_vel = make_head_velocity(head_profile, config, total_time)
    ctrl_fn = qp_ctrl if controller == "QP" else nn_ctrl

    t0 = time.time()
    cost, vels, state, _ = run_with_state(
        config, eval_hdv_config, Q, R, head_vel,
        controller_fn=ctrl_fn,
        seed=seed,
        noise_seed=seed + 1000,
        enable_aeb=True,
        update_s_star=False,
        comm_delay_ms=comm_delay_ms,
        acel_noise=0.0,
        hdv_dynamics_fn=idm_dynamics,
    )
    elapsed = time.time() - t0

    metrics = compute_metrics(vels, head_vel, config, full_state=state)
    result = CellResult(
        head_profile=head_profile,
        controller=controller,
        comm_delay_ms=comm_delay_ms,
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


def write_csv(results: list[CellResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "head_profile", "controller", "comm_delay_ms", "seed",
        "total_cost", "msve_avg", "msve_cav0", "msve_cav1",
        "min_spacing", "max_spacing",
        "collision_count", "violation_count", "aeb_trigger_count",
        "failure_flag", "elapsed_s",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"  CSV written to {out_path}")


def _with_severity(result: CellResult, severity: str) -> SimpleNamespace:
    """Adapter so plot_delay_sweep (which reads r.severity) can consume
    our CellResult objects that have no severity field."""
    return SimpleNamespace(**result.__dict__, severity=severity)


def plot_delay_sweep_wrapper(
    results: list[CellResult],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    delays_ms: tuple[float, ...],
) -> None:
    # plot_delay_sweep from eval_failure_modes is typed to its own CellResult
    # dataclass. SimpleNamespace duck-types via attribute access, so cast.
    fake = cast(list, [_with_severity(r, "idm") for r in results])
    plot_delay_sweep(
        fake, metric, title, ylabel, out_path,
        severities=("idm",), delays_ms=delays_ms,
    )


def plot_baseline_bars(
    results: list[CellResult],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    fmt: str = ".2f",
) -> None:
    """Single bar per controller (QP, NN) at delay=0, mean over seeds."""
    means: dict[str, float] = {}
    for c in CONTROLLERS:
        vals = [
            getattr(r, metric) for r in results
            if r.controller == c and r.comm_delay_ms == 0.0
        ]
        means[c] = float(np.mean(vals)) if vals else float("nan")

    x = np.arange(len(CONTROLLERS))
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"QP": "#1f77b4", "NN": "#d62728"}
    bars = ax.bar(x, [means[c] for c in CONTROLLERS], width=0.5,
                  color=[colors[c] for c in CONTROLLERS])
    for bar, c in zip(bars, CONTROLLERS):
        v = means[c]
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    format(v, fmt), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(list(CONTROLLERS))
    ax.set_xlabel("Controller")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved to {out_path}")
