"""IDM × V2V delay-sweep evaluation for DeeP-LCC.

Sweeps the full delay grid under pure IDM HDVs, across flat and sine
head profiles. Both controllers are the existing OVM-trained artifacts.

Usage:
    uv run rl_mixed_traffic/deep_lcc/eval_idm_delay_sweep.py --head both
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rl_mixed_traffic.deep_lcc.config import DeepLCCConfig
from rl_mixed_traffic.deep_lcc.eval_failure_modes import plot_timeseries
from rl_mixed_traffic.deep_lcc.eval_idm_core import (
    CONTROLLERS,
    DELAYS_MS,
    HEAD_PROFILES,
    SEEDS,
    CellResult,
    build_controllers,
    build_weights,
    plot_delay_sweep_wrapper,
    run_one_cell,
    write_csv,
)
from rl_mixed_traffic.deep_lcc.nnmpc_config import NNMPCConfig


def run_head_profile(
    head_profile: str,
    config: DeepLCCConfig,
    Q: np.ndarray,
    R: np.ndarray,
    qp_ctrl,
    nn_ctrl,
    eval_hdv_config,
    out_dir: Path,
    delays_ms: tuple[float, ...],
    seeds: tuple[int, ...],
    total_time: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"IDM Delay Sweep — head={head_profile}")
    print("=" * 60)

    results: list[CellResult] = []
    traces: dict[tuple[str, float, int], dict] = {}
    total = len(CONTROLLERS) * len(delays_ms) * len(seeds)
    idx = 0
    t0 = time.time()
    for ctrl in CONTROLLERS:
        for d in delays_ms:
            for seed in seeds:
                idx += 1
                record = seed == seeds[0] and d in (0.0, 200.0, 500.0)
                print(f"  [{idx:3d}/{total}] {ctrl} delay={d:.0f}ms seed={seed}")
                result, trace = run_one_cell(
                    ctrl, head_profile, d, seed,
                    config, Q, R, qp_ctrl, nn_ctrl, eval_hdv_config,
                    total_time=total_time, return_trace=record,
                )
                results.append(result)
                if trace is not None:
                    traces[(ctrl, d, seed)] = trace
    print(f"  Done in {time.time() - t0:.1f}s")

    write_csv(results, out_dir / "results.csv")

    plot_delay_sweep_wrapper(
        results, "total_cost",
        f"Total Cost vs V2V delay — head={head_profile}",
        "Total cost", out_dir / "delay_sweep_cost.png", delays_ms,
    )
    plot_delay_sweep_wrapper(
        results, "msve_avg",
        f"MSVE vs V2V delay — head={head_profile}",
        "MSVE", out_dir / "delay_sweep_msve.png", delays_ms,
    )
    plot_delay_sweep_wrapper(
        results, "aeb_trigger_count",
        f"AEB Triggers vs V2V delay — head={head_profile}",
        "AEB triggers", out_dir / "delay_sweep_aeb.png", delays_ms,
    )
    plot_delay_sweep_wrapper(
        results, "min_spacing",
        f"Min CAV Spacing vs V2V delay — head={head_profile}",
        "Min spacing (m)", out_dir / "delay_sweep_min_spacing.png", delays_ms,
    )

    ts_dir = out_dir / "timeseries"
    for ctrl in CONTROLLERS:
        for d in (0.0, 200.0, 500.0):
            key = (ctrl, d, seeds[0])
            if key not in traces:
                continue
            plot_timeseries(
                traces[key], config, ctrl, "idm",
                ts_dir / f"delay{int(d):04d}ms_{ctrl.lower()}.png",
                delay_ms=d,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeeP-LCC IDM × V2V delay-sweep evaluation."
    )
    parser.add_argument("--head", choices=("flat", "sine", "both"), default="both")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny smoke version (delays=(0,500), seed=0, total_time=10s).",
    )
    args = parser.parse_args()

    config = DeepLCCConfig()
    Q, R = build_weights(config)

    nnmpc_cfg = NNMPCConfig()  # existing OVM-trained checkpoint
    qp_ctrl, nn_ctrl, eval_hdv_config = build_controllers(
        config, Q, R, nnmpc_cfg,
    )

    if args.smoke:
        delays_ms = (0.0, 500.0)
        seeds = (0,)
        total_time = 10.0
    else:
        delays_ms = DELAYS_MS
        seeds = SEEDS
        total_time = 100.0

    heads = HEAD_PROFILES if args.head == "both" else (args.head,)
    base_dir = Path("deep_lcc_results/idm_delay_sweep")

    for head in heads:
        run_head_profile(
            head, config, Q, R, qp_ctrl, nn_ctrl, eval_hdv_config,
            base_dir / head,
            delays_ms=delays_ms, seeds=seeds, total_time=total_time,
        )


if __name__ == "__main__":
    main()
