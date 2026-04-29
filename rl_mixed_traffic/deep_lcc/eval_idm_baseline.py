"""IDM baseline evaluation (no V2V delay).

Compares QP vs NN under pure IDM HDVs at delay=0, across flat and sine
head profiles. Both controllers are the existing OVM-trained artifacts.

Usage:
    uv run rl_mixed_traffic/deep_lcc/eval_idm_baseline.py --head both
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
    HEAD_PROFILES,
    SEEDS,
    CellResult,
    build_controllers,
    build_weights,
    plot_baseline_bars,
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
    seeds: tuple[int, ...],
    total_time: float,
) -> list[CellResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"IDM Baseline — head={head_profile}")
    print("=" * 60)

    t0 = time.time()
    results: list[CellResult] = []
    traces: dict[tuple[str, int], dict] = {}
    total = len(CONTROLLERS) * len(seeds)
    idx = 0
    for ctrl in CONTROLLERS:
        for seed in seeds:
            idx += 1
            record = seed == seeds[0]
            print(f"  [{idx:3d}/{total}] {ctrl} seed={seed} (delay=0ms)")
            result, trace = run_one_cell(
                ctrl, head_profile, 0.0, seed,
                config, Q, R, qp_ctrl, nn_ctrl, eval_hdv_config,
                total_time=total_time, return_trace=record,
            )
            results.append(result)
            if trace is not None:
                traces[(ctrl, seed)] = trace
    print(f"  Done in {time.time() - t0:.1f}s")

    write_csv(results, out_dir / "results.csv")

    plot_baseline_bars(
        results, "total_cost",
        f"Total Cost at delay=0 — head={head_profile}",
        "Total cost", out_dir / "bar_cost.png", fmt=".0f",
    )
    plot_baseline_bars(
        results, "msve_avg",
        f"MSVE at delay=0 — head={head_profile}",
        "MSVE", out_dir / "bar_msve.png", fmt=".3f",
    )
    plot_baseline_bars(
        results, "aeb_trigger_count",
        f"AEB Triggers at delay=0 — head={head_profile}",
        "AEB triggers", out_dir / "bar_aeb.png", fmt=".1f",
    )
    plot_baseline_bars(
        results, "min_spacing",
        f"Min CAV Spacing at delay=0 — head={head_profile}",
        "Min spacing (m)", out_dir / "bar_min_spacing.png", fmt=".2f",
    )

    ts_dir = out_dir / "timeseries"
    for ctrl in CONTROLLERS:
        key = (ctrl, seeds[0])
        if key not in traces:
            continue
        plot_timeseries(
            traces[key], config, ctrl, "idm",
            ts_dir / f"{ctrl.lower()}.png", delay_ms=0.0,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeeP-LCC IDM baseline evaluation (no V2V delay)."
    )
    parser.add_argument("--head", choices=("flat", "sine", "both"), default="both")
    args = parser.parse_args()

    config = DeepLCCConfig()
    Q, R = build_weights(config)

    nnmpc_cfg = NNMPCConfig()  # existing OVM-trained checkpoint
    qp_ctrl, nn_ctrl, eval_hdv_config = build_controllers(
        config, Q, R, nnmpc_cfg,
    )

    heads = HEAD_PROFILES if args.head == "both" else (args.head,)
    seeds = SEEDS
    total_time = 100.0
    base_dir = Path("deep_lcc_results/idm_baseline")

    for head in heads:
        run_head_profile(
            head, config, Q, R, qp_ctrl, nn_ctrl, eval_hdv_config,
            base_dir / head, seeds=seeds, total_time=total_time,
        )


if __name__ == "__main__":
    main()
