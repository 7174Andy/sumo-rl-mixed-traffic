"""Evaluate RLMPC controllers on the SUMO platoon across scenarios + HDV configs.

Compares: NNMPC alone, Warm-Start RL, RL+NNMPC (residual).
Produces per-(scenario, hdv_config, controller) metrics + plots.

Usage:
    uv run rl_mixed_traffic/deep_lcc/rlmpc_eval.py
    SCENARIOS=brake,sinusoidal uv run rl_mixed_traffic/deep_lcc/rlmpc_eval.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.deep_lcc.eval_classical import (
    make_aggressive_sine,
    make_extreme_brake,
    make_nedc,
    make_sinusoidal,
    make_stop_and_go,
    make_varying_sine,
)
from rl_mixed_traffic.deep_lcc.nnmpc_actor_critic import NNMPCActorCritic
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_env import PlatoonNNMPCEnv
from rl_mixed_traffic.deep_lcc.rlmpc_head_controller import PerturbMixHeadController
from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper


# ----------------------------------------------------------------------
# Controller adapters (each takes obs → action in the env's action space)
# ----------------------------------------------------------------------


def make_nnmpc_controller(nnmpc_path: str):
    """NNMPC alone: forward pass returns full action ∈ [-5, 3]."""
    ckpt = torch.load(nnmpc_path, map_location="cpu", weights_only=False)
    nnmpc = NNMPCNetwork(
        input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"],
        hidden_dims=ckpt["config"]["hidden_dims"],
        accel_min=ckpt["config"]["accel_min"],
        accel_max=ckpt["config"]["accel_max"],
    )
    nnmpc.load_state_dict(ckpt["model_state_dict"])
    nnmpc.eval()

    def controller(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = nnmpc(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().ravel()
        return out  # already in [-5, 3]

    return controller


def make_rl_controller(agent_path: str, config: RLMPCConfig):
    """Warm-start or residual RL policy: deterministic mean action."""
    network = NNMPCActorCritic(
        obs_dim=config.obs_dim, action_dim=config.action_dim,
        hidden_dims=(256, 128),
        log_std_init=(
            config.log_std_init_warm if config.mode == "warm_start"
            else config.log_std_init_residual
        ),
    )
    agent = PPOAgent(
        obs_dim=config.obs_dim, action_dim=config.action_dim,
        config=config.ppo, continuous=True,
    )
    agent.network = network  # ty: ignore[invalid-assignment]
    agent.load(agent_path, map_location="cpu")
    agent.network.eval()

    def controller(obs: np.ndarray) -> np.ndarray:
        action_tanh = agent.act(obs, eval_mode=True)
        action_tanh = np.asarray(action_tanh, dtype=np.float32).reshape(-1)
        if config.mode == "warm_start":
            amin, amax = config.accel_min, config.accel_max
            return 0.5 * (action_tanh + 1.0) * (amax - amin) + amin
        return action_tanh * config.residual_max

    return controller


# ----------------------------------------------------------------------
# Eval runner — runs one (controller, scenario, hdv_seed) and returns metrics
# ----------------------------------------------------------------------


def run_eval_episode(
    config: RLMPCConfig,
    controller_fn: Callable[[np.ndarray], np.ndarray],
    head_trace: np.ndarray,
    hdv_seed: int,
) -> dict:
    """Run a single eval episode and collect metrics."""
    eval_cfg = replace(
        config,
        episode_length_s=len(head_trace) * config.Tstep,
    )
    inner = PlatoonNNMPCEnv(eval_cfg)
    env = FourToFiveTupleWrapper(inner)

    obs, _ = env.reset(seed=hdv_seed)
    head = inner.head_vehicle_controller
    assert isinstance(head, PerturbMixHeadController), (
        f"head controller is {type(head).__name__}, expected PerturbMixHeadController"
    )
    head.trace = head_trace.copy()
    head._step_idx = 0

    cum_reward = 0.0
    cum_cost = 0.0
    n_collisions = 0
    n_violations = 0
    cav_velocities: dict[str, list[float]] = {aid: [] for aid in inner.agent_ids}
    cav_actions: dict[str, list[float]] = {aid: [] for aid in inner.agent_ids}
    head_velocities: list[float] = []
    spacings: dict[str, list[float]] = {aid: [] for aid in inner.agent_ids}
    latencies_us: list[float] = []

    done = False
    truncated = False
    while not (done or truncated):
        t0 = time.perf_counter()
        action = controller_fn(obs)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

        obs, reward, done, truncated, info = env.step(action)
        cum_reward += reward
        cum_cost += (1.0 - info["r_base"]) * inner.J_max_multi
        if info["collision"]:
            n_collisions += 1
        if info["violation"] > 0:
            n_violations += 1

        import traci
        for aid in inner.agent_ids:
            if aid in traci.vehicle.getIDList():
                cav_velocities[aid].append(traci.vehicle.getSpeed(aid))
                cav_actions[aid].append(inner.prev_accels[aid])
                spacings[aid].append(inner._get_gap_to_leader(aid))
        head_velocities.append(
            traci.vehicle.getSpeed("car0")
            if "car0" in traci.vehicle.getIDList() else 0.0
        )

    env.close()

    cav_v_arr = {a: np.asarray(v, dtype=np.float32) for a, v in cav_velocities.items()}
    head_v = np.asarray(head_velocities, dtype=np.float32)
    n = min(len(head_v), min(len(v) for v in cav_v_arr.values()))
    msve = {
        a: float(np.mean((cav_v_arr[a][:n] - head_v[:n]) ** 2))
        for a in cav_v_arr
    }
    msve_avg = float(np.mean(list(msve.values())))

    s_arrs = {a: np.asarray(s, dtype=np.float32) for a, s in spacings.items()}
    min_spacing = float(min(s.min() for s in s_arrs.values())) if s_arrs else 0.0
    max_spacing = float(max(s.max() for s in s_arrs.values())) if s_arrs else 0.0

    return {
        "total_cost": cum_cost,
        "cumulative_reward": cum_reward,
        "msve": msve,
        "msve_avg": msve_avg,
        "n_collisions": n_collisions,
        "n_violations": n_violations,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "mean_latency_us": float(np.mean(latencies_us)) if latencies_us else 0.0,
        "head_velocities": head_v,
        "cav_velocities": cav_v_arr,
        "cav_actions": {a: np.asarray(v) for a, v in cav_actions.items()},
    }


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------


def plot_velocity_comparison(
    scenario_name: str, hdv_label: str,
    results_per_controller: dict[str, dict],
    Tstep: float, out_dir: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    head = next(iter(results_per_controller.values()))["head_velocities"]
    t = np.arange(len(head)) * Tstep
    axes[0].plot(t, head, color="gray", linestyle="--", linewidth=1.2,
                 label="head", alpha=0.8)
    cmap = plt.get_cmap("tab10")
    for i, (ctrl_name, res) in enumerate(results_per_controller.items()):
        for j, aid in enumerate(res["cav_velocities"]):
            v = res["cav_velocities"][aid]
            n = min(len(t), len(v))
            axes[0].plot(t[:n], v[:n],
                         color=cmap(i), linestyle=("-" if j == 0 else ":"),
                         linewidth=1.0,
                         label=f"{ctrl_name} {aid}")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title(f"{scenario_name} / {hdv_label} — CAV velocities")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i, (_ctrl_name, res) in enumerate(results_per_controller.items()):
        for j, aid in enumerate(res["cav_actions"]):
            a = res["cav_actions"][aid]
            n = min(len(t), len(a))
            axes[1].plot(t[:n], a[:n],
                         color=cmap(i), linestyle=("-" if j == 0 else ":"),
                         linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Applied a (m/s²)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{scenario_name}_{hdv_label}_velocities.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    nnmpc_cfg = RLMPCConfig(mode="warm_start")
    warm_cfg = RLMPCConfig(mode="warm_start")
    resid_cfg = RLMPCConfig(mode="residual")

    out_dir = Path("deep_lcc_results/rlmpc/")
    out_dir.mkdir(parents=True, exist_ok=True)

    Tstep = warm_cfg.Tstep
    v_star = warm_cfg.v_star
    scen_specs = {
        "brake": (lambda: make_extreme_brake(int(30.0 / Tstep), Tstep, v_star)),
        "sinusoidal": (lambda: make_sinusoidal(int(40.0 / Tstep), Tstep, v_star, amplitude=2.0)),
        "varying_sine": (lambda: make_varying_sine(200.0, Tstep, v_star)),
        "aggressive_sine": (lambda: make_aggressive_sine(200.0, Tstep, v_star)),
        "stop_and_go": (lambda: make_stop_and_go(200.0, Tstep, v_star)),
        "NEDC": (lambda: make_nedc(Tstep)),
    }
    if "SCENARIOS" in os.environ:
        wanted = {s.strip() for s in os.environ["SCENARIOS"].split(",")}
        scen_specs = {k: v for k, v in scen_specs.items() if k in wanted}

    hdv_configs = {
        "nominal": [42],
        "hetero_fixed": [999],
        "hetero_random": [101, 202, 303, 404, 505],
    }

    nnmpc_ctrl = make_nnmpc_controller(nnmpc_cfg.nnmpc_path)
    warm_path = Path(warm_cfg.out_dir.format(mode="warm_start")) / "agent.pth"
    resid_path = Path(resid_cfg.out_dir.format(mode="residual")) / "agent.pth"
    warm_ctrl = make_rl_controller(str(warm_path), warm_cfg) if warm_path.exists() else None
    resid_ctrl = make_rl_controller(str(resid_path), resid_cfg) if resid_path.exists() else None

    controllers = {"nnmpc": (nnmpc_ctrl, nnmpc_cfg)}
    if warm_ctrl is not None:
        controllers["warm_rl"] = (warm_ctrl, warm_cfg)
    if resid_ctrl is not None:
        controllers["rl_residual"] = (resid_ctrl, resid_cfg)

    csv_path = out_dir / "summary.csv"
    rows = []
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "hdv_config", "seed", "controller",
            "total_cost", "cum_reward", "msve_avg",
            "n_collisions", "n_violations",
            "min_spacing", "max_spacing", "mean_latency_us",
        ])

        for scen_name, scen_fn in scen_specs.items():
            head_trace = scen_fn()
            for hdv_label, seeds in hdv_configs.items():
                for seed in seeds:
                    results_per_ctrl = {}
                    for ctrl_name, (ctrl_fn, ctrl_cfg) in controllers.items():
                        print(f"[{scen_name}/{hdv_label}/seed={seed}] running {ctrl_name}")
                        res = run_eval_episode(ctrl_cfg, ctrl_fn, head_trace, seed)
                        results_per_ctrl[ctrl_name] = res
                        writer.writerow([
                            scen_name, hdv_label, seed, ctrl_name,
                            f"{res['total_cost']:.4f}",
                            f"{res['cumulative_reward']:.4f}",
                            f"{res['msve_avg']:.4f}",
                            res['n_collisions'], res['n_violations'],
                            f"{res['min_spacing']:.3f}", f"{res['max_spacing']:.3f}",
                            f"{res['mean_latency_us']:.1f}",
                        ])
                        f.flush()
                        rows.append((scen_name, hdv_label, seed, ctrl_name, res))
                    plot_velocity_comparison(
                        f"{scen_name}", f"{hdv_label}_seed{seed}",
                        results_per_ctrl, Tstep, out_dir,
                    )

    # Render a markdown summary grouped by scenario, ranked by total cost.
    md_path = out_dir / "summary.md"
    with md_path.open("w") as md:
        md.write("# RLMPC Eval Summary\n\n")
        from collections import defaultdict
        grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for scen, hdv, _seed, ctrl, res in rows:
            grouped[(scen, hdv)][ctrl].append(res["total_cost"])
        for (scen, hdv), per_ctrl in grouped.items():
            md.write(f"## {scen} / {hdv}\n\n")
            md.write("| controller | mean cost | n seeds |\n")
            md.write("|---|---:|---:|\n")
            ranked = sorted(
                per_ctrl.items(), key=lambda kv: float(np.mean(kv[1]))
            )
            for ctrl, costs in ranked:
                md.write(f"| {ctrl} | {float(np.mean(costs)):.2f} | {len(costs)} |\n")
            md.write("\n")
    print(f"[rlmpc_eval] Done. Summary → {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
