"""Train Warm-Start RL or RL+NNMPC residual via PPO on PlatoonNNMPCEnv.

Usage:
    uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode warm_start
    uv run rl_mixed_traffic/deep_lcc/rlmpc_train.py --mode residual

Outputs:
    deep_lcc_results/rlmpc_{mode}/agent.pth       — final model
    deep_lcc_results/rlmpc_{mode}/agent_step_N.pth — per-checkpoint snapshots
    deep_lcc_results/rlmpc_{mode}/returns.csv     — per-episode return + diagnostics
    deep_lcc_results/rlmpc_{mode}/returns.png     — return curve
    deep_lcc_results/rlmpc_{mode}/ppo_metrics.png — policy/value/entropy/clip
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.deep_lcc.nnmpc_actor_critic import NNMPCActorCritic
from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_env import PlatoonNNMPCEnv
from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper
from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_ppo_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["warm_start", "residual"], default="warm_start")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def build_network(config: RLMPCConfig) -> NNMPCActorCritic:
    log_std_init = (
        config.log_std_init_warm if config.mode == "warm_start"
        else config.log_std_init_residual
    )
    final_gain = (
        config.final_layer_gain_warm if config.mode == "warm_start"
        else config.final_layer_gain_residual
    )
    network = NNMPCActorCritic(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        hidden_dims=(256, 128),
        log_std_init=log_std_init,
        final_layer_gain=final_gain,
    )
    if config.mode == "warm_start":
        network.warm_start_from_nnmpc(config.nnmpc_path)
    return network


def scale_action(
    action_tanh: np.ndarray, config: RLMPCConfig,
) -> np.ndarray:
    """Map tanh-squashed action ∈ [-1,1]^2 to the per-mode action box."""
    a = np.asarray(action_tanh, dtype=np.float32).reshape(-1)
    if config.mode == "warm_start":
        # Asymmetric: [-5, 3]
        amin, amax = config.accel_min, config.accel_max
        return 0.5 * (a + 1.0) * (amax - amin) + amin
    # Residual: symmetric ±residual_max
    return a * config.residual_max


def train(config: RLMPCConfig) -> tuple[list[float], Path]:
    out_dir = Path(config.out_dir.format(mode=config.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = PlatoonNNMPCEnv(config)
    env = FourToFiveTupleWrapper(env)

    network = build_network(config)
    agent = PPOAgent(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        config=config.ppo,
        continuous=True,
        total_steps=config.total_steps,
        rollout_steps=config.rollout_steps,
    )
    # Replace the default ActorCriticNetwork with the NNMPC-shaped one.
    # Both implement the same get_action_and_value contract; PPOAgent only uses that surface.
    agent.network = network.to(agent.device)  # ty: ignore[invalid-assignment]
    # Re-create optimizer/scheduler around the new network parameters.
    import torch
    agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=config.ppo.lr)
    if config.ppo.anneal_lr and config.total_steps > 0:
        max_updates = max(config.total_steps // config.rollout_steps, 1)
        lr_lambda = lambda u: 1.0 - (u / max_updates)
    else:
        lr_lambda = lambda _u: 1.0
    agent.scheduler = torch.optim.lr_scheduler.LambdaLR(
        agent.optimizer, lr_lambda=lr_lambda,
    )

    print(f"[rlmpc_train] mode={config.mode} obs_dim={config.obs_dim} "
          f"action_dim={config.action_dim} total_steps={config.total_steps}")

    returns: list[float] = []
    metrics_history: dict[str, list[float]] = {
        "policy_loss": [], "value_loss": [], "entropy": [], "clipfrac": [],
    }

    s, _ = env.reset(seed=config.seed)
    ep_ret, ep_len = 0.0, 0
    ep_collisions = 0
    ep_violations = 0.0
    step_count = 0

    csv_path = out_dir / "returns.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["episode", "return", "length", "collisions", "mean_violation"]
    )

    while step_count < config.total_steps:
        last_done = False
        for _ in range(config.rollout_steps):
            if step_count >= config.total_steps:
                break

            action_tanh, value, log_prob = agent.get_action_and_value(s)
            action_scaled = scale_action(action_tanh, config)
            s_next, r, done, truncated, info = env.step(action_scaled)

            agent.store_transition(
                s, action_tanh, r, value, log_prob, done or truncated,
            )

            s = s_next
            ep_ret += r
            ep_len += 1
            ep_collisions += int(info.get("collision", False))
            ep_violations += float(info.get("violation", 0.0))
            step_count += 1
            episode_done = done or truncated
            last_done = episode_done

            if episode_done:
                returns.append(ep_ret)
                csv_writer.writerow([
                    len(returns), f"{ep_ret:.4f}", ep_len, ep_collisions,
                    f"{ep_violations / max(ep_len, 1):.4f}",
                ])
                csv_file.flush()
                print(f"Step {step_count:>7d} | Ep {len(returns):>5d} | "
                      f"Ret={ep_ret:>8.2f} | Len={ep_len:>4d} | "
                      f"Coll={ep_collisions} | Viol={ep_violations / max(ep_len, 1):.3f}")
                s, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
                ep_collisions = 0
                ep_violations = 0.0

        # Bootstrap last value
        if last_done:
            last_value = 0.0
        else:
            _, last_value, _ = agent.get_action_and_value(s)

        m = agent.learn(last_value=last_value)
        if m:
            for k in metrics_history:
                if k in m:
                    metrics_history[k].append(m[k])
            print(f"  Update {agent.update_count:>4d} | "
                  f"PL={m['policy_loss']:>7.4f} | "
                  f"VL={m['value_loss']:>7.4f} | "
                  f"Ent={m['entropy']:>6.4f} | "
                  f"CF={m['clipfrac']:>5.3f}")

        if step_count % config.save_freq == 0 or step_count >= config.total_steps:
            agent.save(str(out_dir / f"agent_step_{step_count}.pth"))

    csv_file.close()

    agent.save(str(out_dir / "agent.pth"))
    env.close()

    plot_returns(returns, out_path=str(out_dir / "returns.png"),
                 title=f"RLMPC ({config.mode}) Training Returns")
    plot_ppo_metrics(metrics_history, out_dir=str(out_dir))

    print(f"[rlmpc_train] Done. Saved to {out_dir}")
    return returns, out_dir


def main() -> None:
    args = parse_args()
    config = RLMPCConfig(mode=args.mode, seed=args.seed, use_gui=args.gui)
    if args.total_steps is not None:
        config.total_steps = args.total_steps
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    train(config)


if __name__ == "__main__":
    main()
