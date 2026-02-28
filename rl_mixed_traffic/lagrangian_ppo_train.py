"""Lagrangian PPO training with safety layer.

Extends the standard PPO training loop with:
1. A safety layer that clips unsafe accelerations before applying them to SUMO
2. A Lagrange multiplier that penalizes spacing violations during training

The safety layer provides hard guarantees at execution time, while the
Lagrangian penalty teaches the policy to naturally avoid unsafe states.
Over time, safety layer interventions should approach zero.
"""

from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym

from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper
from rl_mixed_traffic.env.scenario import make_head_controller
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_ppo_metrics


def make_env(
    sumocfg_path: str,
    gui: bool = False,
    num_vehicles: int = 4,
    num_agents: int = 1,
    head_vehicle_controller=None,
    enable_safety_layer: bool = True,
    disable_sumo_safety: bool = True,
    spacing_min: float = 5.0,
):
    """Create the ring road environment with safety layer for Lagrangian PPO.

    Args:
        sumocfg_path: Path to the SUMO config file.
        gui: Whether to use SUMO GUI.
        num_vehicles: Total number of vehicles in the ring (including head).
        num_agents: Number of CAVs controlled by RL.
        head_vehicle_controller: Optional controller for the head vehicle.
        enable_safety_layer: Enable the hard-constraint safety layer.
        disable_sumo_safety: Disable SUMO's built-in safety checks (setSpeedMode(0)).
        spacing_min: Minimum allowed spacing in meters.

    Returns:
        Wrapped environment with 5-tuple step output.
    """
    sumo_config = SumoConfig(
        sumocfg_path=sumocfg_path,
        use_gui=gui,
    )
    env = RingRoadEnv(
        sumo_config=sumo_config,
        gui=gui,
        num_vehicles=num_vehicles,
        num_agents=num_agents,
        head_vehicle_controller=head_vehicle_controller,
        enable_safety_layer=enable_safety_layer,
        disable_sumo_safety=disable_sumo_safety,
        spacing_min=spacing_min,
    )
    env = FourToFiveTupleWrapper(env)
    env = gym.wrappers.ClipAction(env)
    return env


def train(cfg: DictConfig):
    """Train Lagrangian PPO agent on ring road environment.

    Supports both single-agent and multi-agent modes.

    Args:
        cfg: Hydra DictConfig with training parameters.

    Returns:
        Tuple of (episode returns list, output directory Path).
    """
    orig_cwd = hydra.utils.get_original_cwd()
    sumocfg_path = str(Path(orig_cwd) / cfg.env.sumocfg_path)
    num_agents = OmegaConf.select(cfg, "env.num_agents", default=1)

    head_controller = make_head_controller(cfg.scenario)

    enable_safety_layer = OmegaConf.select(cfg, "env.enable_safety_layer", default=True)
    disable_sumo_safety = OmegaConf.select(cfg, "env.disable_sumo_safety", default=True)
    spacing_min = OmegaConf.select(cfg, "env.spacing_min", default=5.0)

    env = make_env(
        sumocfg_path=sumocfg_path,
        gui=cfg.gui,
        num_vehicles=cfg.env.num_vehicles,
        num_agents=num_agents,
        head_vehicle_controller=head_controller,
        enable_safety_layer=enable_safety_layer,
        disable_sumo_safety=disable_sumo_safety,
        spacing_min=spacing_min,
    )

    if num_agents > 1:
        return _train_multi_agent(cfg, env, orig_cwd, num_agents)
    else:
        return _train_single_agent(cfg, env, orig_cwd)


def _train_single_agent(cfg: DictConfig, env, orig_cwd: str):
    """Single-agent Lagrangian PPO training loop."""
    returns = []
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_cfg = PPOConfig(
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        lam=cfg.agent.lam,
        clip_epsilon=cfg.agent.clip_epsilon,
        k_epochs=cfg.agent.k_epochs,
        batch_size=cfg.agent.batch_size,
        entropy_coef=cfg.agent.entropy_coef,
        value_coef=cfg.agent.value_coef,
    )
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=ppo_cfg,
        continuous=True,
    )

    # Lagrangian multiplier
    enable_lagrangian = OmegaConf.select(cfg, "agent.enable_lagrangian", default=True)
    lambda_val = OmegaConf.select(cfg, "agent.lambda_init", default=0.0)
    lambda_lr = OmegaConf.select(cfg, "agent.lambda_lr", default=0.01)
    lambda_max = OmegaConf.select(cfg, "agent.lambda_max", default=10.0)

    metrics_history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clipfrac": [],
        "lambda": [],
        "mean_violation": [],
        "safety_clip_rate": [],
    }

    s, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    step_count = 0

    max_accel = float(env.unwrapped.max_accel)

    # Per-rollout tracking
    rollout_violations = []
    rollout_clips = []

    print(f"Starting Lagrangian PPO training for {cfg.total_steps} steps (single-agent)")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim} (continuous)")
    print(f"Safety layer: {env.unwrapped.enable_safety_layer}, "
          f"SUMO safety disabled: {env.unwrapped.disable_sumo_safety}")
    print(f"Lagrangian: {enable_lagrangian}, lambda_init={lambda_val}, "
          f"lambda_lr={lambda_lr}, lambda_max={lambda_max}")
    print(f"Rollout steps: {cfg.rollout_steps}")
    print(f"Config: {ppo_cfg}")
    print("-" * 80)

    while step_count < cfg.total_steps:
        last_done = False
        for _ in range(cfg.rollout_steps):
            if step_count >= cfg.total_steps:
                break

            action_tanh, value, log_prob = agent.get_action_and_value(s)
            action_scaled = action_tanh * max_accel

            s_next, r_base, done, truncated, info = env.step(action_scaled)

            # Track safety layer clips
            safety_clipped = info.get("safety_clipped", False)
            rollout_clips.append(1.0 if safety_clipped else 0.0)

            # Compute spacing violation for Lagrangian penalty
            violation = env.unwrapped.get_spacing_violation()
            rollout_violations.append(violation)

            # Augmented reward: r_base - lambda * violation
            if enable_lagrangian:
                r_augmented = r_base - lambda_val * violation
            else:
                r_augmented = r_base

            agent.store_transition(s, action_tanh, r_augmented, value, log_prob, done or truncated)

            s = s_next
            ep_ret += r_base  # Track base reward for fair comparison
            ep_len += 1
            step_count += 1
            episode_done = done or truncated
            last_done = episode_done

            if episode_done:
                print(
                    f"Step: {step_count:>7d} | Episode Return: {ep_ret:>8.2f} | "
                    f"Episode Length: {ep_len:>4d} | Lambda: {lambda_val:.4f} | "
                    f"Updates: {agent.update_count}"
                )
                returns.append(ep_ret)
                s, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Update Lagrange multiplier
        if enable_lagrangian and rollout_violations:
            mean_violation = float(np.mean(rollout_violations))
            lambda_val = float(np.clip(
                lambda_val + lambda_lr * mean_violation, 0.0, lambda_max
            ))
        else:
            mean_violation = 0.0

        safety_clip_rate = float(np.mean(rollout_clips)) if rollout_clips else 0.0

        # Log Lagrangian metrics
        metrics_history["lambda"].append(lambda_val)
        metrics_history["mean_violation"].append(mean_violation)
        metrics_history["safety_clip_rate"].append(safety_clip_rate)

        # Reset per-rollout trackers
        rollout_violations = []
        rollout_clips = []

        if last_done:
            last_value = 0.0
        else:
            _, last_value, _ = agent.get_action_and_value(s)

        metrics = agent.learn(last_value=last_value)

        if metrics:
            for key, val in metrics.items():
                if key in metrics_history:
                    metrics_history[key].append(val)

            print(
                f"Update {agent.update_count:>4d} | "
                f"Policy Loss: {metrics['policy_loss']:>7.4f} | "
                f"Value Loss: {metrics['value_loss']:>7.4f} | "
                f"Entropy: {metrics['entropy']:>6.4f} | "
                f"Lambda: {lambda_val:>6.4f} | "
                f"Violation: {mean_violation:>6.4f} | "
                f"Clip Rate: {safety_clip_rate:>5.3f}"
            )

        if step_count % cfg.save_freq == 0 or step_count >= cfg.total_steps:
            out_dir = Path(orig_cwd) / cfg.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(out_dir / f"ppo_agent_step_{step_count}.pth"))
            print(f"Saved checkpoint at step {step_count}")

    out_dir = Path(orig_cwd) / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    agent.save(str(out_dir / "ppo_agent.pth"))
    env.close()

    plot_ppo_metrics(metrics_history, out_dir=str(out_dir))

    print("-" * 80)
    print(f"Training completed! Total episodes: {len(returns)}")
    print(f"Final agent saved to: {out_dir / 'ppo_agent.pth'}")

    return returns, out_dir


def _train_multi_agent(
    cfg: DictConfig, env, orig_cwd: str, num_agents: int
):
    """Multi-agent Lagrangian PPO training loop with shared policy and shared reward."""
    returns = []

    obs_dim = env.observation_space.shape[0]
    action_dim = 1

    ppo_cfg = PPOConfig(
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        lam=cfg.agent.lam,
        clip_epsilon=cfg.agent.clip_epsilon,
        k_epochs=cfg.agent.k_epochs,
        batch_size=cfg.agent.batch_size,
        entropy_coef=cfg.agent.entropy_coef,
        value_coef=cfg.agent.value_coef,
    )
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=ppo_cfg,
        continuous=True,
    )

    # Lagrangian multiplier
    enable_lagrangian = OmegaConf.select(cfg, "agent.enable_lagrangian", default=True)
    lambda_val = OmegaConf.select(cfg, "agent.lambda_init", default=0.0)
    lambda_lr = OmegaConf.select(cfg, "agent.lambda_lr", default=0.01)
    lambda_max = OmegaConf.select(cfg, "agent.lambda_max", default=10.0)

    metrics_history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clipfrac": [],
        "lambda": [],
        "mean_violation": [],
        "safety_clip_rate": [],
    }

    agent_ids = env.agent_ids

    obs_dict, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    step_count = 0

    max_accel = float(env.unwrapped.max_accel)

    rollout_violations = []
    rollout_clips = []

    print(f"Starting Lagrangian PPO training for {cfg.total_steps} steps "
          f"(multi-agent, {num_agents} CAVs)")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim} (per-agent)")
    print(f"Agent IDs: {agent_ids}")
    print(f"Safety layer: {env.unwrapped.enable_safety_layer}, "
          f"SUMO safety disabled: {env.unwrapped.disable_sumo_safety}")
    print(f"Lagrangian: {enable_lagrangian}, lambda_init={lambda_val}, "
          f"lambda_lr={lambda_lr}, lambda_max={lambda_max}")
    print(f"Rollout steps: {cfg.rollout_steps}")
    print(f"Config: {ppo_cfg}")
    print("-" * 80)

    while step_count < cfg.total_steps:
        last_done = False
        for _ in range(cfg.rollout_steps):
            if step_count >= cfg.total_steps:
                break

            per_agent_data = []
            all_actions_scaled = np.zeros(num_agents, dtype=np.float32)

            for i, aid in enumerate(agent_ids):
                obs_i = obs_dict[aid]
                action_tanh_i, value_i, log_prob_i = agent.get_action_and_value(obs_i)
                action_scaled_i = action_tanh_i * max_accel
                all_actions_scaled[i] = float(np.asarray(action_scaled_i).reshape(-1)[0])
                per_agent_data.append({
                    "obs": obs_i,
                    "action_tanh": action_tanh_i,
                    "value": value_i,
                    "log_prob": log_prob_i,
                })

            obs_dict_next, r_base, done, truncated, info = env.step(all_actions_scaled)

            safety_clipped = info.get("safety_clipped", False)
            rollout_clips.append(1.0 if safety_clipped else 0.0)

            violation = env.unwrapped.get_spacing_violation()
            rollout_violations.append(violation)

            if enable_lagrangian:
                r_augmented = r_base - lambda_val * violation
            else:
                r_augmented = r_base

            episode_done = done or truncated
            for i, aid in enumerate(agent_ids):
                data = per_agent_data[i]
                agent.store_transition(
                    data["obs"],
                    data["action_tanh"],
                    r_augmented,
                    data["value"],
                    data["log_prob"],
                    episode_done,
                )

            obs_dict = obs_dict_next
            ep_ret += r_base
            ep_len += 1
            step_count += 1
            last_done = episode_done

            if episode_done:
                print(
                    f"Step: {step_count:>7d} | Episode Return: {ep_ret:>8.2f} | "
                    f"Episode Length: {ep_len:>4d} | Lambda: {lambda_val:.4f} | "
                    f"Updates: {agent.update_count}"
                )
                returns.append(ep_ret)
                obs_dict, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Update Lagrange multiplier
        if enable_lagrangian and rollout_violations:
            mean_violation = float(np.mean(rollout_violations))
            lambda_val = float(np.clip(
                lambda_val + lambda_lr * mean_violation, 0.0, lambda_max
            ))
        else:
            mean_violation = 0.0

        safety_clip_rate = float(np.mean(rollout_clips)) if rollout_clips else 0.0

        metrics_history["lambda"].append(lambda_val)
        metrics_history["mean_violation"].append(mean_violation)
        metrics_history["safety_clip_rate"].append(safety_clip_rate)

        rollout_violations = []
        rollout_clips = []

        if last_done:
            last_value = 0.0
        else:
            values = []
            for i, aid in enumerate(agent_ids):
                _, v_i, _ = agent.get_action_and_value(obs_dict[aid])
                values.append(v_i)
            last_value = float(np.mean(values))

        metrics = agent.learn(last_value=last_value)

        if metrics:
            for key, val in metrics.items():
                if key in metrics_history:
                    metrics_history[key].append(val)

            print(
                f"Update {agent.update_count:>4d} | "
                f"Policy Loss: {metrics['policy_loss']:>7.4f} | "
                f"Value Loss: {metrics['value_loss']:>7.4f} | "
                f"Entropy: {metrics['entropy']:>6.4f} | "
                f"Lambda: {lambda_val:>6.4f} | "
                f"Violation: {mean_violation:>6.4f} | "
                f"Clip Rate: {safety_clip_rate:>5.3f}"
            )

        if step_count % cfg.save_freq == 0 or step_count >= cfg.total_steps:
            out_dir = Path(orig_cwd) / cfg.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(out_dir / f"ppo_agent_step_{step_count}.pth"))
            print(f"Saved checkpoint at step {step_count}")

    out_dir = Path(orig_cwd) / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    agent.save(str(out_dir / "ppo_agent.pth"))
    env.close()

    plot_ppo_metrics(metrics_history, out_dir=str(out_dir))

    print("-" * 80)
    print(f"Training completed! Total episodes: {len(returns)}")
    print(f"Agents: {num_agents} CAVs with shared policy")
    print(f"Final agent saved to: {out_dir / 'ppo_agent.pth'}")

    return returns, out_dir


@hydra.main(version_base=None, config_path="conf", config_name="lagrangian_ppo_train")
def main(cfg: DictConfig):
    returns, out_dir = train(cfg)
    plot_returns(
        returns,
        out_path=str(out_dir / "lagrangian_ppo_training_returns.png"),
        title="Lagrangian PPO Training Returns",
    )


if __name__ == "__main__":
    main()
