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
):
    """Create the ring road environment for continuous PPO training.

    Applies the wrapper chain:
        RingRoadEnv -> FourToFiveTuple -> ClipAction

    Args:
        sumocfg_path: Path to the SUMO config file
        gui: Whether to use SUMO GUI
        num_vehicles: Total number of vehicles in the ring (including head)
        num_agents: Number of CAVs controlled by RL (car1..carN). car0 is head.
        head_vehicle_controller: Optional controller for the head vehicle.
            If None, the default random controller is used.

    Returns:
        Wrapped environment with 5-tuple step output
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
    )
    env = FourToFiveTupleWrapper(env)
    env = gym.wrappers.ClipAction(env)
    return env


def train(cfg: DictConfig):
    """Train PPO agent on ring road environment with continuous actions.

    Supports both single-agent (num_agents=1) and multi-agent (num_agents>1) modes.
    In multi-agent mode, a shared PPO policy is queried once per agent per step,
    and all agents share a single global reward.

    Args:
        cfg: Hydra DictConfig with training parameters

    Returns:
        Tuple of (episode returns list, output directory Path)
    """
    orig_cwd = hydra.utils.get_original_cwd()
    sumocfg_path = str(Path(orig_cwd) / cfg.env.sumocfg_path)
    num_agents = OmegaConf.select(cfg, "env.num_agents", default=1)

    head_controller = make_head_controller(cfg.scenario)

    env = make_env(
        sumocfg_path=sumocfg_path,
        gui=cfg.gui,
        num_vehicles=cfg.env.num_vehicles,
        num_agents=num_agents,
        head_vehicle_controller=head_controller,
    )

    if num_agents > 1:
        return _train_multi_agent(cfg, env, orig_cwd, num_agents)
    else:
        return _train_single_agent(cfg, env, orig_cwd)


def _train_single_agent(cfg: DictConfig, env: RingRoadEnv, orig_cwd: str):
    """Original single-agent PPO training loop."""
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

    metrics_history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clipfrac": [],
    }

    s, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    step_count = 0

    # Scalar max accel for tanh→action scaling: [-1,1] * max_accel → [-3, 3]
    max_accel = float(env.unwrapped.max_accel)

    print(f"Starting PPO training for {cfg.total_steps} steps (single-agent)")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim} (continuous)")
    print(f"Rollout steps: {cfg.rollout_steps}, Update frequency: every {cfg.rollout_steps} steps")
    print(f"Config: {ppo_cfg}")
    print("-" * 80)

    while step_count < cfg.total_steps:
        last_done = False
        for _ in range(cfg.rollout_steps):
            if step_count >= cfg.total_steps:
                break

            action_tanh, value, log_prob = agent.get_action_and_value(s)

            action_scaled = action_tanh * max_accel

            s_next, r, done, truncated, _ = env.step(action_scaled)

            agent.store_transition(s, action_tanh, r, value, log_prob, done or truncated)

            s = s_next
            ep_ret += r
            ep_len += 1
            step_count += 1
            episode_done = done or truncated
            last_done = episode_done

            if episode_done:
                print(
                    f"Step: {step_count:>7d} | Episode Return: {ep_ret:>8.2f} | "
                    f"Episode Length: {ep_len:>4d} | Updates: {agent.update_count}"
                )
                returns.append(ep_ret)
                s, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

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
                f"Clip Frac: {metrics['clipfrac']:>5.3f}"
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
    cfg: DictConfig, env: RingRoadEnv, orig_cwd: str, num_agents: int
):
    """Multi-agent PPO training loop with shared policy and shared reward.

    Each env step:
    1. For each agent i: query shared PPO policy with obs_i -> action_i, value_i, log_prob_i
    2. Concatenate actions -> env.step(all_actions)
    3. Get shared reward r from env
    4. For each agent i: store_transition(obs_i, action_i, r, value_i, log_prob_i, done)

    The buffer gets num_agents transitions per env step.
    """
    returns = []

    # Shared policy: obs_dim = 2 * num_vehicles + 1, action_dim = 1 (per-agent)
    obs_dim = env.observation_space.shape[0]  # 2 * num_vehicles + 1
    action_dim = 1  # Each agent outputs one acceleration

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

    metrics_history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clipfrac": [],
    }

    agent_ids = env.agent_ids

    # Reset env and get per-agent observations
    obs_dict, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    step_count = 0

    # Scalar max accel for tanh→action scaling
    max_accel = float(env.unwrapped.max_accel)

    print(f"Starting PPO training for {cfg.total_steps} steps (multi-agent, {num_agents} CAVs)")
    print(f"Observation dim: {obs_dim} (global state + agent index), Action dim: {action_dim} (per-agent)")
    print(f"Agent IDs: {agent_ids}")
    print(f"Rollout steps: {cfg.rollout_steps}, Update frequency: every {cfg.rollout_steps} steps")
    print(f"Config: {ppo_cfg}")
    print("-" * 80)

    while step_count < cfg.total_steps:
        last_done = False
        for _ in range(cfg.rollout_steps):
            if step_count >= cfg.total_steps:
                break

            # Query shared policy for each agent
            per_agent_data = []
            all_actions_scaled = np.zeros(num_agents, dtype=np.float32)

            for i, aid in enumerate(agent_ids):
                obs_i = obs_dict[aid]

                # Get action (tanh output in [-1, 1]), value, and log prob
                action_tanh_i, value_i, log_prob_i = agent.get_action_and_value(obs_i)

                # Scale action from [-1, 1] to [-max_accel, max_accel]
                action_scaled_i = action_tanh_i * max_accel
                all_actions_scaled[i] = float(np.asarray(action_scaled_i).reshape(-1)[0])

                per_agent_data.append({
                    "obs": obs_i,
                    "action_tanh": action_tanh_i,
                    "value": value_i,
                    "log_prob": log_prob_i,
                })

            # Step environment with scaled actions
            obs_dict_next, r, done, truncated, _ = env.step(all_actions_scaled)

            # Store one transition per agent with the SHARED reward
            episode_done = done or truncated
            for i, aid in enumerate(agent_ids):
                data = per_agent_data[i]
                agent.store_transition(
                    data["obs"],
                    data["action_tanh"],
                    r,  # shared reward
                    data["value"],
                    data["log_prob"],
                    episode_done,
                )

            obs_dict = obs_dict_next
            ep_ret += r
            ep_len += 1
            step_count += 1
            last_done = episode_done

            if episode_done:
                print(
                    f"Step: {step_count:>7d} | Episode Return: {ep_ret:>8.2f} | "
                    f"Episode Length: {ep_len:>4d} | Updates: {agent.update_count}"
                )
                returns.append(ep_ret)
                obs_dict, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Get last value for bootstrapping
        if last_done:
            last_value = 0.0
        else:
            # Average value estimates across agents for bootstrapping
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
                f"Clip Frac: {metrics['clipfrac']:>5.3f}"
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


@hydra.main(version_base=None, config_path="conf", config_name="ppo_train")
def main(cfg: DictConfig):
    returns, out_dir = train(cfg)
    plot_returns(
        returns,
        out_path=str(out_dir / "ppo_training_returns.png"),
        title="PPO Training Returns",
    )


if __name__ == "__main__":
    main()
