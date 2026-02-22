from pathlib import Path
import hydra
from omegaconf import DictConfig

from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.agents.ppo_agent import PPOAgent

from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_ppo_metrics


def make_env(sumocfg_path: str, gui: bool = False, num_vehicles: int = 4):
    """Create the ring road environment for continuous PPO.

    Args:
        sumocfg_path: Path to the SUMO config file
        gui: Whether to use SUMO GUI
        num_vehicles: Number of vehicles in the ring

    Returns:
        RingRoadEnv with continuous action space (no discretization wrapper)
    """
    sumo_config = SumoConfig(
        sumocfg_path=sumocfg_path,
        use_gui=gui,
    )
    env = RingRoadEnv(
        sumo_config=sumo_config,
        gui=gui,
        num_vehicles=num_vehicles,
    )
    return env


def train(cfg: DictConfig):
    """Train PPO agent on ring road environment with continuous actions.

    Args:
        cfg: Hydra DictConfig with training parameters

    Returns:
        Tuple of (episode returns list, output directory Path)
    """
    orig_cwd = hydra.utils.get_original_cwd()
    sumocfg_path = str(Path(orig_cwd) / cfg.env.sumocfg_path)

    env = make_env(
        sumocfg_path=sumocfg_path,
        gui=cfg.gui,
        num_vehicles=cfg.env.num_vehicles,
    )

    returns = []
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # Continuous action space dimension

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
        continuous=True,  # Use continuous actions (Gaussian policy)
    )

    # Metrics tracking
    metrics_history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clipfrac": [],
    }

    s, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    step_count = 0

    print(f"Starting PPO training for {cfg.total_steps} steps")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim} (continuous)")
    print(f"Rollout steps: {cfg.rollout_steps}, Update frequency: every {cfg.rollout_steps} steps")
    print(f"Config: {ppo_cfg}")
    print("-" * 80)

    while step_count < cfg.total_steps:
        # Collect rollout
        last_done = False  # Track if the last transition was terminal
        for _ in range(cfg.rollout_steps):
            if step_count >= cfg.total_steps:
                break

            # Get action, value, and log prob from agent
            # action is in [-1, 1] range from tanh
            action_tanh, value, log_prob = agent.get_action_and_value(s)

            # Scale action from [-1, 1] (tanh output) to action space bounds [-3, 3]
            # action_scaled = (action + 1) / 2 * (high - low) + low
            action_scaled = (action_tanh + 1.0) / 2.0 * (env.action_space.high - env.action_space.low) + env.action_space.low

            # Step environment with scaled action
            s_next, r, done, _ = env.step(action_scaled)

            # Store transition with UNSCALED action (tanh output) for proper PPO training
            agent.store_transition(s, action_tanh, r, value, log_prob, done)

            s = s_next
            ep_ret += r
            ep_len += 1
            step_count += 1
            last_done = done  # Update last_done flag

            if done:
                print(
                    f"Step: {step_count:>7d} | Episode Return: {ep_ret:>8.2f} | "
                    f"Episode Length: {ep_len:>4d} | Updates: {agent.update_count}"
                )
                returns.append(ep_ret)
                s, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Get last value for bootstrapping
        # If the last transition ended the episode (done=True), use 0.0
        # Otherwise, bootstrap from the value of the current state
        if last_done:
            last_value = 0.0  # Terminal state has zero value
        else:
            _, last_value, _ = agent.get_action_and_value(s)

        # Update policy
        metrics = agent.learn(last_value=last_value)

        # Track metrics
        if metrics:
            for key, value in metrics.items():
                if key in metrics_history:
                    metrics_history[key].append(value)

            print(
                f"Update {agent.update_count:>4d} | "
                f"Policy Loss: {metrics['policy_loss']:>7.4f} | "
                f"Value Loss: {metrics['value_loss']:>7.4f} | "
                f"Entropy: {metrics['entropy']:>6.4f} | "
                f"Clip Frac: {metrics['clipfrac']:>5.3f}"
            )

        # Save checkpoint
        if step_count % cfg.save_freq == 0 or step_count >= cfg.total_steps:
            out_dir = Path(orig_cwd) / cfg.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(out_dir / f"ppo_agent_step_{step_count}.pth"))
            print(f"Saved checkpoint at step {step_count}")

    out_dir = Path(orig_cwd) / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save final agent
    agent.save(str(out_dir / "ppo_agent.pth"))
    env.close()

    # Plot training metrics
    plot_ppo_metrics(metrics_history, out_dir=str(out_dir))

    print("-" * 80)
    print(f"Training completed! Total episodes: {len(returns)}")
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
