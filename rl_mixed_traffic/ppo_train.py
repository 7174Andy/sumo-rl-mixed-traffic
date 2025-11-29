from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.agents.ppo_agent import PPOAgent

from rl_mixed_traffic.utils.plot_utils import plot_returns, plot_ppo_metrics
import numpy as np


def make_env(gui: bool = False):
    """Create the ring road environment for continuous PPO.

    Args:
        gui: Whether to use SUMO GUI

    Returns:
        RingRoadEnv with continuous action space (no discretization wrapper)
    """
    sumo_config = SumoConfig(
        sumocfg_path="configs/ring/simulation.sumocfg",
        use_gui=gui,
    )
    env = RingRoadEnv(
        sumo_config=sumo_config,
        gui=gui,
        num_vehicles=4,
    )
    return env


def train(
    total_steps: int = 500_000,
    rollout_steps: int = 2048,
    save_freq: int = 50_000,
):
    """Train PPO agent on ring road environment with continuous actions.

    Args:
        total_steps: Total number of environment steps to train for
        rollout_steps: Number of steps to collect before each policy update
        save_freq: Save agent checkpoint every N steps

    Returns:
        List of episode returns
    """
    env = make_env(gui=False)

    returns = []
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # Continuous action space dimension

    cfg = PPOConfig()
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=cfg,
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

    print(f"Starting PPO training for {total_steps} steps")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim} (continuous)")
    print(f"Rollout steps: {rollout_steps}, Update frequency: every {rollout_steps} steps")
    print(f"Config: {cfg}")
    print("-" * 80)

    while step_count < total_steps:
        # Collect rollout
        for _ in range(rollout_steps):
            if step_count >= total_steps:
                break

            # Get action, value, and log prob from agent
            action, value, log_prob = agent.get_action_and_value(s)

            # Clip action to environment bounds
            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )

            # Step environment
            s_next, r, done, truncated, _ = env.step(action)

            # Store transition
            agent.store_transition(s, action, r, value, log_prob, done)

            s = s_next
            ep_ret += r
            ep_len += 1
            step_count += 1

            if done or truncated:
                print(
                    f"Step: {step_count:>7d} | Episode Return: {ep_ret:>8.2f} | "
                    f"Episode Length: {ep_len:>4d} | Updates: {agent.update_count}"
                )
                returns.append(ep_ret)
                s, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        # Get last value for bootstrapping (if episode didn't end)
        if step_count < total_steps:
            _, last_value, _ = agent.get_action_and_value(s)
        else:
            last_value = 0.0  # Terminal state value is 0

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
        if step_count % save_freq == 0 or step_count >= total_steps:
            agent.save(f"ppo_results/ppo_agent_step_{step_count}.pth")
            print(f"Saved checkpoint at step {step_count}")

    # Save final agent
    agent.save("ppo_results/ppo_agent.pth")
    env.close()

    # Plot training metrics
    plot_ppo_metrics(metrics_history, out_dir="ppo_results")

    print("-" * 80)
    print(f"Training completed! Total episodes: {len(returns)}")
    print("Final agent saved to: ppo_results/ppo_agent.pth")

    return returns


if __name__ == "__main__":
    returns = train()
    plot_returns(
        returns,
        out_path="ppo_results/ppo_training_returns.png",
        title="PPO Training Returns",
    )
