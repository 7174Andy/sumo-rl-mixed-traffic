from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.utils.plot_utils import plot_vehicle_speeds

import numpy as np
import traci


def make_env(gui: bool = False):
    """Create the ring road environment for continuous PPO evaluation.

    Args:
        gui: Whether to use SUMO GUI

    Returns:
        RingRoadEnv with continuous action space
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


def evaluate(agent_path: str, gui: bool = True, plot_speeds: bool = True):
    """Evaluate a trained PPO agent on the ring road environment.

    Args:
        agent_path: Path to saved agent checkpoint
        gui: Whether to use SUMO GUI for visualization
        plot_speeds: Whether to plot vehicle speed diagram after evaluation

    Returns:
        Total episode return
    """
    env = make_env(gui=gui)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    cfg = PPOConfig()
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=cfg,
        continuous=True,
    )
    agent.load(agent_path)

    s, _ = env.reset()
    done = False
    truncated = False
    G = 0.0
    steps = 0

    # Track vehicle speeds
    head_speeds = []
    cav_speeds = []

    print(f"Starting evaluation with agent from: {agent_path}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print("-" * 80)

    try:
        while not (done or truncated):
            # Track speeds before action
            if env.head_id in traci.vehicle.getIDList():
                head_speeds.append(traci.vehicle.getSpeed(env.head_id))
            else:
                head_speeds.append(0.0)

            if env.agent_id in traci.vehicle.getIDList():
                cav_speeds.append(traci.vehicle.getSpeed(env.agent_id))
            else:
                cav_speeds.append(0.0)

            # Get action from agent (deterministic evaluation)
            # agent.act() returns tanh-squashed action in [-1, 1]
            a_tanh = agent.act(state=s, eval_mode=True)
            print(a_tanh)

            # Scale from [-1, 1] to action space bounds [-3, 3]
            # This matches the scaling used during training in ppo_train.py
            a = (a_tanh + 1.0) / 2.0 * (env.action_space.high - env.action_space.low) + env.action_space.low

            # Step environment
            s_next, r, done, _ = env.step(a)

            s = s_next
            G += r
            steps += 1

            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"Step: {steps}, Return so far: {G:.2f}")

        print("-" * 80)
        print("Evaluation completed!")
        print(f"Total Return: {G:.2f}")
        print(f"Total Steps: {steps}")
        print(f"Average Reward per Step: {G/steps:.4f}")

        # Plot vehicle speeds
        if plot_speeds and len(cav_speeds) > 0:
            print("Plotting vehicle speeds to: ppo_results/vehicle_speeds.png")
            plot_vehicle_speeds(
                head_speeds=head_speeds,
                cav_speeds=cav_speeds,
                out_path="ppo_results/vehicle_speeds.png",
                title="PPO Evaluation: Vehicle Speeds",
            )

    finally:
        env.close()

    return G


if __name__ == "__main__":
    evaluate(agent_path="ppo_results/ppo_agent.pth", gui=True)
