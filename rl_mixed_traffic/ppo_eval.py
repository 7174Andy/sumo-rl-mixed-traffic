import click
import gymnasium as gym
import time
import traci

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper
from rl_mixed_traffic.env.scenario import make_head_controller
from rl_mixed_traffic.configs.sumo_config import SumoConfig
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.utils.plot_utils import (
    plot_vehicle_speeds,
    plot_cav_spacing,
    plot_accelerations,
)


class _ScenarioCfg:
    """Minimal config object for make_head_controller."""
    def __init__(self, type: str):
        self.type = type


def make_env(gui: bool = False, head_vehicle_controller=None):
    """Create the ring road environment for continuous PPO evaluation.

    Applies FourToFiveTuple -> ClipAction only (no reward normalization
    so we get raw rewards for eval reporting).

    Args:
        gui: Whether to use SUMO GUI
        head_vehicle_controller: Optional controller for the head vehicle.

    Returns:
        Wrapped environment with 5-tuple step output
    """
    sumo_config = SumoConfig(
        sumocfg_path="configs/ring/simulation.sumocfg",
        use_gui=gui,
    )
    env = RingRoadEnv(
        sumo_config=sumo_config,
        gui=gui,
        num_vehicles=4,
        head_vehicle_controller=head_vehicle_controller,
    )
    env = FourToFiveTupleWrapper(env)
    env = gym.wrappers.ClipAction(env)
    return env


def evaluate(agent_path: str, gui: bool = True, plot_speeds: bool = True, scenario: str = "random"):
    """Evaluate a trained PPO agent on the ring road environment.

    Args:
        agent_path: Path to saved agent checkpoint
        gui: Whether to use SUMO GUI for visualization
        plot_speeds: Whether to plot vehicle speed diagram after evaluation
        scenario: Scenario type ("random" or "emergency_braking")

    Returns:
        Total episode return
    """
    scenario_cfg = _ScenarioCfg(type=scenario.replace("-", "_"))
    head_controller = make_head_controller(scenario_cfg)

    env = make_env(gui=gui, head_vehicle_controller=head_controller)
    raw_env = env.unwrapped  # access RingRoadEnv attributes through wrappers

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
    max_accel = float(env.unwrapped.max_accel)
    done = False
    truncated = False
    G = 0.0
    steps = 0

    # Track vehicle speeds, spacing, and accelerations
    head_speeds = []
    cav_speeds = []
    head_accels = []
    cav_spacings = {aid: [] for aid in raw_env.agent_ids}
    cav_accels = {aid: [] for aid in raw_env.agent_ids}

    print(f"Starting evaluation with agent from: {agent_path}")
    print(f"Scenario: {scenario}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print("-" * 80)

    try:
        while not done and not truncated:
            # Track speeds before action
            if raw_env.head_id in traci.vehicle.getIDList():
                head_speeds.append(traci.vehicle.getSpeed(raw_env.head_id))
            else:
                head_speeds.append(0.0)

            if raw_env.agent_id in traci.vehicle.getIDList():
                cav_speeds.append(traci.vehicle.getSpeed(raw_env.agent_id))
            else:
                cav_speeds.append(0.0)

            # Track head vehicle acceleration
            if raw_env.head_id in traci.vehicle.getIDList():
                head_accels.append(traci.vehicle.getAcceleration(raw_env.head_id))
            else:
                head_accels.append(0.0)

            # Track CAV spacing (gap to leader)
            for aid in raw_env.agent_ids:
                if aid in traci.vehicle.getIDList():
                    gap = raw_env._get_gap_to_leader(aid)
                    # Cap at ring length to avoid sentinel values
                    if raw_env.ring_length is not None:
                        gap = min(gap, raw_env.ring_length)
                    cav_spacings[aid].append(gap)
                else:
                    cav_spacings[aid].append(0.0)

            # Get action from agent (deterministic evaluation)
            # agent.act() returns tanh-squashed action in [-1, 1]
            a_tanh = agent.act(state=s, eval_mode=True)

            # Scale from [-1, 1] to action space bounds [-3, 3]
            a = a_tanh * max_accel

            # Track CAV commanded acceleration
            for i, aid in enumerate(raw_env.agent_ids):
                if len(a.shape) > 0 and a.shape[0] > 1:
                    cav_accels[aid].append(float(a[i]))
                else:
                    cav_accels[aid].append(float(a[0]) if hasattr(a, '__len__') else float(a))

            time.sleep(0.01)

            # Step environment
            s_next, r, done, truncated, _ = env.step(a)

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

            print("Plotting CAV spacing to: ppo_results/cav_spacing.png")
            plot_cav_spacing(
                spacings=cav_spacings,
                out_path="ppo_results/cav_spacing.png",
                title="PPO Evaluation: CAV Spacing",
            )

            print("Plotting accelerations to: ppo_results/vehicle_accelerations.png")
            plot_accelerations(
                head_accels=head_accels,
                cav_accels=cav_accels,
                out_path="ppo_results/vehicle_accelerations.png",
                title="PPO Evaluation: Vehicle Accelerations",
            )

    finally:
        env.close()

    return G


@click.command()
@click.option(
    "--agent-path",
    default="ppo_results/ppo_agent.pth",
    help="Path to saved PPO agent checkpoint.",
)
@click.option("--gui/--no-gui", default=True, help="Enable SUMO GUI.")
@click.option(
    "--scenario",
    type=click.Choice(["random", "emergency-braking"], case_sensitive=False),
    default="random",
    help="Head vehicle scenario.",
)
def main(agent_path: str, gui: bool, scenario: str):
    evaluate(agent_path=agent_path, gui=gui, scenario=scenario)


if __name__ == "__main__":
    main()
