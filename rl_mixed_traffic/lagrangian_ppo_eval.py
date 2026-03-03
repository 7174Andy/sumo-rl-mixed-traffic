"""Evaluate a trained Lagrangian PPO agent on the ring road environment.

Mirrors ppo_eval.py but adds safety-layer and spacing-violation tracking.
"""

import click
import time
import traci
import numpy as np

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.env.scenario import make_head_controller
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.lagrangian_ppo_train import make_env
from rl_mixed_traffic.utils.plot_utils import (
    plot_vehicle_speeds,
    plot_cav_spacing,
    plot_accelerations,
)


class _ScenarioCfg:
    """Minimal config object for make_head_controller."""
    def __init__(self, type: str):
        self.type = type


def evaluate(agent_path: str, gui: bool = True, plot_speeds: bool = True, scenario: str = "random"):
    """Evaluate a trained Lagrangian PPO agent on the ring road environment.

    Args:
        agent_path: Path to saved agent checkpoint.
        gui: Whether to use SUMO GUI for visualization.
        plot_speeds: Whether to plot diagrams after evaluation.
        scenario: Scenario type ("random" or "emergency_braking").

    Returns:
        Total episode return.
    """
    scenario_cfg = _ScenarioCfg(type=scenario.replace("-", "_"))
    head_controller = make_head_controller(scenario_cfg)

    env = make_env(
        sumocfg_path="configs/ring/simulation.sumocfg",
        gui=gui,
        head_vehicle_controller=head_controller,
        enable_safety_layer=True,
        disable_sumo_safety=True,
        spacing_min=5.0,
    )
    raw_env = env.unwrapped

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
    max_accel = float(raw_env.max_accel)
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

    # Safety metrics
    safety_clips = []
    violations = []
    min_gap_observed = float("inf")

    out_dir = "lagrangian_ppo_results"

    print(f"Starting Lagrangian PPO evaluation with agent from: {agent_path}")
    print(f"Scenario: {scenario}")
    print(f"Safety layer: {raw_env.enable_safety_layer}, "
          f"SUMO safety disabled: {raw_env.disable_sumo_safety}")
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
            # Cap at ring_length (or 50 m fallback) to filter sentinel values
            # from leader-detection failures at episode boundaries
            max_gap = raw_env.ring_length if raw_env.ring_length else 50.0
            for aid in raw_env.agent_ids:
                if aid in traci.vehicle.getIDList():
                    gap = raw_env._get_gap_to_leader(aid)
                    gap = min(gap, max_gap)
                    cav_spacings[aid].append(gap)
                    if gap < min_gap_observed:
                        min_gap_observed = gap
                else:
                    cav_spacings[aid].append(0.0)

            # Get action from agent (deterministic evaluation)
            a_tanh = agent.act(state=s, eval_mode=True)

            # Scale from [-1, 1] to action space bounds
            a = a_tanh * max_accel

            # Track CAV commanded acceleration
            for i, aid in enumerate(raw_env.agent_ids):
                if len(a.shape) > 0 and a.shape[0] > 1:
                    cav_accels[aid].append(float(a[i]))
                else:
                    cav_accels[aid].append(float(a[0]) if hasattr(a, '__len__') else float(a))

            time.sleep(0.01)

            # Step environment
            s_next, r, done, truncated, info = env.step(a)

            # Track safety metrics
            safety_clipped = info.get("safety_clipped", False)
            safety_clips.append(1.0 if safety_clipped else 0.0)

            violation = raw_env.get_spacing_violation()
            violations.append(violation)

            s = s_next
            G += r
            steps += 1

            # Print progress every 100 steps
            if steps % 100 == 0:
                clip_rate = np.mean(safety_clips) * 100
                print(f"Step: {steps}, Return so far: {G:.2f}, "
                      f"Safety clip rate: {clip_rate:.1f}%")

        # Summary
        print("-" * 80)
        print("Evaluation completed!")
        print(f"Total Return: {G:.2f}")
        print(f"Total Steps: {steps}")
        print(f"Average Reward per Step: {G/steps:.4f}")
        print()
        print("Safety Metrics:")
        clip_rate = np.mean(safety_clips) * 100 if safety_clips else 0.0
        mean_violation = np.mean(violations) if violations else 0.0
        print(f"  Safety clip rate: {clip_rate:.2f}% ({int(sum(safety_clips))}/{steps} steps)")
        print(f"  Mean spacing violation: {mean_violation:.4f} m")
        print(f"  Min gap observed: {min_gap_observed:.2f} m")

        # Plots
        if plot_speeds and len(cav_speeds) > 0:
            print(f"\nPlotting vehicle speeds to: {out_dir}/vehicle_speeds.png")
            plot_vehicle_speeds(
                head_speeds=head_speeds,
                cav_speeds=cav_speeds,
                out_path=f"{out_dir}/vehicle_speeds.png",
                title="Lagrangian PPO Evaluation: Vehicle Speeds",
            )

            print(f"Plotting CAV spacing to: {out_dir}/cav_spacing.png")
            plot_cav_spacing(
                spacings=cav_spacings,
                out_path=f"{out_dir}/cav_spacing.png",
                title="Lagrangian PPO Evaluation: CAV Spacing",
            )

            print(f"Plotting accelerations to: {out_dir}/vehicle_accelerations.png")
            plot_accelerations(
                head_accels=head_accels,
                cav_accels=cav_accels,
                out_path=f"{out_dir}/vehicle_accelerations.png",
                title="Lagrangian PPO Evaluation: Vehicle Accelerations",
            )

    finally:
        env.close()

    return G


@click.command()
@click.option(
    "--agent-path",
    default="lagrangian_ppo_results/ppo_agent.pth",
    help="Path to saved Lagrangian PPO agent checkpoint.",
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
