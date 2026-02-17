# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for traffic flow optimization using SUMO (Simulation of Urban MObility). The project implements RL agents that control a single autonomous vehicle in a ring road scenario to maximize traffic flow by learning optimal speed control policies. Two RL approaches are implemented: tabular Q-learning and Deep Q-Networks (DQN).

## Environment Setup

**SUMO_HOME Required**: You must set the `SUMO_HOME` environment variable before running any simulations:
```bash
export SUMO_HOME="/path/to/your/sumo/installation"
```

**Dependencies**: Install using uv:
```bash
uv sync
```

**Python Version**: Requires Python 3.13+

## Common Commands

### Training

**Q-Learning (Tabular)**:
```bash
uv run rl_mixed_traffic/q_train.py
```
- Runs 250 episodes by default with GUI enabled
- Outputs: `output/q_table.pkl`, `output/returns.csv`, `output/returns.png`

**DQN (Deep Q-Network)**:
```bash
uv run rl_mixed_traffic/dqn_train.py
```
- Runs 350,000 total steps by default
- Outputs: `dqn_results/dqn_agent.pth`, `dqn_results/dqn_training_returns.png`, `dqn_results/dqn_training_losses.png`

### Evaluation

**Q-Learning**:
```bash
uv run rl_mixed_traffic/q_eval_policy.py
```
- Loads trained Q-table from `output/q_table.pkl`
- Runs with GUI enabled for visualization

**DQN**:
```bash
uv run rl_mixed_traffic/dqn_eval.py
```
- Loads trained DQN model from `dqn_results/dqn_agent.pth`
- Runs with GUI enabled for visualization

### Testing

```bash
uv run pytest
```

## Architecture

### Core Environment (`rl_mixed_traffic/env/ring_env.py`)

**RingRoadEnv**: Main Gymnasium environment that wraps SUMO TraCI for RL training.

- **Observation Space**: Normalized velocities and positions of all vehicles concatenated as `[v_norm_0..N, p_norm_0..N]` where values are in [0, 1]
  - Velocity normalized by `v_max` (30 m/s)
  - Position normalized by ring circumference
  - Fixed size based on `num_vehicles` parameter, padded if fewer vehicles exist

- **Action Space**: Continuous acceleration command in m/s² (bounded by `min_accel=-3.0`, `max_accel=3.0`)
  - Wrapped with `DiscretizeActionWrapper` to discretize into bins for Q-learning/DQN

- **Reward Function** (rl_mixed_traffic/env/ring_env.py:277): Multi-component reward balancing safety, efficiency, and comfort:
  1. **TTC Penalty** (`R_ttc`): Penalizes time-to-collision < 0.6s with lead vehicle
  2. **Headway Distance Penalty** (`r_d`): Penalizes gaps > 15m to encourage closer following
  3. **Jerk Penalty** (`R_jerk`): Penalizes rapid acceleration changes for comfort
  - Weights: 0.15 * R_ttc + 1.0 * r_d + 0.2 * R_jerk

- **Episode Termination**: Episodes end when:
  - Max steps reached (500s / 0.1s step = 5000 steps default)
  - Simulation completes (all vehicles removed)
  - Collision detected

### Agent Implementations

**Q-Learning** (`rl_mixed_traffic/q_agent.py`):
- Uses tabular Q-table (defaultdict with state tuples as keys)
- Epsilon-greedy exploration with linear decay
- Standard Q-learning update: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- Requires state/action discretization via `StateDiscretizer` and `DiscretizeActionWrapper`

**DQN** (`rl_mixed_traffic/dqn/dqn_agent.py`):
- Double DQN with target network (hard/soft updates)
- Replay buffer for experience replay
- Neural network: 2-layer MLP (rl_mixed_traffic/dqn/network.py)
- Uses SmoothL1Loss (Huber loss)
- Gradient clipping for stability

### Discretization System

**DiscretizeActionWrapper** (rl_mixed_traffic/env/discretizer.py:5): Converts continuous acceleration to discrete actions via linear binning.

**StateDiscretizer** (rl_mixed_traffic/env/discretizer.py:38): Converts continuous observations to discrete state tuples for tabular Q-learning. Uses uniform binning with optional position-heavy-tail mode.

### SUMO Integration

**Key utilities** (`rl_mixed_traffic/utils/sumo_utils.py`):
- `start_traci()`: Initializes TraCI connection with SUMO
- `get_vehicles_pos_speed()`: Retrieves all vehicle states sorted by position
- `compute_ring_length()`: Computes total ring road circumference

**Configuration**: Ring road defined in `configs/ring/`:
- `simulation.sumocfg`: Main SUMO config (references network and routes)
- `circle.net.xml`: Road network topology
- `circle.rou.xml`: Vehicle routes and insertion parameters

### Head Vehicle Behavior

The environment includes a "head vehicle" (`car0` by default) whose speed changes randomly every 15 seconds (rl_mixed_traffic/env/ring_env.py:105):
- Speed randomly sampled from [5.0, 20.0] m/s
- Creates dynamic traffic conditions for the agent to adapt to
- Agent vehicle learns to respond to these speed changes

## Key Implementation Details

**TraCI Speed Control Mode**: Agent vehicle uses speed mode `95` (0b1011111) to disable some safety checks and allow direct speed commands (rl_mixed_traffic/env/ring_env.py:185).

**Action Application**: Acceleration is integrated to velocity using `v_next = clip(v_now + a*dt, 0, v_max)` and applied via `setSpeed()` (non-smooth) rather than `slowDown()` for immediate response (rl_mixed_traffic/env/ring_env.py:198).

**Leader Detection Fallback**: When `getLeader()` fails (e.g., on ring roads), the environment falls back to distance-based detection to find the vehicle ahead (rl_mixed_traffic/env/ring_env.py:289).

**Jerk Calculation**: Jerk (rate of acceleration change) is tracked across steps as `(a_current - a_previous) / dt` and stored in `self.last_jerk` for reward computation (rl_mixed_traffic/env/ring_env.py:256).

## Output Structure

```
output/                    # Q-learning results
├── q_table.pkl           # Best Q-table snapshot
├── returns.csv           # Episode returns
└── returns.png           # Training curve

dqn_results/              # DQN results
├── dqn_agent.pth         # Trained model checkpoint
├── dqn_training_returns.png
└── dqn_training_losses.png
```

## Configuration Files

**SumoConfig** (rl_mixed_traffic/configs/sumo_config.py): SUMO simulation parameters
- `sumocfg_path`: Path to .sumocfg file
- `use_gui`: Enable/disable GUI
- `step_length`: Simulation time step (0.1s, must match SUMO config)

**DQNConfig** (rl_mixed_traffic/configs/dqn_config.py): DQN hyperparameters
- Learning rate, batch size, buffer size
- Epsilon schedule, discount factor
- Target network update frequency

## Modifying Behavior

**Change reward function**: Edit `RingRoadEnv.compute_reward()` in rl_mixed_traffic/env/ring_env.py:277

**Adjust observation space**: Modify `RingRoadEnv.get_state()` in rl_mixed_traffic/env/ring_env.py:127 and update `observation_space` property

**Change ring road layout**: Edit SUMO network files in `configs/ring/` (use SUMO netedit tool)

**Tune hyperparameters**:
- Q-learning: Edit agent initialization in q_train.py or q_eval_policy.py
- DQN: Modify `DQNConfig` dataclass in rl_mixed_traffic/configs/dqn_config.py
