# Traffic Simulation with SUMO and Reinforcement Learning

This repository contains a traffic simulation environment using SUMO (Simulation of Urban MObility) integrated with reinforcement learning techniques. The project implements RL agents that control a single autonomous vehicle in a ring road scenario to maximize traffic flow by learning optimal speed control policies. Three RL approaches are implemented: tabular Q-learning, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO).

## Directory Structure

```
/
├───.gitignore
├───.python-version
├───pyproject.toml
├───README.md
├───configs/
│   └───ring/
│       ├───circle.net.xml      # Network file for the ring road
│       ├───circle.rou.xml      # Route file for the vehicles
│       └───simulation.sumocfg  # Main SUMO configuration for the ring scenario
├───rl_mixed_traffic/
│   ├───agents/
│   │   ├───q_agent.py          # Q-learning agent implementation
│   │   ├───dqn_agent.py        # DQN agent implementation
│   │   └───ppo_agent.py        # PPO agent implementation
│   ├───configs/
│   │   ├───sumo_config.py      # SUMO simulation parameters
│   │   ├───q_config.py         # Q-learning hyperparameters
│   │   ├───dqn_config.py       # DQN hyperparameters
│   │   └───ppo_config.py       # PPO hyperparameters
│   ├───env/
│   │   ├───ring_env.py         # Gymnasium environment for the SUMO ring road
│   │   └───discretizer.py      # State/action discretization for tabular methods
│   ├───dqn/
│   │   ├───network.py          # DQN neural network architecture
│   │   └───replay_mem.py       # Experience replay buffer
│   ├───ppo/
│   │   ├───network.py          # PPO actor-critic network architecture
│   │   └───rollout_buffer.py   # PPO rollout buffer for trajectory collection
│   ├───utils/
│   │   ├───plot_utils.py       # Utilities for plotting results
│   │   └───sumo_utils.py       # Utilities for interacting with SUMO/TraCI
│   ├───q_train.py              # Q-learning training script
│   ├───q_eval_policy.py        # Q-learning evaluation script
│   ├───dqn_train.py            # DQN training script
│   ├───dqn_eval.py             # DQN evaluation script
│   ├───ppo_train.py            # PPO training script
│   └───ppo_eval.py             # PPO evaluation script
├───output/                     # Q-learning results
│   ├───q_table.pkl
│   ├───returns.csv
│   └───returns.png
├───dqn_results/                # DQN results
│   ├───dqn_agent.pth
│   ├───dqn_training_returns.png
│   └───dqn_training_losses.png
├───ppo_results/                # PPO results
│   ├───ppo_agent.pth
│   └───ppo_training_returns.png
└───tests/
```

## Installation

1.  **Install SUMO:**
    Download and install SUMO from the official website: [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/).

2.  **Set SUMO_HOME:**
    You must set the `SUMO_HOME` environment variable to your SUMO installation path. For example:

    ```bash
    export SUMO_HOME="/path/to/your/sumo/installation"
    ```

3.  **Install Python Dependencies:**
    This project requires Python 3.13+ and uses `uv` for dependency management. Install dependencies with:
    ```bash
    uv sync
    ```

## How to Use

### Training Agents

Three RL algorithms are available for training:

#### Q-Learning (Tabular)

```bash
uv run rl_mixed_traffic/q_train.py
```

- Runs 250 episodes by default with GUI enabled
- Uses tabular Q-learning with discretized state/action spaces
- Outputs: `output/q_table.pkl`, `output/returns.csv`, `output/returns.png`

#### DQN (Deep Q-Network)

```bash
uv run rl_mixed_traffic/dqn_train.py
```

- Runs 350,000 total steps by default
- Uses deep neural network with experience replay and target network
- Outputs: `dqn_results/dqn_agent.pth`, `dqn_results/dqn_training_returns.png`, `dqn_results/dqn_training_losses.png`

#### PPO (Proximal Policy Optimization)

```bash
uv run rl_mixed_traffic/ppo_train.py
```

- Runs 500,000 total steps by default
- Uses continuous action space with Gaussian policy
- Outputs: `ppo_results/ppo_agent.pth`, `ppo_results/ppo_training_returns.png`, `ppo_results/ppo_training_metrics.png`

### Evaluating Trained Policies

Once you have a trained agent, you can evaluate its performance using the corresponding evaluation script:

#### Q-Learning

```bash
uv run rl_mixed_traffic/q_eval_policy.py
```

Loads trained Q-table from `output/q_table.pkl` and runs with GUI enabled.

#### DQN

```bash
uv run rl_mixed_traffic/dqn_eval.py
```

Loads trained DQN model from `dqn_results/dqn_agent.pth` and runs with GUI enabled.

#### PPO

```bash
uv run rl_mixed_traffic/ppo_eval.py
```

Loads trained PPO model from `ppo_results/ppo_agent.pth` and runs with GUI enabled.

### Running Tests

```bash
uv run pytest
```

## Configuration

### SUMO Configuration

The SUMO simulation settings are in the `configs/ring/` directory:

- `simulation.sumocfg`: Main SUMO configuration file
- `circle.net.xml`: Ring road network topology
- `circle.rou.xml`: Vehicle routes and insertion parameters

### Algorithm Configuration

Each RL algorithm has a dedicated configuration file in `rl_mixed_traffic/configs/`:

- `sumo_config.py`: SUMO simulation parameters (step length, GUI settings)
- `q_config.py`: Q-learning hyperparameters (learning rate, epsilon decay, discount factor)
- `dqn_config.py`: DQN hyperparameters (learning rate, batch size, buffer size, target network update frequency)
- `ppo_config.py`: PPO hyperparameters (learning rate, clip ratio, GAE parameters, entropy coefficient)

## Environment Details

**Observation Space**: Normalized velocities and positions of all vehicles concatenated as `[v_norm_0..N, p_norm_0..N]` where values are in [0, 1].

**Action Space**:

- Q-learning/DQN: Discretized acceleration commands
- PPO: Continuous acceleration in m/s² (bounded by [-3.0, 3.0])

**Reward Function**: Multi-component reward balancing safety, efficiency, and comfort:

1. **TTC Penalty**: Penalizes time-to-collision < 0.6s with lead vehicle
2. **Headway Distance Penalty**: Penalizes gaps > 15m to encourage closer following
3. **Jerk Penalty**: Penalizes rapid acceleration changes for comfort

See [CLAUDE.md](CLAUDE.md) for detailed implementation details and architecture information.
