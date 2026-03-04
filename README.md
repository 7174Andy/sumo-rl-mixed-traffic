# Traffic Simulation with SUMO and Reinforcement Learning

[![CI](https://github.com/7174Andy/sumo-rl-mixed-traffic/actions/workflows/ci.yml/badge.svg)](https://github.com/7174Andy/sumo-rl-mixed-traffic/actions/workflows/ci.yml)
[![Docs](https://github.com/7174Andy/sumo-rl-mixed-traffic/actions/workflows/docs.yml/badge.svg)](https://7174andy.github.io/sumo-rl-mixed-traffic/)

This project implements reinforcement learning agents that control one or more connected autonomous vehicles (CAVs) on a ring road to optimize traffic flow. The environment wraps SUMO (Simulation of Urban MObility) via TraCI and exposes a Gymnasium interface. Four RL approaches are implemented: tabular Q-learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Lagrangian PPO for constrained optimization with safety guarantees.

## Directory Structure

```
/
├── configs/
│   └── ring/
│       ├── circle.net.xml           # Ring road network topology
│       ├── circle.rou.xml           # Vehicle routes and insertion
│       └── simulation.sumocfg       # Main SUMO configuration
├── rl_mixed_traffic/
│   ├── agents/
│   │   ├── base_agent.py            # Abstract base class for all agents
│   │   ├── q_agent.py               # Q-learning agent
│   │   ├── dqn_agent.py             # DQN agent
│   │   └── ppo_agent.py             # PPO agent
│   ├── configs/
│   │   ├── sumo_config.py           # SUMO simulation parameters
│   │   ├── q_config.py              # Q-learning hyperparameters
│   │   ├── dqn_config.py            # DQN hyperparameters
│   │   └── ppo_config.py            # PPO hyperparameters (+ Lagrangian support)
│   ├── conf/                        # Hydra YAML configs
│   │   ├── q_train.yaml
│   │   ├── dqn_train.yaml
│   │   ├── ppo_train.yaml
│   │   └── lagrangian_ppo_train.yaml
│   ├── env/
│   │   ├── ring_env.py              # RingRoadEnv (single & multi-agent)
│   │   ├── discretizer.py           # State/action discretization
│   │   ├── head_vehicle_controller.py  # Head vehicle controllers
│   │   ├── safety_layer.py          # Hard-constraint safety layer
│   │   ├── scenario.py              # Head controller factory
│   │   ├── reward.py                # Reward utilities
│   │   └── wrappers.py              # Gymnasium wrappers
│   ├── dqn/
│   │   ├── network.py               # DQN neural network
│   │   └── replay_mem.py            # Experience replay buffer
│   ├── ppo/
│   │   ├── network.py               # PPO actor-critic network
│   │   └── rollout_buffer.py        # Rollout buffer
│   ├── utils/
│   │   ├── sumo_utils.py            # SUMO/TraCI utilities
│   │   └── plot_utils.py            # Plotting utilities
│   ├── scripts/
│   │   └── classic_controller.py    # Classical control baseline
│   ├── q_train.py                   # Q-learning training
│   ├── q_eval_policy.py             # Q-learning evaluation
│   ├── dqn_train.py                 # DQN training
│   ├── dqn_eval.py                  # DQN evaluation
│   ├── ppo_train.py                 # PPO training
│   ├── ppo_eval.py                  # PPO evaluation
│   ├── lagrangian_ppo_train.py      # Lagrangian PPO training
│   └── lagrangian_ppo_eval.py       # Lagrangian PPO evaluation
├── tests/
│   ├── test_compute_lcc_reward.py   # DeeP-LCC reward tests
│   ├── test_discretizer.py          # Discretizer tests
│   ├── test_safety_layer.py         # Safety layer tests
│   ├── test_head_vehicle_controller.py
│   ├── test_emergency_braking.py
│   ├── test_scenario.py
│   ├── test_network.py
│   ├── test_ppo_agent.py
│   └── test_wrappers.py
└── docs/                            # MkDocs documentation source
```

## Installation

1. **Install SUMO:**
   Download and install SUMO from the official website: [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/).

2. **Set SUMO_HOME:**
   You must set the `SUMO_HOME` environment variable to your SUMO installation path. For example:

   ```bash
   export SUMO_HOME="/path/to/your/sumo/installation"
   ```

3. **Install Python Dependencies:**
   This project requires Python 3.12+ and uses `uv` for dependency management. Install dependencies with:
   ```bash
   uv sync
   ```

## How to Use

### Training Agents

Four RL algorithms are available:

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

#### Lagrangian PPO (Constrained RL)

```bash
uv run rl_mixed_traffic/lagrangian_ppo_train.py
```

- PPO with Lagrangian relaxation for enforcing spacing constraints
- Configurable via Hydra: override parameters with `key=value` on the command line
- Outputs: `lagrangian_ppo_results/`

### Evaluating Trained Policies

```bash
uv run rl_mixed_traffic/q_eval_policy.py         # Q-learning
uv run rl_mixed_traffic/dqn_eval.py               # DQN
uv run rl_mixed_traffic/ppo_eval.py               # PPO
uv run rl_mixed_traffic/lagrangian_ppo_eval.py    # Lagrangian PPO
```

Each script loads the corresponding trained model and runs with GUI enabled.

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

Each RL algorithm has a dataclass config in `rl_mixed_traffic/configs/` and Hydra YAML overrides in `rl_mixed_traffic/conf/`:

- `sumo_config.py`: SUMO simulation parameters (step length, GUI settings)
- `q_config.py`: Q-learning hyperparameters (learning rate, epsilon decay, discount factor)
- `dqn_config.py`: DQN hyperparameters (learning rate, batch size, buffer size, target network update)
- `ppo_config.py`: PPO hyperparameters (learning rate, clip ratio, GAE, entropy coefficient, Lagrangian multiplier)

## Environment Details

### Observation Space

Normalized velocities and positions of all vehicles concatenated as `[v_norm_0..N, p_norm_0..N]` where values are in [0, 1]. In multi-agent mode, each agent receives the global state augmented with its normalized agent index.

### Action Space

- **Q-learning / DQN**: Discretized acceleration commands via `DiscretizeActionWrapper`
- **PPO / Lagrangian PPO**: Continuous acceleration in m/s² (bounded by [-3.0, 3.0])

### Reward Function (DeeP-LCC)

The reward is based on the DeeP-LCC formulation, transforming a quadratic cost into a bounded [0, 1] reward:

```
r = max(J_max - J, 0) / J_max
```

where the cost J combines three components:

1. **Velocity error**: `weight_v * sum((v_i - v_star)^2)` for all non-head vehicles
2. **Spacing error**: `weight_s * (gap - s_star)^2` between the CAV and its leader
3. **Control penalty**: `weight_u * accel^2` to discourage aggressive inputs

At equilibrium (all vehicles at `v_star`, gap = `s_star`, zero acceleration), the reward is 1.0.

### Safety Layer

An optional physics-based safety layer clips unsafe accelerations to enforce hard constraints:

- **s_min constraint**: Prevents the CAV from getting too close to its leader
- **s_max constraint**: Prevents the CAV from falling too far behind the head vehicle

Enable with `enable_safety_layer=True` when constructing the environment.

### Head Vehicle Controllers

The head vehicle (`car0`) behavior is configurable via scenarios:

- **Random**: Speed changes randomly every 15 seconds (default)
- **Emergency Braking**: Cruise → brake → hold → recover cycle for testing safety
- **EUDC**: European Urban Driving Cycle for realistic speed profiles

### Multi-Agent Support

`RingRoadEnv` supports controlling multiple CAVs:

- **Single-agent** (`num_agents=1`): Standard Gymnasium interface
- **Multi-agent** (`num_agents>1`): Returns observation dictionaries and shared reward

## Documentation

Full documentation is available at [7174andy.github.io/sumo-rl-mixed-traffic](https://7174andy.github.io/sumo-rl-mixed-traffic/).
