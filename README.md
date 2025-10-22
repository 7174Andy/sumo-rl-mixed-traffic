# Traffic Simulation with SUMO and Reinforcement Learning

This repository contains a traffic simulation environment using SUMO (Simulation of Urban MObility) integrated with reinforcement learning techniques. The goal is to optimize traffic flow and reduce congestion through intelligent traffic signal control. This implementation focuses on a single autonomous vehicle in a ring road scenario, learning to adjust its speed to maximize the flow of the surrounding traffic.

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
│   ├───agent.py          # Q-learning agent implementation
│   ├───config.py         # Configuration for SUMO simulation
│   ├───env.py            # OpenAI Gym environment for the SUMO ring road
│   ├───eval_policy.py    # Script to evaluate a trained policy
│   ├───train.py          # Main script to train the agent
│   ├───output/           # Directory for training outputs (Q-table, plots)
│   │   ├───q_table.pkl
│   │   ├───returns.csv
│   │   └───returns.png
│   └───utils/
│       ├───plot_utils.py # Utilities for plotting results
│       └───sumo_utils.py # Utilities for interacting with SUMO/TraCI
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
    This project requires Python 3.8+ and the following libraries. You can install them using pip:
    ```bash
    pip install numpy gymnasium matplotlib
    ```

## How to Use

### Training the Agent

To train the reinforcement learning agent, run the `train.py` script. You can specify the number of episodes and whether to render the GUI.

```bash
python rl_mixed_traffic/train.py
```

By default, this will run for 250 episodes with the GUI enabled. The script will save the best Q-table to `rl_mixed_traffic/output/q_table.pkl`, the episode returns to `rl_mixed_traffic/output/returns.csv`, and a plot of the returns to `rl_mixed_traffic/output/returns.png`.

### Evaluating a Trained Policy

Once you have a trained Q-table, you can evaluate its performance using the `eval_policy.py` script.

```bash
python rl_mixed_traffic/eval_policy.py
```

This script loads the Q-table from `rl_mixed_traffic/output/q_table.pkl` and runs the simulation with the learned policy, rendering the GUI so you can observe the agent's behavior.

## Configuration

The SUMO simulation settings are located in the `configs/` directory. The main configuration for the ring road scenario is `configs/ring/simulation.sumocfg`. You can modify the network, routes, and simulation parameters in the corresponding `.xml` files.