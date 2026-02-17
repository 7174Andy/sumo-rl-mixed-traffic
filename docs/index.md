# SUMO RL Mixed Traffic

Reinforcement learning for traffic flow optimization using SUMO (Simulation of Urban Mobility).

## Overview

This project implements RL agents that control a single autonomous vehicle in a ring road scenario to maximize traffic flow by learning optimal speed control policies.

## Getting Started

### Prerequisites

- Python 3.13+
- [SUMO](https://sumo.dlr.de/) installed with `SUMO_HOME` environment variable set

### Installation

```bash
uv sync
```

### Training

**Q-Learning:**

```bash
uv run rl_mixed_traffic/q_train.py
```

**DQN:**

```bash
uv run rl_mixed_traffic/dqn_train.py
```

### Evaluation

**Q-Learning:**

```bash
uv run rl_mixed_traffic/q_eval_policy.py
```

**DQN:**

```bash
uv run rl_mixed_traffic/dqn_eval.py
```
