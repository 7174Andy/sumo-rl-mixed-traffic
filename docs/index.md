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

**PPO:**

```bash
uv run rl_mixed_traffic/ppo_train.py
```

**Lagrangian PPO:**

```bash
uv run rl_mixed_traffic/lagrangian_ppo_train.py
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

**PPO:**

```bash
uv run rl_mixed_traffic/ppo_eval.py
```

**Lagrangian PPO:**

```bash
uv run rl_mixed_traffic/lagrangian_ppo_eval.py
```

## Design Notes

- [Reward Redesign](reward-redesign.md) — investigation into the SUMO safety override problem and the proposed non-negative reward solution
- [Lagrangian PPO](lagrangian-ppo.md) — safety-constrained training with a physics-based safety layer and adaptive Lagrange multiplier
- [Exploration Log: Lagrangian PPO Reward Tuning](exploration/lagrangian_ppo_reward_tuning.md) — detailed experiment log covering v_eq smoothing, weight tuning, and training diagnostics
