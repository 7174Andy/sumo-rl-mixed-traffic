# SUMO RL Mixed Traffic

Reinforcement learning for traffic flow optimization using SUMO (Simulation of Urban Mobility).

## Overview

This project implements RL agents that control a single Connected Autonomous Vehicle (CAV) on a ring road shared with human-driven vehicles. The CAV must learn a speed-control policy that safely follows a lead vehicle through random speed transitions, maintaining safe spacing while minimizing velocity tracking error and control effort.

Three RL algorithms were developed iteratively — tabular Q-learning, DQN, and PPO — each addressing limitations of the previous approach. The final system uses **Lagrangian PPO** with a physics-based safety layer and constrained optimization to achieve crash-free car-following without relying on SUMO's built-in safety model.

## Design Journey

The documents below are ordered to tell the story of the project's evolution. A reviewer unfamiliar with the project should read them in this order.

### 1. Problem Formulation

**[Problem Formulation](problem.md)** — defines the ring road environment, state/action/reward spaces, and the MDP. Also contains the **Constrained MDP (CMDP)** reformulation with $s_\text{min}$ / $s_\text{max}$ spacing constraints and Lagrangian relaxation (Section 1.5).

### 2. Algorithm Progression

**[Algorithms](algorithms.md)** — reference documentation for the three RL algorithms tried:

| Algorithm | Action Space | Key Limitation |
| --------- | ------------ | -------------- |
| Q-Learning | Discrete (20 bins) | Curse of dimensionality; can't scale to more vehicles |
| DQN | Discrete (21 bins) | Discretization limits control smoothness |
| PPO | Continuous (Gaussian) | Requires careful reward design to learn safely |

Each algorithm's failure mode motivated the next approach.

### 3. The SUMO Safety Override Problem

**[Reward Redesign](reward-redesign.md)** — the pivotal investigation. All three algorithms independently converged to "always accelerate" because SUMO's Krauss model silently overrides 99.1% of agent commands. The document traces two failed fixes (speed mode 0 + AEB, Lagrangian PPO with safety layer) before identifying the root cause: **negative-only rewards incentivize the agent to crash early** to escape accumulated penalty. The solution: transform the DeeP-LCC cost into a non-negative $[0, 1]$ reward following EnduRL's approach.

### 4. PPO Implementation Lessons

**[PPO Refactoring](ppo-refactor.md)** — documents an attempt to modernize the PPO code using CleanRL conventions (raw Gaussian policy, `NormalizeReward` wrapper). The refactoring caused training instability due to the environment's discontinuous reward landscape ($-100$ safety penalty). Key lesson: **tanh action squashing provides critical implicit regularization** in environments with hard safety constraints. The final implementation uses a hybrid approach keeping CleanRL's initialization and activations while restoring tanh squashing.

### 5. Lagrangian PPO (Main Design Document)

**[Lagrangian PPO](lagrangian-ppo.md)** — the current approach. Disables SUMO safety entirely (speed mode 0) and replaces it with:

- **Hard constraint**: a physics-based safety layer that clips accelerations to maintain $s_\text{min}$ (don't crash) and $s_\text{max}$ (don't drift)
- **Soft penalty**: a Lagrange multiplier that penalizes $s_\text{min}$ violations in the reward signal

Contains the full algorithm description, training configuration, results, and **Section 6: Design Iterations** — a chronological record of the 4 major design changes made during development (CAV slow-down exploit, value loss explosion, $s_\text{max}$ hard constraint, collision penalty).

### 6. Experiment Log

**[Exploration Log: Lagrangian PPO Reward Tuning](exploration/lagrangian_ppo_reward_tuning.md)** — the raw experimental record. Documents pre-fix vs post-fix training runs, the smoothed $v_\text{eq}$ fix, and an analysis of why the Lagrangian multiplier never engages (the safety layer prevents violations, keeping $\lambda$ at zero).

## Getting Started

### Prerequisites

- Python 3.13+
- [SUMO](https://sumo.dlr.de/) installed with `SUMO_HOME` environment variable set

### Installation

```bash
uv sync
```

### Training

```bash
# Q-Learning (tabular)
uv run rl_mixed_traffic/q_train.py

# DQN
uv run rl_mixed_traffic/dqn_train.py

# PPO (standard)
uv run rl_mixed_traffic/ppo_train.py

# Lagrangian PPO (current approach)
uv run rl_mixed_traffic/lagrangian_ppo_train.py
```

### Evaluation

```bash
uv run rl_mixed_traffic/q_eval_policy.py      # Q-Learning
uv run rl_mixed_traffic/dqn_eval.py            # DQN
uv run rl_mixed_traffic/ppo_eval.py            # PPO
uv run rl_mixed_traffic/lagrangian_ppo_eval.py # Lagrangian PPO
```
