
# Reinforcement Learning for Traffic Flow Control

This document explains the reinforcement learning setup used in this project to control traffic flow in a SUMO simulation.

## 1. Environment, States, and Actions

The core of the project is the `RingRoadEnv` class in `env.py`, which defines the environment for the reinforcement learning agent.

### 1.1. Simulation Environment

*   **Simulator:** The project uses [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) to simulate traffic.
*   **Scenario:** The specific scenario is a ring road, as defined in the `configs/ring/simulation.sumocfg` file.
*   **Controlled Vehicle:** The RL agent controls a single vehicle, identified by the `agent_id` (default is "car0").

### 1.2. State Space

The state space represents the information the agent uses to make decisions. It's a discretized representation of the agent's surroundings. The state is a tuple of three integer values:

1.  **Gap to Leader:** The distance to the vehicle directly in front of the agent.
    *   **Continuous Range:** 0 to 60 meters.
    *   **Discretization:** Binned into steps of 5 meters.

2.  **Ego Speed:** The agent's own speed.
    *   **Continuous Range:** 0 to 30 m/s.
    *   **Discretization:** Binned into steps of 2 m/s.

3.  **Relative Speed to Leader:** The difference between the agent's speed and the leader's speed.
    *   **Continuous Range:** -10 to 10 m/s.
    *   **Discretization:** Binned into steps of 2 m/s.

The `Discretizer` class in `env.py` handles the conversion of these continuous values into discrete integer indices.

### 1.3. Action Space

The action space defines the set of possible actions the agent can take at each step. The agent's goal is to choose the best action to maximize its future rewards.

*   **Action:** The agent's action is to adjust its commanded speed.
*   **Discrete Actions:** The action space is discrete and is determined by the `dv` and `action_k` parameters in the `RingRoadEnv`. The total number of actions is `2 * k + 1`.
*   **Example:** With the default `dv=0.4` and `action_k=3`, the possible changes to the commanded speed are `[-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2]` m/s.

## 2. Reward Function

The reward function (`compute_reward` in `env.py`) is crucial for guiding the agent's learning process. It provides feedback to the agent on its performance. The reward is a weighted sum of several components:

*   **Platoon Speed (Primary Goal):** The dominant part of the reward comes from the average speed of the agent and its immediate followers (up to 5). This encourages the agent to act in a way that promotes smooth flow for the vehicles behind it.
*   **Ego Speed:** The agent also receives a smaller reward based on its own speed, incentivizing it to maintain a reasonable velocity.
*   **Comfort Penalty:** A small penalty is subtracted for large differences between the commanded speed and the actual speed. This discourages abrupt acceleration or deceleration (jerk).
*   **Safety Penalty:** A penalty is applied under two conditions:
    1.  The gap to the following vehicle is too small (less than 5 meters).
    2.  The time-to-collision (TTC) with the follower is low, meaning the follower is closing in at a high speed.

The weights for these components are defined in the `compute_reward` function: `w_speed`, `w_comfort`, and `w_safety`.

## 3. Algorithm: Q-Learning

The project uses the **Q-learning** algorithm, a model-free reinforcement learning algorithm, to learn the optimal policy. The implementation is in the `QLearningAgent` class in `agent.py`.

### 3.1. Q-Table

*   The agent uses a **Q-table** to store the expected future rewards for taking a given action in a given state.
*   The Q-table is implemented as a Python `defaultdict`, where the keys are the state tuples and the values are NumPy arrays representing the Q-values for each possible action.

### 3.2. Learning Process

The agent learns through the following process, as seen in the `train.py` script:

1.  **Initialization:** The Q-table is initialized with all zeros.
2.  **Action Selection:** The agent uses an **epsilon-greedy policy** to select actions.
    *   With probability `epsilon`, it chooses a random action (exploration).
    *   With probability `1 - epsilon`, it chooses the action with the highest Q-value for the current state (exploitation).
    *   The value of `epsilon` decays over the course of training, shifting the agent from exploration to exploitation.
3.  **Q-Value Update:** After each action, the agent observes the reward and the next state, and updates the Q-value for the state-action pair it just took using the Bellman equation:

    ```
    Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))
    ```

    *   `alpha` is the learning rate.
    *   `gamma` is the discount factor.
    *   `r` is the reward.
    *   `s` is the current state, and `s'` is the next state.

### 3.3. Training

*   The `train` function in `train.py` runs the main training loop.
*   It iterates for a specified number of episodes, and in each episode, the agent interacts with the SUMO environment until the episode ends.
*   The cumulative reward (return) for each episode is tracked.
*   The Q-table that results in the highest return is saved to a file (`q_table.pkl`) for later use in `eval_policy.py`.
