# Algorithms

## 1. Problem Formulation

### 1.1 Environment: Ring Road Mixed Traffic
<!-- Describe the ring road scenario, the role of the autonomous vehicle, and the MDP formulation (state, action, reward, transition). -->

### 1.2 State Space
<!-- Detail the observation vector: normalized velocities and positions, padding scheme, and normalization constants. -->

### 1.3 Action Space
<!-- Describe the continuous acceleration range and the discretization strategy used for tabular/DQN methods. -->

### 1.4 Reward Function
<!-- Break down each reward component: TTC penalty, headway distance penalty, jerk penalty, and their weights. -->

---

## 2. Tabular Q-Learning

### 2.1 Overview
<!-- High-level description of tabular Q-learning and why it applies here. -->

### 2.2 State Discretization
<!-- Explain uniform binning, number of bins per dimension, and the position heavy-tail mode. -->

### 2.3 Update Rule
<!-- Present the Q-learning update equation and explain each term. -->

### 2.4 Exploration Strategy
<!-- Describe epsilon-greedy exploration with linear decay schedule. -->

### 2.5 Hyperparameters
<!-- Table of key hyperparameters: learning rate, discount factor, epsilon schedule, number of episodes. -->

---

## 3. Deep Q-Network (DQN)

### 3.1 Overview
<!-- High-level description of DQN and motivation for using it over tabular methods. -->

### 3.2 Network Architecture
<!-- Describe the 2-layer MLP: input size, hidden sizes, activation functions, output size. -->

### 3.3 Double DQN
<!-- Explain the Double DQN modification and how it reduces overestimation bias. -->

### 3.4 Experience Replay
<!-- Describe the replay buffer, sampling strategy, and buffer size. -->

### 3.5 Target Network
<!-- Explain hard vs. soft target network updates and update frequency. -->

### 3.6 Training Details
<!-- Loss function (Huber/SmoothL1), optimizer, gradient clipping, and training loop structure. -->

### 3.7 Hyperparameters
<!-- Table of key hyperparameters: learning rate, batch size, buffer size, epsilon schedule, target update frequency. -->

---

## 4. SUMO Integration Details

### 4.1 TraCI Control
<!-- Explain how the RL agent interfaces with SUMO via TraCI, speed mode flags, and action application. -->

### 4.2 Head Vehicle Behavior
<!-- Describe the stochastic speed changes of the lead vehicle and their role in creating dynamic traffic. -->

### 4.3 Leader Detection
<!-- Explain the primary getLeader() approach and the distance-based fallback for ring roads. -->

---

## References
<!-- List relevant papers and resources. -->
