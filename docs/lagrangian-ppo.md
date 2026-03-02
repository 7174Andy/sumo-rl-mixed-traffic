# Lagrangian PPO: Safety-Constrained Training

This page documents the Lagrangian PPO variant — an attempt to solve the SUMO safety override problem by disabling SUMO's built-in safety checks and replacing them with a learned + hard-constraint approach.

## 1. Motivation

Standard PPO training relies on SUMO's speed mode 95, which keeps the Krauss car-following model active. As documented in [Reward Redesign](reward-redesign.md), this causes SUMO to silently override 99.1% of agent commands — the agent learns a degenerate "always accelerate" policy because SUMO handles all braking.

The idea behind Lagrangian PPO: **disable SUMO safety entirely** (speed mode 0) and have the agent learn safe behavior through two mechanisms:

1. **Hard constraint** — a physics-based safety layer that clips unsafe accelerations before they reach SUMO.
2. **Soft penalty** — a Lagrange multiplier that penalizes spacing violations in the reward signal.

The PPO algorithm itself is unchanged. Only the environment wrapper and reward augmentation differ.

## 2. Safety Layer

The safety layer (`rl_mixed_traffic/env/safety_layer.py`) is a pure-math filter with no SUMO dependency. It uses a one-step constant-acceleration prediction to check whether a requested acceleration would violate the minimum spacing threshold.

### Physics model

Predicted spacing after one timestep:

$$
\hat{s} = s + v_\text{rel} \cdot \Delta t - \tfrac{1}{2} a \cdot \Delta t^2
$$

where $s$ is the current bumper-to-bumper gap, $v_\text{rel} = v_\text{leader} - v_\text{ego}$ (negative means closing), and $a$ is the requested acceleration.

### Clipping rule

If $\hat{s} < s_\text{min}$, compute the maximum safe acceleration that keeps the predicted spacing exactly at the threshold:

$$
a_\text{safe} = \frac{2(s - s_\text{min} + v_\text{rel} \cdot \Delta t)}{\Delta t^2}
$$

The filtered acceleration is:

$$
a_\text{filtered} = \min(a,\; a_\text{safe})
$$

The filter only restricts — it never allows a more aggressive acceleration than the agent requested.

### Parameters

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| $s_\text{min}$ | 5.0 m | Minimum allowed bumper-to-bumper gap |
| $\Delta t$ | 0.1 s | Simulation timestep |

## 3. Lagrangian Reward Augmentation

The Lagrangian mechanism adds a soft spacing penalty to the base reward, with an adaptive multiplier that grows when violations occur.

### Constraint violation

At each step, the environment computes the spacing violation for each agent vehicle (`ring_env.py:335`):

$$
c_t = \max(s_\text{min} - s_t,\; 0)
$$

This is zero when the gap $s_t$ exceeds the threshold, and positive when the agent is too close to the leader.

### Augmented reward

The PPO agent sees an augmented reward instead of the base reward:

$$
r_\text{aug} = r_\text{base} - \lambda \cdot c_t
$$

The base reward is unchanged — the Lagrange multiplier $\lambda$ only adds an additional penalty proportional to the violation magnitude.

### Lambda update

At the end of each rollout (2048 steps), the multiplier is updated based on the mean violation across the rollout:

$$
\lambda \leftarrow \text{clip}\!\left(\lambda + \alpha_\lambda \cdot \bar{c},\; 0,\; \lambda_\text{max}\right)
$$

where $\bar{c}$ is the mean violation over the rollout. Lambda starts at 0 and grows only when violations occur. The clip keeps it bounded.

### Parameters

| Parameter | Config key | Default | Description |
| --------- | ---------- | ------- | ----------- |
| $\lambda_0$ | `lambda_init` | 0.0 | Initial multiplier value |
| $\alpha_\lambda$ | `lambda_lr` | 0.01 | Multiplier learning rate |
| $\lambda_\text{max}$ | `lambda_max` | 10.0 | Upper bound on multiplier |
| $s_\text{min}$ | `spacing_min` | 5.0 m | Minimum spacing threshold |

## 4. Training Configuration

### Standard PPO vs Lagrangian PPO

| Setting | Standard PPO | Lagrangian PPO |
| ------- | ------------ | -------------- |
| Total steps | 800,000 | 600,000 |
| Rollout steps | 2,048 | 2,048 |
| Learning rate | 1e-4 | 3e-4 |
| Clip epsilon | 0.15 | 0.2 |
| PPO epochs | 6 | 10 |
| SUMO speed mode | 95 (safety on) | 0 (safety off) |
| Safety layer | No | Yes |
| Lagrangian penalty | No | Yes |

The Lagrangian variant uses a higher learning rate and more PPO epochs per update to compensate for the harder optimization landscape when SUMO safety is disabled.

### Config file

Source: `rl_mixed_traffic/conf/lagrangian_ppo_train.yaml`

Key environment flags that differ from standard PPO:

```yaml
env:
  enable_safety_layer: true
  disable_sumo_safety: true
  spacing_min: 5.0

agent:
  enable_lagrangian: true
  lambda_init: 0.0
  lambda_lr: 0.01
  lambda_max: 10.0
```

### Running

```bash
uv run rl_mixed_traffic/lagrangian_ppo_train.py
```

Output goes to `lagrangian_ppo_results/`. The training loop tracks three additional metrics beyond standard PPO: `lambda` (current multiplier value), `mean_violation` (average spacing violation per rollout), and `safety_clip_rate` (fraction of steps where the safety layer clipped the action).

## 5. Results and Limitations

### Outcome

The agent still collides under speed mode 0, even with both the safety layer and Lagrangian penalty active.

### Why it fails

The safety layer uses a **one-step prediction horizon** ($\Delta t = 0.1\;\text{s}$). In fast-closing scenarios — where the lead vehicle brakes suddenly — 0.1 seconds of lookahead is not enough to prevent a collision. By the time the predicted spacing falls below $s_\text{min}$, the vehicles are already too close for a single-step correction to avoid contact.

The Lagrangian penalty cannot compensate for a fundamentally inadequate safety model. It can only shape the reward signal — it cannot override physics.

### Insight

This experiment led to the deeper realization documented in [Reward Redesign](reward-redesign.md): the root cause is not insufficient safety mechanisms but the **negative-only reward structure**. With rewards always $\leq 0$, the agent is incentivized to crash early (terminating the episode escapes accumulated negative reward). The fix is to transform the reward into a non-negative $[0, 1]$ range where collision termination is always the worst outcome.
