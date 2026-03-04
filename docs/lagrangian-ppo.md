# Lagrangian PPO: Safety-Constrained Training

This page documents the Lagrangian PPO variant — an approach to solve the SUMO safety override problem by disabling SUMO's built-in safety checks and replacing them with a learned + hard-constraint approach.

**Prerequisites:** This document builds on the [Reward Redesign](reward-redesign.md) investigation (the non-negative reward transformation) and the [PPO Refactoring](ppo-refactor.md) lessons (tanh squashing, orthogonal init). The problem is formulated as a [Constrained MDP](problem.md#15-constrained-mdp-cmdp-formulation).

## 1. Motivation

Standard PPO training relies on SUMO's speed mode 95, which keeps the Krauss car-following model active. As documented in [Reward Redesign](reward-redesign.md), this causes SUMO to silently override 99.1% of agent commands — the agent learns a degenerate "always accelerate" policy because SUMO handles all braking.

The idea behind Lagrangian PPO: **disable SUMO safety entirely** (speed mode 0) and have the agent learn safe behavior through two mechanisms:

1. **Hard constraint** — a physics-based safety layer that clips unsafe accelerations before they reach SUMO.
2. **Soft penalty** — a Lagrange multiplier that penalizes spacing violations in the reward signal.

The PPO algorithm itself is unchanged. Only the environment wrapper and reward augmentation differ.

## 2. Safety Layer

The safety layer (`rl_mixed_traffic/env/safety_layer.py`) is a pure-math filter with no SUMO dependency. It uses a one-step constant-acceleration prediction to enforce two spacing constraints: a minimum gap to the physical leader and a maximum gap to the head vehicle.

### Physics model

Predicted spacing after one timestep:

$$
\hat{s} = s + v_\text{rel} \cdot \Delta t - \tfrac{1}{2} a \cdot \Delta t^2
$$

where $s$ is the current bumper-to-bumper gap, $v_\text{rel} = v_\text{leader} - v_\text{ego}$ (negative means closing), and $a$ is the requested acceleration.

### Constraint 1: $s_\text{min}$ — too close to leader (clip DOWN)

If $\hat{s} < s_\text{min}$, compute the maximum safe acceleration that keeps the predicted spacing exactly at the threshold:

$$
a_\text{safe\_max} = \frac{2(s - s_\text{min} + v_\text{rel} \cdot \Delta t)}{\Delta t^2}
$$

$$
a_\text{filtered} = \min(a,\; a_\text{safe\_max})
$$

This caps acceleration from above — prevents the CAV from getting too close.

### Constraint 2: $s_\text{max}$ — too far from head vehicle (clip UP)

If the predicted gap to the head vehicle exceeds $s_\text{max}$, compute the minimum acceleration needed to keep the gap at the threshold:

$$
a_\text{safe\_min} = \frac{2(s_\text{head} - s_\text{max} + v_\text{rel,head} \cdot \Delta t)}{\Delta t^2}
$$

$$
a_\text{filtered} = \max(a,\; a_\text{safe\_min})
$$

This caps acceleration from below — forces the CAV to speed up when it drifts too far from the head vehicle.

### Conflict resolution

When both constraints activate simultaneously (rare but possible in tight scenarios), $s_\text{min}$ takes priority. The filter applies $s_\text{max}$ first, then $s_\text{min}$ — so the final clipping always prevents collision even if it means violating $s_\text{max}$.

### Parameters

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| $s_\text{min}$ | 5.0 m | Minimum allowed bumper-to-bumper gap to leader |
| $s_\text{max}$ | 40.0 m | Maximum allowed gap to head vehicle ($2 \times s^*$) |
| $\Delta t$ | 0.1 s | Simulation timestep |

## 3. Lagrangian Reward Augmentation

The Lagrangian mechanism adds a soft spacing penalty to the base reward, with an adaptive multiplier that grows when violations occur.

### Constraint violation

At each step, the environment computes a normalised $s_\text{min}$ violation for each agent vehicle (`ring_env.py:get_spacing_violation`):

$$
c_t = \frac{\max(0,\; s_\text{min} - s_t)}{s_\text{min}}
$$

This is zero when the gap $s_t \geq s_\text{min}$, and increases toward 1.0 as the gap approaches zero. The division by $s_\text{min}$ keeps violations dimensionless and on a $[0, 1]$ scale, preventing the Lagrangian penalty from overwhelming the $[0, 1]$ reward signal.

The $s_\text{max}$ constraint is **not** included in the Lagrangian penalty — it is enforced only by the safety layer as a hard constraint (see Section 2).

### Augmented reward

The PPO agent sees an augmented reward instead of the base reward:

$$
r_\text{aug} = r_\text{base} - \lambda \cdot c_t
$$

The base reward is unchanged — the Lagrange multiplier $\lambda$ only adds an additional penalty proportional to the violation magnitude.

### Lambda update

At the end of each rollout (4096 steps), the multiplier is updated via dual gradient ascent with a violation tolerance:

$$
\lambda \leftarrow \text{clip}\!\left(\lambda + \alpha_\lambda \cdot (\bar{c} - d),\; 0,\; \lambda_\text{max}\right)
$$

where $\bar{c}$ is the mean normalised violation over the rollout and $d$ is the violation tolerance. When $\bar{c} < d$, the update is negative and $\lambda$ decreases — this prevents the multiplier from growing indefinitely when violations are near zero. Lambda starts at 1.0 to provide immediate penalty pressure from the first rollout.

### Parameters

| Parameter | Config key | Default | Description |
| --------- | ---------- | ------- | ----------- |
| $\lambda_0$ | `lambda_init` | 1.0 | Initial multiplier value |
| $\alpha_\lambda$ | `lambda_lr` | 0.05 | Multiplier learning rate |
| $\lambda_\text{max}$ | `lambda_max` | 10.0 | Upper bound on multiplier |
| $d$ | `violation_tolerance` | 0.1 | Threshold below which $\lambda$ decreases |
| $s_\text{min}$ | `spacing_min` | 5.0 m | Minimum spacing threshold (Lagrangian) |
| $s_\text{max}$ | `spacing_max` | 40.0 m | Maximum gap to head vehicle (safety layer only) |

## 4. Training Configuration

### Standard PPO vs Lagrangian PPO

| Setting | Standard PPO | Lagrangian PPO |
| ------- | ------------ | -------------- |
| Total steps | 800,000 | 1,500,000 |
| Rollout steps | 4,096 | 4,096 |
| Learning rate | 3e-4 | 3e-4 |
| Clip epsilon | 0.15 | 0.15 |
| PPO epochs | 6 | 6 |
| Value function clipping | No | Yes (`vf_clip_coef=0.2`) |
| SUMO speed mode | 95 (safety on) | 0 (safety off) |
| Safety layer | No | Yes ($s_\text{min}$ + $s_\text{max}$) |
| Lagrangian penalty | No | Yes ($s_\text{min}$ only) |
| Collision penalty | No | Yes ($r = -1.0$) |
| `weight_v` / `weight_s` / `weight_u` | 1.0 / 0.5 / 0.2 | 5.0 / 0.5 / 0.1 |
| `lambda_init` / `lambda_lr` | — | 1.0 / 0.05 |
| `spacing_max` | — | 40.0 m |

The Lagrangian variant uses stronger velocity tracking (`weight_v=5`) and lower control penalty (`weight_u=0.1`) to reduce catch-up lag when the head vehicle changes speed. Value function clipping stabilizes the critic during longer training runs.

### 4.1 Reward smoothing: time-averaged v_eq

The DeeP-LCC reward penalizes velocity deviation from an equilibrium speed $v_\text{eq}$. Initially, $v_\text{eq}$ was the head vehicle's instantaneous speed — but the head changes speed every 15 seconds, causing sudden reward spikes that prevented convergence.

The fix: $v_\text{eq}$ is now a **20-second rolling average** of the head vehicle's speed. This smooths over speed transitions and gives the agent a learnable, gradually shifting target. The desired spacing $s^*$ is fixed at 20 m (no longer dynamic).

See the [Exploration Log, Section 4](exploration/lagrangian_ppo_reward_tuning.md#4-fix-applied--smoothed-v_eq--fixed-s_star) for implementation details and before/after comparisons.

### Config file

Source: `rl_mixed_traffic/conf/lagrangian_ppo_train.yaml`

Key environment and agent flags that differ from standard PPO:

```yaml
env:
  enable_safety_layer: true
  disable_sumo_safety: true
  spacing_min: 5.0
  spacing_max: 40.0             # max gap to head vehicle (2 × s_star)
  weight_v: 5.0                 # stronger velocity tracking
  weight_s: 0.5
  weight_u: 0.1                 # less control penalty

agent:
  clip_vloss: true
  vf_clip_coef: 0.2
  enable_lagrangian: true
  lambda_init: 1.0              # start with penalty pressure
  lambda_lr: 0.05               # faster dual update
  lambda_max: 10.0
  violation_tolerance: 0.1      # lambda decreases when below this
```

### Running

```bash
uv run rl_mixed_traffic/lagrangian_ppo_train.py
```

Output goes to `lagrangian_ppo_results/`. The training loop tracks three additional metrics beyond standard PPO: `lambda` (current multiplier value), `mean_violation` (average spacing violation per rollout), and `safety_clip_rate` (fraction of steps where the safety layer clipped the action).

## 5. Results

### Current results

After all design iterations (see [Section 6](#6-design-iterations)), the agent achieves functional car-following with no crashes.

| Metric | Result |
| ------ | ------ |
| Speed tracking | CAV follows head within 1-2 m/s in steady state across speed transitions (5-15 m/s range) |
| Acceleration | Responsive (0 to +2.5 m/s²), clear modulation at speed transitions, mild chatter in steady state |
| Spacing | Tight 2-20 m for entire episode (~4700 steps), well within $s_\text{min}$-$s_\text{max}$ bounds |
| Returns | MA(10) rises to ~4000 by episode 50, plateaus at 3800-4200 over 380 episodes |
| Entropy | Healthy explore-then-exploit curve (1.80 peak, settles to 1.40) |
| Crashes | **None** — full-episode car-following achieved |
| Lambda | Decays from 1.0 to 0 — safety layer prevents violations from exceeding tolerance (see [analysis](exploration/lagrangian_ppo_reward_tuning.md#6-why-the-lagrangian-multiplier-never-engages)) |

### Training curves

![Lagrangian PPO Training Returns](assets/images/lagrangian_ppo_training_returns.png)

*Training over 380 episodes (1.5M steps). Early episodes are bimodal — the agent either follows successfully (~4000+) or loses the head vehicle (~0-1000). The MA(10) rises quickly to ~4000 by episode 50 and stabilizes in the 3800-4200 range. Occasional drops (episodes 100-140, ~295) reflect the stochastic head vehicle speed profile: some random speed sequences are inherently harder to track. After episode 150 the returns are consistently high with minimal variance, confirming stable convergence.*

![Lagrangian PPO Training Metrics](assets/images/lagrangian_ppo_training_metrics.png)

*Top-left: Policy loss starts high (~200) and drops to near zero by update 50 — rapid initial learning followed by stable optimization. Top-right: Value loss starts at ~100, decreases to ~10-20 by update 100 and stays stable — value function clipping (`vf_clip_coef=0.2`) prevents the oscillations seen in earlier runs (Iteration 2). Middle-left: Entropy rises from 1.45 to 1.80 during early exploration, then gradually declines to ~1.40 — a healthy explore-then-exploit trajectory. Middle-right: Clip fraction ranges 0.02-0.14 with some spikes, indicating the policy makes progressively larger updates. Bottom-left: Lambda starts at 1.0 (`lambda_init`) and linearly decays to 0 by ~update 300 — the safety layer keeps violations below the tolerance, so every dual update drives $\lambda$ down (see [exploration log analysis](exploration/lagrangian_ppo_reward_tuning.md#6-why-the-lagrangian-multiplier-never-engages)). Bottom-right: Mean violation (red) stays at 0.002-0.018 (normalised, near zero); safety clip rate (yellow) fluctuates at 2-16%, confirming the safety layer intervenes infrequently but non-trivially.*

### Evaluation

![Vehicle Speeds](assets/images/lagrangian_ppo_vehicle_speeds.png)

*The CAV (orange) tracks the head vehicle (blue) through ~15 random speed transitions over the full episode (~4700 steps). Tracking is tight across both upward and downward transitions in the 5-15 m/s range. Brief undershoot is visible on sharp downward transitions (e.g., step ~250 dips to ~1.5 m/s, step ~1500 dips to ~3 m/s) as the CAV reacts to sudden braking. In steady-state regions the CAV closely overlaps the head vehicle speed. The drop to 0 m/s at step ~4700 is the head vehicle exiting the simulation at episode end.*

![Vehicle Accelerations](assets/images/lagrangian_ppo_vehicle_accelerations.png)

*The head vehicle (blue) applies instantaneous speed changes via `setSpeed()`, producing sharp acceleration spikes (up to +2.5 m/s² and -3 m/s²). The CAV (orange) responds with accelerations in the 0 to +2.5 m/s² range, with clear spikes at speed transitions followed by settling to steady-state values (~1.0-1.5 m/s²). The CAV also brakes during downward transitions, though braking is less aggressive than the head vehicle's instantaneous deceleration. Mild acceleration chatter is visible in steady-state regions — the policy oscillates around the target rather than holding a perfectly constant acceleration.*

![CAV Spacing](assets/images/lagrangian_ppo_cav_spacing.png)

*Bumper-to-bumper gap stays in the 2-20 m range for the entire episode (steps 0-4500), well within the $s_\text{min} = 5\;\text{m}$ and $s_\text{max} = 40\;\text{m}$ bounds. The gap is tight — mostly 2-15 m — indicating the CAV follows closely. Small oscillations correlate with head speed transitions: the gap briefly widens when the head accelerates and narrows as the CAV catches up. The spike to ~950 m at step ~4700 is an artifact: the head vehicle is removed from the simulation at episode end, and the gap measurement wraps around the ring circumference. No spacing violations occur during normal operation.*

### Known limitations

- **Undershoot on sharp braking:** When the head vehicle drops speed sharply (e.g., 14 → 5 m/s), the CAV undershoots briefly (dipping to 1-3 m/s) before recovering. The 20 s $v_\text{eq}$ averaging window delays the target signal.
- **Steady-state acceleration chatter:** The CAV oscillates around the target acceleration in steady-state regions rather than holding a smooth constant value. This may reflect the stochastic Gaussian policy exploring around the mean.
- **Lagrangian multiplier inactive:** $\lambda$ decays from 1.0 to 0 because the safety layer prevents violations from exceeding the tolerance. The policy learns car-following purely from the DeeP-LCC reward, with no learned safety behavior from the Lagrangian. This is an open question — see the [exploration log](exploration/lagrangian_ppo_reward_tuning.md#6-why-the-lagrangian-multiplier-never-engages) for analysis and possible fixes.

### Pre-fix results (historical)

Before the smoothed $v_\text{eq}$ fix, the agent collided under speed mode 0. The root cause was the **negative-only reward structure** combined with instantaneous $v_\text{eq}$ causing reward whiplash. With rewards always $\leq 0$, the agent was incentivized to crash early (terminating the episode escapes accumulated negative reward).

This insight led to the reward redesign documented in [Reward Redesign](reward-redesign.md) and detailed in the [Exploration Log](exploration/lagrangian_ppo_reward_tuning.md).

## 6. Design Iterations

This section documents the major design changes made during Lagrangian PPO development, in chronological order. Each iteration describes the problem observed, the fix applied, and the rationale. For the raw experimental data (training curves, evaluation plots, metric tables), see the [Exploration Log](exploration/lagrangian_ppo_reward_tuning.md).

### Iteration 1: CAV slow-down exploit

**Problem:** The CAV learned to slow down and let the head vehicle (`car0`) lap it on the ring. Once `car0` wrapped around, the HDV behind the CAV became the physical leader at approximately $s^*$ gap. The CAV could then earn ~0.76/step doing nothing — a degenerate policy that optimizes spacing reward without actually following the head vehicle.

**Fix:** Added an $s_\text{max}$ constraint on the gap to the head vehicle. Initially implemented as a Lagrangian soft penalty (second cost term in the CMDP), but the penalty alone was insufficient — the agent continued exploiting even after 400k steps of training. The constraint was later moved to the safety layer as a hard constraint (see Iteration 3).

### Iteration 2: Value loss explosion (538M)

**Problem:** Raw spacing violations in meters (range 0–200 m on the ring) were used directly as the Lagrangian cost. With the base reward scaled to $[0, 1]$, the value function had to predict values mixing a $[0, 1]$ reward with a $[0, 200]$ penalty. The value loss exploded to 538M, destabilizing training completely.

**Fix:** Per-constraint normalisation in `get_spacing_violation()`. The $s_\text{min}$ violation is divided by $s_\text{min}$:

$$
c_1 = \max(0,\; s_\text{min} - s) / s_\text{min}
$$

This keeps violations in $[0, 1]$, matching the reward scale. The value function can now fit both reward and penalty on the same magnitude.

### Iteration 3: Safety layer $s_\text{max}$ hard constraint

**Problem:** The Lagrangian soft penalty for $s_\text{max}$ was too slow to prevent the CAV from overtaking the head vehicle. After 400k steps the agent was still exploiting the slow-down strategy (Iteration 1). The penalty grows $\lambda$ gradually, but by the time $\lambda$ is large enough the agent's policy is already entrenched.

**Fix:** Added $s_\text{max}$ to the safety layer as a hard constraint. When the predicted gap to the head vehicle exceeds $s_\text{max}$, the safety layer clips the acceleration **upward** (forces the CAV to accelerate). This is the opposite direction from the $s_\text{min}$ constraint, which clips **downward**.

The two constraints work in complementary directions:
- $s_\text{min}$: clips DOWN — don't crash into leader
- $s_\text{max}$: clips UP — don't drift from head vehicle
- If both conflict, $s_\text{min}$ wins (collision avoidance > formation keeping)

After this change, the $s_\text{max}$ Lagrangian penalty was removed — the constraint is now enforced only by the safety layer. The Lagrangian mechanism only penalises $s_\text{min}$ violations.

### Iteration 4: Collision penalty

**Problem:** No explicit negative signal existed for collisions. The agent could collide and receive a normal reward for that step, with the only consequence being episode termination (which can be beneficial under negative reward structures).

**Fix:** Override the reward to $-1.0$ when a collision is detected:

```python
if traci.simulation.getCollidingVehiclesNumber() > 0:
    reward = -1.0
```

This provides a clear negative signal regardless of the reward structure, making collisions unambiguously bad from the agent's perspective.
