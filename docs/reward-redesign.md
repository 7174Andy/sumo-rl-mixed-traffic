# Reward Redesign: The SUMO Safety Override Problem

This page documents an investigation into why the PPO agent learns a degenerate "always accelerate" policy, the approaches tried to fix it, and the proposed solution based on non-negative reward shaping.

## 1. The Problem: SUMO Safety Override

### Speed mode 95

The agent vehicle uses speed mode 95 (`0b1011111`), which keeps most of SUMO's internal safety checks active:

| Bit | Check | Active? |
| --- | ----- | ------- |
| 0 | Safe speed for leader (collision avoidance) | Yes |
| 1 | Safe speed for right leader | Yes |
| 2 | Safe speed for max braking | Yes |
| 3 | Right-of-way at intersections | Yes |
| 4 | Brake for red lights | Yes |
| 5 | Respect max acceleration | No |
| 6 | Respect max deceleration | Yes |

The critical bit here is **bit 0** — SUMO's Krauss car-following model silently overrides any `setSpeed()` command that would cause a collision with the leader vehicle.

### Instrumented evaluation

An instrumented evaluation script recorded both the speed commanded via `setSpeed()` and the actual speed returned by `traci.vehicle.getSpeed()` after each simulation step. The results were stark:

```
RESULTS: 4829/4872 steps had SUMO override (99.1%)
Max speed override: 13.0144 m/s
Mean speed override (when overridden): 0.1699 m/s
Max commanded accel: 1.6714 m/s²
Min commanded accel: -0.4560 m/s²
Max actual accel: 1.6219 m/s²
Min actual accel: -2.3625 m/s²
```

**99.1% of agent commands were overridden by SUMO.** The scatter plot of commanded vs. actual acceleration showed nearly all commanded accelerations clustered at $x \approx +1.5\;\text{m/s}^2$, while actual accelerations ranged from $-2.5$ to $+1.6\;\text{m/s}^2$ — a vertical spread at that single x-value. The agent had learned a trivial "always accelerate at $+1.5\;\text{m/s}^2$" strategy, and SUMO's Krauss model was doing the actual driving.

## 2. Why This Happens

The root cause is that the reward signal reflects **SUMO's behavior**, not the **agent's actions**:

1. Agent commands "accelerate" $\rightarrow$ SUMO overrides when unsafe $\rightarrow$ traffic flows reasonably $\rightarrow$ decent reward.
2. Agent commands "brake" $\rightarrow$ SUMO allows it $\rightarrow$ unnecessary slowdown $\rightarrow$ spacing error increases $\rightarrow$ worse reward.

The optimal strategy under SUMO safety is to always command maximum acceleration and let SUMO handle all braking. This is analogous to a student who learns to guess "C" on every multiple-choice question because the teacher curves the grades.

The following agents all independently converged to some variant of this policy (Q-learning, DQN, PPO), confirming it is a property of the environment rather than a specific algorithm.

## 3. Approach 1: Disable SUMO Safety (Speed Mode 0)

The first approach was to make SUMO execute agent commands faithfully by setting speed mode to 0 ("aggressive"), which disables all safe-speed checks. To compensate, a three-layer safety hierarchy was added:

| Layer | Mechanism | Gap threshold | Effect |
| ----- | --------- | ------------- | ------ |
| 1 (soft) | Proximity penalty | $< 5.0\;\text{m}$ | Continuous reward shaping: $R_\text{prox} = -10 \cdot (1 - g/s_\text{min})^2$ |
| 2 (hard) | AEB filter | $< 2.5\;\text{m}$ | Override to emergency brake ($-3.0\;\text{m/s}^2$) |
| 3 (terminal) | Collision penalty | $0\;\text{m}$ (contact) | $-1.0$ terminal reward |

**Result**: the agent now collides. The AEB filter uses a one-step physics prediction ($\hat{s} = s + v_\text{rel} \cdot \Delta t - \tfrac{1}{2} a \cdot \Delta t^2$ with $\Delta t = 0.1\;\text{s}$) which is too short-horizon to catch fast-closing scenarios.

## 4. Approach 2: Lagrangian PPO with Safety Layer

The second approach added a physics-based safety layer that clips unsafe accelerations before they reach SUMO:

$$
\hat{s} = s + v_\text{rel} \cdot \Delta t - \tfrac{1}{2} a \cdot \Delta t^2
$$

If $\hat{s} < s_\text{min}$, compute the maximum safe acceleration:

$$
a_\text{safe} = \frac{2(s - s_\text{min} + v_\text{rel} \cdot \Delta t)}{\Delta t^2}
$$

Then $a_\text{filtered} = \min(a, a_\text{safe})$. A Lagrange multiplier was added to penalize spacing violations during training (`lambda_init=0.0`, `lambda_lr=0.01`, `lambda_max=10.0`).

**Result**: the agent still collides. The same one-step prediction horizon is insufficient, and the Lagrangian penalty cannot compensate for a fundamentally inadequate safety model.

## 5. The Deeper Problem: Negative-Only Rewards

Both approaches above treated the safety problem as the root cause, but the real issue is the **reward structure**. The DeeP-LCC reward function (at `ring_env.py:615`) computes:

$$
R = \frac{R_v + R_s + R_u}{100}
$$

where:

- $R_v = -w_v \sum_{i \neq \text{head}} (v_i - v^*)^2$ (velocity tracking error)
- $R_s = -w_s \cdot \text{clip}(d_\text{gap} - s^*, -20, 20)^2$ (spacing error, CAV only)
- $R_u = -w_u \cdot a_\text{prev}^2$ (control effort, CAV only)

All three components are **always $\leq 0$**. The reward is always non-positive. Default weights: $w_v = 1$, $w_s = 0.5$, $w_u = 0.1$, with $v^* = 15.0\;\text{m/s}$.

### With speed mode 95 (SUMO safety on)

The agent cannot earn positive reward no matter how well it drives. Both "drive well" and "always accelerate while SUMO drives" produce similar reward, so there is no incentive to learn a good policy.

### With speed mode 0 (SUMO safety off)

Per-step reward $\approx -0.5$ (after the $/100$ scaling). An episode of 5,000 steps accumulates $\approx -2,500$ in negative reward. The collision penalty is only $-1.0$. **Crashing at step 100 saves $\approx 2,400$ in avoided negative rewards.** The agent learns to collide early because episode termination is an escape from the stream of negative rewards.

No fixed collision penalty can robustly fix this — the required magnitude depends on episode length and reward scale, creating a fragile tuning problem.

A truncation-based fix was attempted (returning collision as `truncated=True` so PPO bootstraps with $V(s)$ instead of 0), but this adds complexity and does not address the fundamental incentive problem.

## 6. Reference: EnduRL's Approach

[EnduRL](https://github.com/poudel-bibek/EnduRL) solves the same ring-road CAV problem with a different reward design. Key differences:

| Aspect | EnduRL | This project (DeeP-LCC cost) |
| ------ | ------ | ----------------------------- |
| Speed mode | 25 (SUMO safety on) | 95 (SUMO safety on) |
| Reward sign | Always $\geq 0$ | Always $\leq 0$ |
| Collision handling | Reward $= 0$ (worst) | Terminal penalty $-1.0$ |
| Accel bounds | $\pm 1.0\;\text{m/s}^2$ | $\pm 3.0\;\text{m/s}^2$ |
| Safety filter | None (SUMO's Krauss model) | AEB / safety layer (when mode 0) |

EnduRL's reward function (`flow/core/rewards.py:desired_velocity`):

```python
def desired_velocity(env, fail=False, edge_list=None):
    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.
    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.linalg.norm([target_vel] * num_vehicles)
    cost = np.linalg.norm(vel - target_vel)
    eps = np.finfo(np.float32).eps
    return max(max_cost - cost, 0) / (max_cost + eps)
```

$$
r = \frac{\max(J_\text{max} - J,\; 0)}{J_\text{max} + \varepsilon}
$$

This produces a reward in $[0, 1]$. On collision, the function returns $0$, which is always the worst outcome. The agent has incentive to maximize positive reward, and crashing loses all future positive rewards. There is no "escape from negative rewards" problem.

From the EnduRL docstring:

> "The function is formulated as a mapping $r: S \times A \to \mathbb{R}_{\geq 0}$ ... naturally punishing early termination."

## 7. Proposed Solution: Non-Negative DeeP-LCC Reward

The fix is to keep SUMO safety on (speed mode 95) and transform the DeeP-LCC cost into a non-negative reward, following EnduRL's principle.

### Transformation

Define $J = |R_v| + |R_s| + |R_u| \geq 0$ as the total DeeP-LCC cost (sum of absolute penalty terms, **before** the $/100$ scaling). Then:

$$
r = \frac{\max(J_\text{max} - J,\; 0)}{J_\text{max}}
$$

- At equilibrium ($J = 0$): $r = 1.0$ (best possible reward).
- At worst case ($J = J_\text{max}$): $r = 0.0$.
- On collision: $r = 0.0$ (same as EnduRL).

The reward is always in $[0, 1]$, which is already PPO-friendly without any additional scaling.

### Computing $J_\text{max}$

$J_\text{max}$ is the worst-case cost, computed once in the environment constructor from the maximum possible error in each component:

$$
J_\text{max} = J_v^\text{max} + J_s^\text{max} + J_u^\text{max}
$$

where:

- $J_v^\text{max} = w_v \cdot \max(v^*, v_\text{max} - v^*)^2 \cdot (N - 1)$
  - Worst-case velocity error per vehicle: $\max(v^*, 30 - v^*) = 15.0\;\text{m/s}$ (for $v^* = 15$)
  - Summed over all $N - 1$ non-head vehicles
- $J_s^\text{max} = w_s \cdot 20^2 = w_s \cdot 400$
  - Spacing error is clipped to $\pm 20\;\text{m}$
- $J_u^\text{max} = w_u \cdot a_\text{max}^2 = w_u \cdot 9.0$
  - Maximum acceleration command: $3.0\;\text{m/s}^2$

**Numeric example** ($N = 4$, default weights):

$$
J_\text{max} = 1.0 \cdot 3 \cdot 225 + 0.5 \cdot 400 + 0.1 \cdot 9 = 675 + 200 + 0.9 = 875.9
$$

### Why this works

The non-negative reward preserves the same information content as the original negative cost (same ordering of states), but changes the incentive structure:

1. **With speed mode 95**: the agent now earns positive reward for driving well. "Always accelerate and let SUMO drive" still earns *some* positive reward, but actively tracking the equilibrium speed earns *more*. The agent has a gradient toward better driving.

2. **Collision termination**: returns reward $= 0$, which is always the worst outcome. The agent loses all future positive rewards from the truncated episode. No explicit collision penalty is needed — the loss of future reward is the penalty.

3. **No fragile tuning**: the collision cost scales automatically with episode length and reward magnitude, unlike a fixed penalty that must be hand-tuned.

### Implementation

In `compute_lcc_reward()`:

```python
J = abs(R_velocity) + abs(R_spacing) + abs(R_control)  # total cost >= 0
reward = max(self.J_max - J, 0.0) / self.J_max          # in [0, 1]
return float(reward)
```

The `/100.0` scaling is removed since the $[0, 1]$ range is already normalized.

For multi-agent (`compute_multi_agent_lcc_reward()`), the same transformation applies using a `J_max_multi` that accounts for multiple agents' spacing and control terms (both scale linearly with the number of agents).
