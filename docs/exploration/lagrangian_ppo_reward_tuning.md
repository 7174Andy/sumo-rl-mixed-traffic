# Lagrangian PPO Reward Tuning — Exploration Log

!!! note "Related documentation"
    - [Lagrangian PPO](../lagrangian-ppo.md) — main design doc with architecture, config tables, results summary, and **Section 6: Design Iterations** (chronological summary of all major changes)
    - [Reward Redesign](../reward-redesign.md) — the non-negative reward transformation that resolved the crash-to-escape problem
    - [Problem Formulation: CMDP](../problem.md#15-constrained-mdp-cmdp-formulation) — formal CMDP with $s_\text{min}$ / $s_\text{max}$ constraints

## 1. Overview

**Goal:** Train a Lagrangian PPO agent for safe car-following on a ring road.

**Setup:**
- 4 vehicles: 1 head vehicle (`car0`) + 1 CAV agent (`car1`) + 2 HDVs
- Head vehicle changes speed randomly every 15s (range 5–20 m/s)
- Reward: DeeP-LCC cost-based (velocity error + spacing error + control penalty)
- Key config flags: `disable_sumo_safety=True`, `enable_safety_layer=True`

**Reward function** (`ring_env.py:645`, `compute_lcc_reward`):

```
J = weight_v * sum((v_i - v_eq)^2)     # all vehicles except head
  + weight_s * clip(gap - s_star, ±20)^2  # CAV only
  + weight_u * accel^2                     # CAV only

reward = max(J_max - J, 0) / J_max    ∈ [0, 1]
```

Initial weights: `weight_v=1`, `weight_s=0.5`, `weight_u=0.2`, `s_star=20m`. (Later rebalanced to `weight_v=5.0`, `weight_u=0.1` — see Section 7.)

---

## 2. Round 1 — Standard PPO Baseline (Pre-Fix)

**Config:** `ppo_train.yaml`, `disable_sumo_safety=False` (SUMO Krauss safety model active)

**Observations from evaluation plots:**
- Speed tracking looks reasonable — CAV appears to follow head vehicle transitions
- Acceleration is constant at +2.0 m/s² with no modulation
- Returns: negative (−30k → −5k range), trained with old `compute_reward()`
- Spacing: 5–15m range, crash at step ~4300

**Root cause:** SUMO's Krauss car-following model performs the actual safe following. The agent simply outputs "accelerate" every step, and Krauss overrides it to prevent collisions. The agent learned to free-ride on SUMO's built-in safety rather than developing its own policy.

**Verdict:** Not real learning — the agent exploits SUMO's safety guarantees.

---

## 3. Round 1 — Lagrangian PPO (Pre-Fix)

**Config:** `lagrangian_ppo_train.yaml`, `disable_sumo_safety=True`

With SUMO safety disabled, the agent must learn car-following from scratch.

**Observations:**
| Metric | Behavior |
|--------|----------|
| Speed tracking | Poor — CAV lags 2–5 m/s behind head, stalls to ~1 m/s |
| Spacing | Oscillates 0–200m, spike to 960m |
| Acceleration | Near-zero — agent learned "do nothing" |
| Returns | 500–4000, no convergence over 450 episodes |
| Entropy | Increasing (1.45 → 1.8) — policy randomizing |
| Value loss | Increasing — critic getting worse |
| Lambda | Flat at ~0 |

**Root cause — Reward whiplash from instantaneous v_eq:**

Before the fix, `v_eq` was computed directly from the head vehicle's current speed. When the head vehicle jumps (e.g., 15 → 7 m/s), `v_eq` changes instantly, creating:

1. **Velocity error spikes:** Every vehicle is suddenly "wrong" relative to the new target
2. **Moving goalpost:** `s_star` was also dynamic (OVM-based), shifting with `v_eq`
3. **Noisy reward signal:** Large reward swings at every head speed change prevent gradient convergence
4. **Agent gives up:** The policy converges to near-zero acceleration (safest strategy when reward is unpredictable)

The entropy increase confirms this — the policy is randomizing rather than exploiting, a hallmark of a reward signal that doesn't carry learnable structure.

---

## 4. Fix Applied — Smoothed v_eq + Fixed s_star

**Reference:** DeeP-LCC paper (arXiv 2203.10639), Section VI.

### Changes to `rl_mixed_traffic/env/ring_env.py`

**1. Head speed buffer** (line 151–153):
```python
_v_eq_window = int(round(20.0 / self.step_length))  # 200 steps at dt=0.1s
self._head_speed_buffer: deque[float] = deque(maxlen=_v_eq_window)
```

**2. v_eq property** (line 178–183):
```python
@property
def v_eq(self) -> float:
    """Time-averaged equilibrium velocity from head vehicle speed buffer."""
    if len(self._head_speed_buffer) > 0:
        return float(np.mean(self._head_speed_buffer))
    return self.v_star
```

**3. Buffer recording** — head speed appended after each `simulationStep()` call in both `_step_single` (line 512) and `_step_multi` (line 564):
```python
self._head_speed_buffer.append(traci.vehicle.getSpeed(self.head_id))
```

**4. Reward functions updated** — both `compute_lcc_reward()` (line 661) and `compute_multi_agent_lcc_reward()` (line 699) now use:
- `self.v_eq` (20s time-averaged) instead of instantaneous head speed
- `self.s_star` (fixed 20m) instead of dynamic OVM-based s_star

**Why this works:**
- The 20s averaging window smooths over head speed changes (which occur every 15s)
- v_eq transitions gradually, giving the agent a learnable signal
- Fixed s_star eliminates the moving-goalpost problem
- J_max normalization remains valid (worst-case bounds unchanged)

---

## 5. Round 2 — Lagrangian PPO (Post-Fix)

**Config:** Same `lagrangian_ppo_train.yaml`, retrained for 800k steps.

### Results

| Metric | Behavior |
|--------|----------|
| Speed tracking | **Good** — CAV follows head within 1–3 m/s across all speed transitions (5–15 m/s range) |
| Acceleration | **Smooth, modulated** — range 0 to +1.5 m/s², clear response to speed changes |
| Spacing | **Stable 5–20m** for entire episode (~4700 steps), near s_star=20m |
| Returns | MA(10) rises to ~3500 by episode 80, plateaus at 3000–3800 |
| Entropy | Rose to 1.65, then declined to 1.50 (healthy explore → exploit curve) |
| Lambda | Still flat at 0 (see Section 6) |
| Value loss | Concerning rise in later updates (critic struggling) |
| Crashes | **None** — full-episode car-following achieved |

### Remaining behaviors

- **Catching-up lag (~2–3s):** After head speed changes, the CAV takes a few seconds to match. This is partly by design — the 20s averaging window inherently delays v_eq, and the reward doesn't explicitly penalize transient tracking error.
- **Slight positive acceleration bias:** The agent tends to accelerate more than brake. Likely because the DeeP-LCC cost penalizes control input symmetrically (`u^2`), but positive acceleration is more useful for catching up.
- **End-of-episode spacing spike:** Normal — the head vehicle exits the simulation before the episode ends, causing the gap measurement to jump.

---

## 6. Why the Lagrangian Multiplier Never Engages

The Lagrangian dual update (`lagrangian_ppo_train.py`):

```python
lambda_val += lambda_lr * (mean_violation - violation_tolerance)
```

With actual values from training (current config: `lambda_init=1.0`, `lambda_lr=0.05`, `violation_tolerance=0.1`):
```
lambda += 0.05 * (0.03 - 0.1) = -0.0035 → lambda decreases each rollout
```

Even starting from `lambda_init=1.0`, the multiplier decays toward zero because the safety layer clips unsafe accelerations **before** they reach SUMO, keeping normalised violations at 0.02-0.04 — well below the tolerance of 0.1.

**The safety layer and Lagrangian work against each other:**

1. Safety layer prevents violations (hard constraint at execution time)
2. Violations stay near zero → dual update is negative → $\lambda$ decreases
3. $\lambda$ decays to zero → Lagrangian penalty vanishes → no safety learning signal
4. The policy learns car-following purely from the DeeP-LCC reward, with no learned safety behavior

Note: earlier experiments used `lambda_init=0.0` and `lambda_lr=0.01`, which meant $\lambda$ never moved from zero at all. Increasing to `lambda_init=1.0` and `lambda_lr=0.05` provides initial penalty pressure, but the same decay dynamic takes over once training stabilizes.

**Possible fixes (not yet applied):**

| Approach | Trade-off |
|----------|-----------|
| Lower `violation_tolerance` (e.g., 0.01) | Lambda engages even with small violations; may over-penalize |
| Use safety layer clip rate as constraint signal | Penalizes the policy for *attempting* unsafe actions, not just outcomes |
| Disable safety layer, rely solely on Lagrangian | Riskier during training — real collisions possible |
| Keep current setup (safety layer + DeeP-LCC) | Simplest; safety is guaranteed by the layer, not learned |

---

## 7. Current Status

### What's working
- Smoothed $v_\text{eq}$ fix resolved the convergence problem
- Agent achieves functional car-following with no crashes
- Acceleration is modulated and responsive (not constant or zero)
- Spacing stays in a reasonable range near the 20 m target
- Safety layer enforces both $s_\text{min}$ and $s_\text{max}$ as hard constraints

### Applied improvements (since this log was written)

These items were originally listed as "potential improvements" and have since been implemented. See [Lagrangian PPO Section 6](../lagrangian-ppo.md#6-design-iterations) for the full iteration history.

- **Weight rebalancing** (applied): `weight_v` increased from 1.0 to 5.0, `weight_u` decreased from 0.2 to 0.1. Reduces catch-up lag.
- **Longer training** (applied): Total steps increased from 800k to 1,500,000.
- **Value function clipping** (applied): `clip_vloss=true` with `vf_clip_coef=0.2` mitigates value loss instability.
- **$s_\text{max}$ hard constraint** (applied): Added to safety layer to prevent CAV from drifting away from head vehicle (see Iteration 5).
- **Collision penalty** (applied): `reward = -1.0` on collision (see Iteration 6).
- **Lambda tuning** (applied): `lambda_init` increased from 0.0 to 1.0, `lambda_lr` from 0.01 to 0.05. $\lambda$ still decays to zero due to safety layer effectiveness (Section 6).

### Open questions
- **Lagrangian engagement:** The Lagrangian multiplier remains ineffective due to the safety layer paradox (Section 6). Whether to tune the mechanism further or accept the safety-layer-only approach is an open research question.
- **Value loss oscillations:** Improved with clipping but not fully resolved.
- **Catch-up lag:** Reduced by weight rebalancing but inherent to the 20 s averaging window.

### Key files modified
| File | Change |
|------|--------|
| `rl_mixed_traffic/env/ring_env.py` | Added `_head_speed_buffer`, `v_eq` property, smoothed reward, collision penalty, normalised violations |
| `rl_mixed_traffic/env/safety_layer.py` | Added $s_\text{max}$ constraint (clip UP), conflict resolution |
| `rl_mixed_traffic/conf/lagrangian_ppo_train.yaml` | Training config for Lagrangian PPO (updated weights, lambda, spacing_max) |
| `rl_mixed_traffic/lagrangian_ppo_train.py` | Lagrangian dual update loop, augmented return tracking |
