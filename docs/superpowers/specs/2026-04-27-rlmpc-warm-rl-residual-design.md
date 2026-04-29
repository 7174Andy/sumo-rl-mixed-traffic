# RLMPC for Mixed-Traffic Platoon — Design

**Date:** 2026-04-27
**Author:** Andrew Park
**Status:** Approved (awaiting plan)

## Goal

Implement two RL-augmented controllers on top of the existing NNMPC, following Jia & Bajaj (2025), "On Architectures for Combining Reinforcement Learning and Model Predictive Control with Runtime Improvements":

1. **Warm-Start RL** — actor network initialized from NNMPC weights, replaces NNMPC entirely.
2. **RL + NNMPC** — NNMPC outputs the bulk of the action; RL outputs a bounded residual; the sum is clipped to the physical action range.

Both methods are evaluated against the existing QP solver and the existing NNMPC on the same SUMO platoon, controlling the same two CAVs (positions 3 and 6 of an 8-follower platoon led by one head vehicle).

## Why

Two motivations from the paper, both apply here:

- **A. Close the QP↔NN gap.** The trained NNMPC has a residual error vs. the QP solver in closed loop; let RL trim that.
- **B. Robustness to HDV heterogeneity.** NNMPC was trained on the nominal homogeneous OVM but deployment-time HDVs vary. Let RL adapt to that mismatch.

Goal C from brainstorming (robustness to other failure modes — communication delay, IDM-driven HDVs, time-varying OVM) is explicitly out of scope for this design. It can be a follow-on once A+B work.

## Setting

- **Simulator:** SUMO (via TraCI). The existing OVM closed-loop simulator (`run_with_state`) is *not* used at training or eval time for this work — both train and eval go through SUMO so the experiment matches the SUMO line of work.
- **Platoon:** 1 head + 8 followers on the existing ring road. CAVs are at positions 3 and 6 (1-indexed within the platoon, matching `DeepLCCConfig`'s `ID = [0, 0, 1, 0, 0, 1, 0, 0]`). The other six are HDVs.
- **HDV behavior:** SUMO IDM car-following, **with per-vehicle parameters sampled fresh at every `reset()`** (this is the goal-B mechanism; nominal homogeneous IDM is a special case in the support).
- **Head behavior:** controlled `setSpeed` from a pre-computed velocity trace sampled per episode from `DeepLCCConfig.perturb_mix` (random ±A, brake, sinusoidal). This matches the head-vehicle distribution NNMPC was trained against.

The mismatch between SUMO IDM (HDV behavior at deployment) and the OVM model NNMPC was trained against is itself a concrete instance of goal A+B — RL has work to do without any artificial HDV randomization.

## Non-goals

- Multi-agent (decentralized per-CAV policy). Both methods run a single centralized policy with a 2-D joint action, matching the NN/QP experiment structure for fair comparison.
- Vectorized SUMO training. Single env per training run.
- Communication delay, IDM/OVM mismatch as a *trained* concern (deployment in SUMO already exposes us to it; we don't simulate further failure modes).
- Re-training NNMPC. We use the existing `deep_lcc_results/nnmpc.pth` as-is.

## Architecture

```
                                       ┌──────────────────────────────┐
                                       │      PlatoonNNMPCEnv         │
                                       │   (gymnasium.Env, in         │
                                       │    rl_mixed_traffic/         │
                                       │    deep_lcc/rlmpc_env.py)    │
  PPOAgent + ──obs (260)──→  NNMPCActor│  ┌────────────────────┐      │
  RolloutBuffer              Critic    │  │  SUMO platoon      │      │
  (existing,    (260→256→128→2,        │  │  1 head + 8 HDVs   │      │
   in agents/)   warm or resid init)   │  │  + CAVs at 3, 6    │      │
              ←─action (2)──           │  │  IDM + per-veh     │      │
                                       │  │  randomization     │      │
                                       │  └────────────────────┘      │
                                       │  ┌────────────────────┐      │
                                       │  │ NNMPC (frozen)     │      │
                                       │  │ inside step() if   │      │
                                       │  │ mode=residual      │      │
                                       │  └────────────────────┘      │
                                       │  ┌────────────────────┐      │
                                       │  │ (uini, yini, eini) │      │
                                       │  │ rolling buffers    │      │
                                       │  └────────────────────┘      │
                                       │  ┌────────────────────┐      │
                                       │  │ Lagrangian LCC     │      │
                                       │  │ reward (r_base −   │      │
                                       │  │ λ·violation; -1 on │      │
                                       │  │ collision)         │      │
                                       │  └────────────────────┘      │
                                       └──────────────────────────────┘
```

The training stack reuses the project's existing CleanRL-style PPO (`rl_mixed_traffic/agents/ppo_agent.py` + `rl_mixed_traffic/ppo/`) — no SB3, matching the rest of the codebase. Only the actor architecture and the env are RLMPC-specific.

### Files added

- `rl_mixed_traffic/deep_lcc/rlmpc_env.py` — `PlatoonNNMPCEnv` (extends/wraps `RingRoadEnv` for the NNMPC observation/action shape and to run NNMPC inside `step()`)
- `rl_mixed_traffic/deep_lcc/nnmpc_actor_critic.py` — `NNMPCActorCritic` network with NNMPC-shaped actor (260 → 256 → 128 → 2) + warm-start loader; reuses `ActorCriticNetwork`'s critic head pattern.
- `rl_mixed_traffic/deep_lcc/rlmpc_config.py` — `RLMPCConfig` dataclass
- `rl_mixed_traffic/deep_lcc/rlmpc_train.py` — train script (mode = `warm_start` or `residual`); reuses `PPOAgent` + `RolloutBuffer` from `rl_mixed_traffic/ppo/` + `rl_mixed_traffic/agents/`
- `rl_mixed_traffic/deep_lcc/rlmpc_eval.py` — eval script (compares all four controllers)
- `rl_mixed_traffic/deep_lcc/rlmpc_head_controller.py` — head-vehicle controller that plays a `perturb_mix` profile (subclass of `HeadVehicleController`)
- `configs/ring/platoon_9.rou.xml` — 1 head + 8 followers route file
- `configs/ring/platoon_9.sumocfg` — SUMO config referencing the new route file. **Must use `<step-length value="0.05"/>`** to match NNMPC's training `Tstep=0.05`. The existing `simulation.sumocfg` uses 0.1; we cannot reuse it.

### Files reused

- `rl_mixed_traffic/utils/sumo_utils.py` — TraCI lifecycle helpers
- `rl_mixed_traffic/configs/sumo_config.py` — `SumoConfig` dataclass
- `rl_mixed_traffic/configs/ppo_config.py` — `PPOConfig` (extend; new `RLMPCConfig` adds RLMPC-specific fields)
- `rl_mixed_traffic/agents/ppo_agent.py` — `PPOAgent` (CleanRL-style, used as-is)
- `rl_mixed_traffic/ppo/network.py` — `ActorCriticNetwork` (subclassed/replaced for NNMPC-shaped actor)
- `rl_mixed_traffic/ppo/rollout_buffer.py` — `RolloutBuffer` (used as-is)
- `rl_mixed_traffic/env/ring_env.py` — `RingRoadEnv` (subclassed; reuse SUMO lifecycle, leader-finding, `compute_multi_agent_lcc_reward`, `get_spacing_violation`)
- `rl_mixed_traffic/env/head_vehicle_controller.py` — `HeadVehicleController` (subclassed for `perturb_mix` profiles)
- `rl_mixed_traffic/deep_lcc/nnmpc_network.py` — `NNMPCNetwork`
- `rl_mixed_traffic/deep_lcc/nnmpc_config.py` — for paths and architecture defaults
- `rl_mixed_traffic/deep_lcc/config.py` — `DeepLCCConfig` for cost weights, perturbation mix, equilibrium
- `rl_mixed_traffic/deep_lcc/measurement.py` — `measure_mixed_traffic` for computing yini

## `PlatoonNNMPCEnv` specification

### Observation space

`Box(low=-inf, high=+inf, shape=(260,), dtype=float32)` — same as NNMPC input, structured as concatenation:

| slice | size | contents |
|---|---|---|
| `obs[:40]` | 40 | `uini`: past 20 steps × 2 CAV accelerations (the *applied* total, not just the policy output) |
| `obs[40:240]` | 200 | `yini`: past 20 steps × 10 measurements = 8 velocity errors `v_i - v_eq` + 2 CAV spacing errors `s_j - s_star` (matches `measurement.measure_mixed_traffic` with `measure_type=3`) |
| `obs[240:260]` | 20 | `eini`: past 20 steps × head-velocity error `v_head - v_eq` |

`v_eq` is the 20-second moving average of the head vehicle's speed (already implemented in `RingRoadEnv` as the `v_eq` property — to be reused). `s_star = 20.0 m` (from `DeepLCCConfig.s_star`).

The observation is normalized using NNMPC's own `input_mean` and `input_std` loaded from `nnmpc.pth`. Normalization happens in the env, so the policy receives standardized inputs identical to NNMPC's training data.

### Action space

`Box(low, high, shape=(2,), dtype=float32)` where `low/high` depend on mode (set at env construction time, not per-step):

- `mode="warm_start"`: `low = [-5, -5]`, `high = [3, 3]` — full physical CAV acceleration range.
- `mode="residual"`: `low = [-2, -2]`, `high = [2, 2]` — bounded correction to NNMPC's output.

### `step(action)` flow

1. Update observation buffers from current SUMO state. Specifically:
   - Append latest CAV accelerations (the actually applied total from the previous step) to `uini` buffer.
   - Compute current 8-velocity-error + 2-spacing-error vector via `measure_mixed_traffic`; append to `yini` buffer.
   - Compute current head-velocity-error; append to `eini` buffer.
2. Compose 260-dim observation, normalize with NNMPC stats.
3. If `mode == "residual"`: forward NNMPC on the (normalized) observation to get `u_nnmpc` ∈ [-5, 3]². Otherwise `u_nnmpc = 0`.
4. Compute `u_total = clip(u_nnmpc + action, [-5, 3])`. (This is a no-op for `warm_start`, since action already lives in [-5, 3].)
5. `apply_acceleration([car3, car6], u_total, smooth=False)`. Apply head's pre-computed velocity for this step via `setSpeed`.
6. `traci.simulationStep()`.
7. Update `v_eq` buffer with current head speed.
8. Compute reward using the **Lagrangian-augmented LCC reward** matching the existing `RingRoadEnv` design:
   - **Base reward** (from `compute_multi_agent_lcc_reward`):
     ```
     J_velocity = weight_v · Σ_i (v_i - v_eq)²        # all 9 vehicles except head
     J_spacing  = weight_s · Σ_cav clip(s_cav - s_star, ±20)²   # both CAVs
     J_control  = weight_u · Σ_cav a_cav²              # both CAVs
     J = J_velocity + J_spacing + J_control
     r_base = max(J_max_multi - J, 0) / J_max_multi    # in [0, 1]
     ```
     `J_max_multi` is precomputed from `weight_v · max_v_error² · (num_vehicles - 1) + 2·weight_s·20² + 2·weight_u·max_accel²`. At equilibrium `r_base = 1.0`; in worst case `r_base = 0.0`.
   - **Lagrangian spacing-violation penalty** (from `get_spacing_violation`):
     ```
     violation = Σ_cav max(0, s_min - gap_cav) / s_min     # zero unless a CAV is within 5 m of its leader
     r_aug = r_base - λ · violation                          # λ = 1.0 default (configurable)
     ```
     The Lagrangian penalty engages *before* a collision happens so the policy gets a graded signal as it approaches unsafe spacing.
   - **Collision override**: if SUMO reports a collision this step, reward = `-1.0` (overrides the above), and `terminated=True`. This is a one-shot strong negative — combined with the otherwise-positive per-step rewards, it makes "less-negative cumulative reward" exactly equivalent to "fewer collisions" for the agent's optimization.
9. Termination conditions:
   - `step_count >= max_steps` → `truncated=True`.
   - SUMO collision → `terminated=True` (and reward already set to `-1.0` above).
10. Return `(obs, reward, terminated, truncated, info)`. `info` carries `{"r_base": ..., "violation": ..., "collision": bool}` for diagnostics.

**Cost weights:** match `DeepLCCConfig` — `weight_v=5.0, weight_s=0.1, weight_u=0.1`, `s_star=20.0`, `spacing_min=5.0`. These are the same weights the QP and NNMPC are designed for, so the RL reward optimizes the same objective the upstream controllers pursue. `PlatoonNNMPCEnv` overrides the `RingRoadEnv` defaults (1.0/0.5/0.2) at construction time by passing the DeepLCCConfig values into the parent's `__init__`; `RingRoadEnv` already accepts these as constructor arguments and recomputes `J_max_multi` automatically.

**Numeric implication for `J_max_multi`** (with `v_star=15`, `v_max=30`, `max_accel=3`, `num_vehicles=9`, `num_agents=2`):
```
J_v_max     = 5.0 · 15² · (9 − 1) = 9000
J_s_max     = 0.1 · 20²            = 40
J_u_max     = 0.1 · 3²             = 0.9
J_max_multi = 9000 + 2·40 + 2·0.9  ≈ 9081.8
```
The velocity term dominates by ~100×, so `r_base` is effectively a normalized velocity-tracking score with small spacing and control corrections — same shape as the QP cost.

### `reset(seed, options)` flow

1. Close any prior TraCI connection. Open a fresh one with the new route file.
2. Sample HDV parameters per follower using `seed` (or env-internal RNG):
   - For each non-CAV follower (`car1, car2, car4, car5, car7, car8`):
     - `tau ~ U[0.8, 1.5]`
     - `accel ~ U[1.5, 2.5]`
     - `decel ~ U[2.5, 3.5]`
     - `minGap ~ U[2.0, 3.0]`
     - `sigma ~ U[0.3, 0.6]`
   - Apply via `traci.vehicle.setTau`, `setAccel`, `setDecel`, `setMinGap`, `setImperfection`.
3. Sample one perturbation type from `DeepLCCConfig.perturb_mix` (random/brake/sinusoidal). Pre-compute the head velocity trace for the whole episode using the same generators that produce training data (`make_extreme_brake`, `make_sinusoidal`, or random ±A).
4. Set CAV speed mode to `95` (matches `RingRoadEnv`) and max speed to `v_max = 30.0`.
5. Warm up: simulate for `T_ini = 20` steps with no policy action.
   - During warm-up, **do not** set CAV speed mode to 95 — leave SUMO IDM in charge of the CAVs so the platoon settles naturally.
   - Each warm-up step, append `traci.vehicle.getAcceleration(cav_id)` (the IDM-applied accel for that step) to `uini`. Likewise update `yini` and `eini`.
   - At step `T_ini = 20`, *then* set CAV speed mode to 95 and `setMaxSpeed(v_max)`. From step 20 onward, the policy controls the CAVs via `apply_acceleration(..., smooth=False)`.
6. Build initial observation from filled buffers; return.

### Episode parameters

- `Tstep = 0.05 s` (matches `DeepLCCConfig`)
- `episode_length = 30.0 s` → 600 steps per episode (post-warm-up: 580 steps under policy control)
- `max_steps = 600`

### Internal state to track

- Rolling buffers: `uini` (deque maxlen 20×2=40 floats), `yini` (deque maxlen 200), `eini` (deque maxlen 20).
- `v_eq` running average (deque of head speeds, maxlen `int(20.0 / Tstep) = 400`).
- `prev_total_accel` per CAV — feeds `uini` next step.
- `step_count`.
- Pre-computed `head_vel_trace: np.ndarray` for the whole episode.
- `nnmpc_model` (loaded once per env instance, frozen, in `eval()` mode, on CPU).
- `input_mean`, `input_std` for observation normalization.

### Reward — implementation notes

The reward is computed by reusing `RingRoadEnv.compute_multi_agent_lcc_reward()` for `r_base` and `RingRoadEnv.get_spacing_violation()` for the Lagrangian penalty — both already implemented and unit-tested in `tests/test_compute_lcc_reward.py`. `PlatoonNNMPCEnv` subclasses `RingRoadEnv` and only needs to:

1. Override `__init__` to set `num_vehicles=9, num_agents=2, agent_ids=["car3", "car6"]`, override `agent_id` to `"car3"` for backward-compat.
2. Override the reward path in `step()` to apply the Lagrangian augmentation (`r_aug = r_base - λ · violation`) and the collision override (`r = -1.0 + terminate`).
3. Maintain `prev_accels[aid]` so `compute_multi_agent_lcc_reward` sees the *applied total* CAV accelerations (NNMPC + residual after clipping), not just the policy output.

This is intentionally minimal: the reward math stays where it is in `RingRoadEnv` and is shared with the existing PPO experiments. Any future changes to the reward apply uniformly to both lines of work.

## Policy specification

The existing `ActorCriticNetwork` (in `rl_mixed_traffic/ppo/network.py`) uses a shared 2-layer MLP with `tanh` activations and separate actor-mean and critic heads, designed for low-dim observations. For RLMPC we need a deeper, ReLU-based actor that matches `NNMPCNetwork`'s architecture so warm-start works by direct weight load. We keep `ActorCriticNetwork`'s API (returns `(actor_output, value)` and provides `get_action_and_value`) so `PPOAgent` works unchanged.

### `NNMPCActorCritic` (new, in `nnmpc_actor_critic.py`)

Subclass-or-replace `ActorCriticNetwork` with:

- **Actor body**: MLP `260 → 256 → 128 → 2` with ReLU activations, final layer's pre-activation linear *output* is the unsquashed action mean. The trained `NNMPCNetwork` is `260 → 256 → 128 → 2 → tanh → scale to [accel_min, accel_max]`. We load NNMPC's `state_dict` into our actor body's first three `Linear` layers (the MLP up to the pre-tanh output). The downstream `tanh` + scale happens in the action distribution / scaling step, not in the network module — so the warm-started actor produces the *same numeric action* as NNMPC at step 0.
- **Critic head**: independent MLP `260 → 256 → 128 → 1` with ReLU. No warm-start.
- **`actor_log_std`**: state-independent learnable `nn.Parameter`, mirroring `ActorCriticNetwork.actor_log_std`. Initialization differs by mode:
  - `warm_start`: `log_std₀ = log(0.1)` so initial sampled actions ≈ NNMPC + small noise.
  - `residual`: `log_std₀ = log(0.5)` for more exploration; actor mean head's last layer is orthogonal-init with gain `0.01`, so initial residual ≈ 0 and total action ≈ NNMPC alone.
- **`get_action_and_value(state, action)`**: matches the signature in `ActorCriticNetwork` so `PPOAgent` is unchanged. Internally:
  ```
  actor_pre_tanh, value = forward(state)
  log_std = clamp(actor_log_std, -2.0, 0.5)
  std = exp(log_std)
  dist = Normal(actor_pre_tanh, std)
  if action is None:                          # rollout
      raw = dist.sample()
      action_squashed = tanh(raw)
  else:                                       # PPO update
      action_squashed = action                 # already in [-1, 1]
      raw = atanh(clip(action_squashed, -0.999, 0.999))
  log_prob = dist.log_prob(raw).sum(-1)
            - log(1 - action_squashed**2 + 1e-6).sum(-1)   # tanh correction
  return action_squashed, log_prob, dist.entropy().sum(-1), value
  ```
  The action returned to the env is `action_squashed ∈ [-1, 1]`. The env's `step()` *scales* it to the action box (`[-5, 3]` for warm-start, `[-2, 2]` for residual) before applying. This mirrors `ppo_train.py`'s existing `action_scaled = action_tanh * max_accel` pattern.

### Warm-start procedure (`NNMPCActorCritic.warm_start_from_nnmpc(ckpt_path)`)

```
1. Load nnmpc.pth → state_dict, input_mean, input_std.
2. Map NNMPC's nn.Sequential layers (Linear/ReLU/Linear/ReLU/Linear/Tanh)
   onto the actor body's first three Linear layers, ignoring the final Tanh
   (we apply tanh in the distribution step, not the body).
3. Verify shape match before copying; raise on mismatch.
4. Return (input_mean, input_std) for the env to use.
```

The env *also* loads `nnmpc.pth` independently — it needs to *run* NNMPC for `mode="residual"` *and* it needs `input_mean/std` to normalize observations consistently with how NNMPC was trained.

### Why not modify `ActorCriticNetwork` directly?

`ActorCriticNetwork` is shared with the existing `RingRoadEnv` PPO line. Changing its hidden dims or activations could break those experiments' loaded checkpoints. A new class adds zero risk to existing code. Both classes implement the same `get_action_and_value` contract, so `PPOAgent` accepts either.

## Training pipeline

### `RLMPCConfig`

Reuses `PPOConfig` for PPO hyperparams; adds RLMPC-specific fields. Hyperparams default to the values in `rl_mixed_traffic/configs/ppo_config.py` so behavior is consistent with the existing PPO experiments.

```python
@dataclass
class RLMPCConfig:
    mode: Literal["warm_start", "residual"] = "warm_start"
    nnmpc_path: str = "deep_lcc_results/nnmpc.pth"
    sumocfg_path: str = "configs/ring/platoon_9.sumocfg"
    use_gui: bool = False

    # Episode
    episode_length_s: float = 30.0
    Tstep: float = 0.05

    # PPO (delegated to PPOConfig — keep names matching the existing dataclass)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    total_steps: int = 1_000_000        # mirrors ppo_train.py
    rollout_steps: int = 4096
    save_freq: int = 100_000

    # Lagrangian penalty
    lambda_violation: float = 1.0       # weight on spacing-violation in r_aug

    # HDV randomization (per-vehicle, per-episode)
    hdv_tau_range: tuple[float, float] = (0.8, 1.5)
    hdv_accel_range: tuple[float, float] = (1.5, 2.5)
    hdv_decel_range: tuple[float, float] = (2.5, 3.5)
    hdv_minGap_range: tuple[float, float] = (2.0, 3.0)
    hdv_sigma_range: tuple[float, float] = (0.3, 0.6)

    # Logging / output
    seed: int = 42
    out_dir: str = "deep_lcc_results/rlmpc_{mode}/"
```

### Training script

Mirrors the structure of `rl_mixed_traffic/ppo_train.py` (multi-agent path) but with the new env, the new actor-critic, and a single 2-D joint action (no per-agent action loop):

```python
config = RLMPCConfig(mode="warm_start")  # or "residual"

env = PlatoonNNMPCEnv(config)
env = FourToFiveTupleWrapper(env)        # reuse existing wrapper

obs_dim = env.observation_space.shape[0]   # 260
action_dim = env.action_space.shape[0]      # 2

network = NNMPCActorCritic(
    obs_dim=obs_dim, action_dim=action_dim,
    hidden_dims=(256, 128),
    log_std_init=log(0.1) if config.mode == "warm_start" else log(0.5),
    final_layer_gain=1.0 if config.mode == "warm_start" else 0.01,
)
if config.mode == "warm_start":
    network.warm_start_from_nnmpc(config.nnmpc_path)

agent = PPOAgent(
    obs_dim=obs_dim, action_dim=action_dim,
    config=config.ppo, continuous=True,
    total_steps=config.total_steps,
    rollout_steps=config.rollout_steps,
)
agent.network = network                    # swap in the NNMPC-shaped network

# rollout / learn loop matches ppo_train.py's _train_single_agent:
#   action_tanh, value, log_prob = agent.get_action_and_value(s)
#   action_scaled = action_tanh * max_accel  (with mode-aware max_accel)
#   agent.store_transition(...) + agent.learn(last_value)
```

The env's `max_accel` is mode-aware: `3.0` for warm-start (asymmetric clip later), `2.0` for residual. The action is scaled to that bound then passed through. Asymmetric `[-5, 3]` clipping for warm-start happens in `apply_acceleration` already (existing code clips to `[min_accel, max_accel]` per CAV).

### Logging

Plain CSV + matplotlib (matches `ppo_train.py`'s `plot_returns` / `plot_ppo_metrics`):
- `deep_lcc_results/rlmpc_{mode}/returns.csv` — episode index, return, length, mean r_base, mean violation, collision flag.
- `deep_lcc_results/rlmpc_{mode}/returns.png` — return vs episode plot.
- `deep_lcc_results/rlmpc_{mode}/ppo_metrics.png` — policy loss, value loss, entropy, clip fraction.
- Per-checkpoint snapshots `agent_step_{N}.pth` saved every `save_freq` steps.

### Compute budget

`1M timesteps / 600 steps-per-ep ≈ 1666 episodes`. Per-episode wall-clock dominated by SUMO step time. Estimated total per training run: 10–20 hours. Both modes train sequentially (no parallel SUMO), total ≈ 20–40 hours.

### Curriculum (optional, off by default)

If warm-start RL diverges in the first ~100 episodes (paper notes this as a risk due to immature critic), enable a critic-only warm-up: freeze the actor's parameters for the first 50 updates, train only the critic head. Implemented inline in the train loop with a flag (`config.critic_warmup_updates: int`). Default 0. Skip for residual since the actor starts near zero anyway.

## Evaluation pipeline

### Controllers

1. **QP** — runs the cvxopt-based `CachedDeepLCCSolver` from `qp_solver.py`. Pre-collected Hankel matrices are built once per controller-instance (not per-episode); fresh for each HDV config since precollection requires HDV dynamics. **NB:** the QP precollection is done in OVM-land (`run_with_state` with `precollect`), but the *online* QP solve runs against the SUMO-driven (uini, yini, eini) coming from `PlatoonNNMPCEnv`. This is consistent with the paper's framing: the controller's predictive model is offline-fit, the plant is the (mismatched) real system.
2. **NNMPC** — pure forward pass of `nnmpc.pth`, no RL.
3. **Warm-Start RL** — `PPOAgent.load("deep_lcc_results/rlmpc_warm_start/agent.pth")`, `agent.act(obs, eval_mode=True)` (deterministic mean action). Env in `mode="warm_start"`.
4. **RL + NNMPC** — `PPOAgent.load("deep_lcc_results/rlmpc_residual/agent.pth")`, `agent.act(obs, eval_mode=True)`. Env in `mode="residual"` so NNMPC runs inside the env.

### Eval scenarios

Same head-vehicle profiles already used in `nnmpc_eval.eval_closed_loop`:
- `brake` (extreme braking, 30 s)
- `sinusoidal` (40 s)
- `varying_sine` (200 s)
- `aggressive_sine` (200 s)
- `stop_and_go` (200 s)
- `NEDC` (~400 s, native length)

For each scenario, episode length matches the scenario duration (overriding training's 30 s).

### HDV configurations

Three per scenario:
- **`nominal_idm`** — homogeneous IDM with default SUMO parameters (the route file's `<vType id="idm" ...>` defaults). Tests goal A.
- **`hetero_fixed`** — fixed heterogeneous parameters from `hdv_ovm_2.mat`, mapped to IDM equivalents (or use a deterministic seed for sampling). Tests goal B with one fixed sample.
- **`hetero_random`** — 5 seeds, each sampling HDV params from the same training distribution. Tests goal B over the proper distribution.

So the eval grid is `{6 scenarios} × {3 HDV configs} × {4 controllers} = 72 evaluations` (with `hetero_random` actually being 5 seeds → 6 × 5 = 30 + 6 × 2 = 12 ⇒ 42 (controller, scenario, hdv) combinations counting `hetero_random` as 5 separate combos).

### Metrics per evaluation

- **Total cost** = sum of `J = J_velocity + J_spacing + J_control` over all post-warm-up steps, computed with `DeepLCCConfig` weights (`weight_v=5.0, weight_s=0.1, weight_u=0.1`). Primary metric — directly comparable across all four controllers and to the existing OVM-based `nnmpc_eval` numbers (modulo the OVM↔SUMO dynamics gap noted below).
- **Cumulative reward** = sum of per-step `r_aug` over the episode. This is the actual quantity the policy optimized.
- **MSVE_cav0**, **MSVE_cav1**, **MSVE_avg** — mean squared velocity error vs head, paper's metric.
- **Collision count** — number of SUMO-reported collisions in the episode.
- **Spacing-violation count** — number of steps with `min(s_cav0, s_cav1) < 5.0 m`.
- **Min spacing**, **max spacing** — safety summary.
- **Fuel consumption** (mL/vehicle, follower vehicles 3–8) using the existing `compute_metrics` formula.
- **Mean per-step latency (μs)** — `time.perf_counter()` around the controller call only.

### Outputs

Under `deep_lcc_results/rlmpc/`:
- `summary.csv` — one row per (scenario, hdv_config, seed, controller).
- `summary.md` — human-readable table grouped by scenario, ranked by cost.
- `{scenario}_{hdv}_velocities.png` — overlay of all four controllers' CAV velocities + applied control trace, head vehicle as dashed reference. Extends `_plot_qp_vs_nn` to four series.
- `{scenario}_{hdv}_cost_bars.png` — bar chart of total cost across controllers.
- `traces/{scenario}_{hdv}_{seed}_{controller}.npz` — raw per-step data (positions, velocities, accels, spacings, costs).

### Determinism

- Trajectory rollout seed: `999` (matches `run_with_state` default), so SUMO step-level randomness is the same across controllers.
- HDV-param sample seed: paired with controller — for each (scenario, hdv_config, seed) tuple, the *same* HDV param sample is used across all four controllers. This makes the comparison apples-to-apples.

### Note on cost comparability with the existing OVM-based eval

`nnmpc_eval.py` runs in OVM-land and produces cost numbers under the OVM dynamics. The new SUMO-based eval will produce *different* absolute cost numbers because the dynamics differ. **Direct numeric comparison between the OVM-based cost and the SUMO-based cost is not meaningful.** The new eval starts a fresh "SUMO leaderboard" — within that leaderboard, all four controllers are directly comparable.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Warm-start RL diverges due to immature critic | Medium | Critic-only warm-up callback (off by default, enable if needed). |
| HDV randomization too wide → CAVs can't avoid collisions | Medium | Conservative ranges already chosen (tau ≥ 0.8 s, decel ≥ 2.5). Tighten if collision rate > ~5% during training. |
| 1M timesteps insufficient for PPO convergence | Medium | Watch return curve in the first 100k steps; bump to 2M if return is still rising at 1M. |
| SUMO-side determinism: head trajectory `setSpeed` may be overridden by SUMO safety checks for `car0` | Low | Set `car0`'s speed mode to a permissive value (e.g., 0 or a custom mask) at reset. Verify by recording actual head speeds vs commanded. |
| Observation normalization drift: training-time HDVs differ from NNMPC's training distribution, so `input_mean/std` may not be appropriate | Medium | Acceptable for warm-start (we want NNMPC to start "as-is"). For residual, this is also fine since the policy is learning from scratch and will adapt. Don't recompute `input_mean/std` on the new distribution — that would invalidate warm-start. |
| Wall-clock training too long for iteration | High | Document explicitly. If needed, reduce `total_steps`, shorten episodes, or vectorize SUMO (separate plan). |
| NNMPC inside residual env: forward pass cost adds to env step time | Low | NNMPC is small (~100K params); CPU forward < 1 ms per step. SUMO step dominates. |
| Ring length too short for 9 vehicles at `s_star=20m` (need ≥ 180 m circumference) | Medium | Verify ring length via `compute_ring_length` at first env init. If short, lengthen by editing `circle.rou.xml`'s `repeat` count or use a longer ring (`circle.net.xml`). Document the verified length in the env's docstring. |
| Step-length mismatch: existing `simulation.sumocfg` uses `0.1` but NNMPC was trained at `0.05` | Mitigated by design | New `platoon_9.sumocfg` uses `step-length="0.05"`. Verify env's `step_length` matches at startup; raise if mismatch. |

## Acceptance criteria

For the design to be considered successfully implemented:

1. Both training runs complete (warm-start and residual), producing saved policies.
2. Both training return curves show monotone improvement after the first ~100 episodes (visual inspection).
3. Eval grid produces a complete `summary.csv`, `summary.md`, and the per-scenario plots.
4. **At least one of {warm-start, residual} achieves lower total cost than NNMPC** averaged across the `hetero_random` config (testing goal A+B). If neither beats NNMPC even on average, the experiment is informative but the methods didn't help — that itself is a useful negative result.
5. Neither method causes more collisions than NNMPC averaged across all eval grid cells (safety regression check).
6. Per-decision latency for RL controllers is within 2× of NNMPC's (runtime claim from paper).

## Out-of-scope follow-ons

If A+B succeed, the natural follow-on is reusing this infrastructure for goal C (delay, IDM, time-varying OVM mismatch as targeted training scenarios). That belongs in a separate spec, sharing the env code but extending the perturbation/HDV sampling.
