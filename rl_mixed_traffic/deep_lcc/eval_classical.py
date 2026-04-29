"""Evaluate classical DeeP-LCC (QP) on extreme braking and NEDC scenarios.

Reports performance metrics matching the paper:
- Total real cost: sum of y'Qy + u'Ru over the simulation
- Fuel consumption (mL/s) using model from [Bowyer 1985] for vehicles 3-8
- MSVE (mean squared velocity error) vs head vehicle
- Max acceleration / deceleration
- Min/max spacing observed (safety check)

Usage:
    uv run rl_mixed_traffic/deep_lcc/eval_classical.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from rl_mixed_traffic.deep_lcc.config import (
    DeepLCCConfig,
    OVMConfig,
    get_heterogeneous_ovm_config,
)
from rl_mixed_traffic.deep_lcc.precollect import precollect
from rl_mixed_traffic.deep_lcc.qp_solver import CachedDeepLCCSolver


# ── Scenario builders ─────────────────────────────────────────────────


def make_extreme_brake(
    total_steps: int,
    tstep: float,
    v_star: float,
    settle_time: float = 5.0,
) -> np.ndarray:
    """Brake head vehicle trajectory matching the reference DeeP-LCC paper.

    Phases (with brake_amp=10, default settle_time=5s):
        0..5s    : settle at v_star (let CAVs reach steady state under QP)
        5..7s    : brake at -5 m/s² → 15 → 5 m/s
        7..12s   : coast at 5 m/s
        12..17s  : accelerate at +2 m/s² → 5 → 15 m/s
        17s..end : cruise at v_star

    Reference: soc-ucsd/DeeP-LCC main_brake_simulation.py with brake_amp=10.
    Returns absolute head vehicle velocity (m/s).
    """
    head_vel = np.full(total_steps, v_star)
    v = v_star
    brake_start = settle_time
    coast_start = brake_start + 2.0   # 2s of -5 m/s² brings 15 → 5
    accel_start = coast_start + 5.0   # 5s of coasting at 5
    accel_end = accel_start + 5.0     # 5s of +2 m/s² brings 5 → 15
    for k in range(total_steps):
        t = k * tstep
        if t < brake_start:
            a = 0.0
        elif t < coast_start:
            a = -5.0
        elif t < accel_start:
            a = 0.0
        elif t < accel_end:
            a = 2.0
        else:
            a = 0.0
        v = max(0.0, v + a * tstep)
        head_vel[k] = v
    return head_vel


def make_sinusoidal(
    total_steps: int,
    tstep: float,
    v_star: float,
    amplitude: float = 5.0,
    period: float = 10.0,
    settle_time: float = 5.0,
) -> np.ndarray:
    """Sinusoidal perturbation on head vehicle velocity.

    Matching the reference (per_type=1): sine_amp=5, period=10s.
    Includes a settle phase before the perturbation starts.

    Returns absolute head vehicle velocity (m/s).
    """
    head_vel = np.full(total_steps, v_star)
    for k in range(total_steps):
        t = k * tstep
        if t >= settle_time:
            head_vel[k] = v_star + amplitude * np.sin(
                2.0 * np.pi / period * (t - settle_time)
            )
    return head_vel


def make_varying_sine(
    total_time: float,
    tstep: float,
    v_star: float,
    settle_time: float = 5.0,
    seed: int = 123,
) -> np.ndarray:
    """Sinusoidal with varying amplitude and frequency, changing at zero crossings.

    Each half-cycle (0 → +A → 0 or 0 → −A → 0) uses a randomly chosen amplitude
    and period from aggressive ranges.  Amplitude and frequency change at every
    zero crossing so the signal is continuous but non-stationary.

    Returns absolute head vehicle velocity (m/s).
    """
    total_steps = int(total_time / tstep)
    rng = np.random.default_rng(seed)

    amplitudes = [6.0, 9.0, 12.0, 15.0]
    periods = [8.0, 12.0, 18.0, 24.0, 30.0]

    head_vel = np.full(total_steps, v_star)

    # Build segments: each segment is one half-cycle (zero → peak → zero)
    t_cursor = settle_time
    sign = 1.0
    segments: list[tuple[float, float, float, float]] = []  # (t_start, duration, amp, sign)
    while t_cursor < total_time:
        amp = rng.choice(amplitudes)
        period = rng.choice(periods)
        half_period = period / 2.0
        segments.append((t_cursor, half_period, amp, sign))
        t_cursor += half_period
        sign *= -1.0

    # Fill the velocity array
    seg_idx = 0
    for k in range(total_steps):
        tk = k * tstep
        if tk < settle_time:
            continue
        # Advance to correct segment
        while (
            seg_idx < len(segments) - 1
            and tk >= segments[seg_idx][0] + segments[seg_idx][1]
        ):
            seg_idx += 1
        t_start, duration, amp, sgn = segments[seg_idx]
        local_t = tk - t_start
        head_vel[k] = v_star + sgn * amp * np.sin(np.pi * local_t / duration)

    # Clip to keep velocity non-negative
    np.clip(head_vel, 0.5, None, out=head_vel)
    return head_vel


def make_aggressive_sine(
    total_time: float,
    tstep: float,
    v_star: float,
    settle_time: float = 5.0,
    seed: int = 456,
) -> np.ndarray:
    """Aggressive varying sine: short periods that push past the QP horizon.

    Periods 1.5-4s mean quarter-periods of 0.375-1s — comparable to T_ini=1s.
    Amplitudes 4-10 m/s push head velocity into [5, 25] m/s, far from v_star.
    Changes occur at zero crossings so the signal stays continuous.

    Returns absolute head vehicle velocity (m/s).
    """
    total_steps = int(total_time / tstep)
    rng = np.random.default_rng(seed)

    amplitudes = [4.0, 6.0, 8.0, 10.0]
    periods = [1.5, 2.0, 3.0, 4.0]

    head_vel = np.full(total_steps, v_star)

    t_cursor = settle_time
    sign = 1.0
    segments: list[tuple[float, float, float, float]] = []
    while t_cursor < total_time:
        amp = rng.choice(amplitudes)
        period = rng.choice(periods)
        half_period = period / 2.0
        segments.append((t_cursor, half_period, amp, sign))
        t_cursor += half_period
        sign *= -1.0

    seg_idx = 0
    for k in range(total_steps):
        tk = k * tstep
        if tk < settle_time:
            continue
        while (
            seg_idx < len(segments) - 1
            and tk >= segments[seg_idx][0] + segments[seg_idx][1]
        ):
            seg_idx += 1
        t_start, duration, amp, sgn = segments[seg_idx]
        local_t = tk - t_start
        head_vel[k] = v_star + sgn * amp * np.sin(np.pi * local_t / duration)

    np.clip(head_vel, 0.5, None, out=head_vel)
    return head_vel


def make_stop_and_go(
    total_time: float,
    tstep: float,
    v_star: float,
    settle_time: float = 5.0,
    seed: int = 789,
) -> np.ndarray:
    """Stop-and-go traffic: repeated decel → stop → accel → cruise cycles.

    Head vehicle oscillates between 0 and 10-15 m/s.  The near-zero minimum
    pushes CAVs far from the DeeP-LCC linearization point (v_star=15).
    Phases are piecewise linear in velocity.

    Returns absolute head vehicle velocity (m/s).
    """
    total_steps = int(total_time / tstep)
    rng = np.random.default_rng(seed)

    head_vel = np.full(total_steps, v_star)
    settle_idx = int(settle_time / tstep)

    v = v_star
    k = settle_idx
    phase = 0  # 0=decel_to_0, 1=stopped, 2=accel_to_cruise, 3=cruise

    while k < total_steps:
        if phase == 0:
            a = -rng.uniform(1.5, 2.5)
            target = 0.0
            while v > target + 0.01 and k < total_steps:
                v = max(0.0, v + a * tstep)
                head_vel[k] = v
                k += 1
        elif phase == 1:
            hold_time = rng.uniform(1.5, 3.0)
            n = int(hold_time / tstep)
            for _ in range(n):
                if k >= total_steps:
                    break
                head_vel[k] = v
                k += 1
        elif phase == 2:
            a = rng.uniform(1.0, 2.0)
            target = rng.uniform(10.0, 15.0)
            while v < target - 0.01 and k < total_steps:
                v = v + a * tstep
                head_vel[k] = v
                k += 1
        else:
            hold_time = rng.uniform(2.0, 4.0)
            n = int(hold_time / tstep)
            for _ in range(n):
                if k >= total_steps:
                    break
                head_vel[k] = v
                k += 1
        phase = (phase + 1) % 4

    return head_vel


def make_nedc(tstep: float) -> np.ndarray:
    """NEDC head vehicle trajectory from reference .mat file (native length)."""
    mat_path = Path("deep_lcc_dataset/nedc_reference.mat")
    if mat_path.exists():
        from scipy.io import loadmat
        data = loadmat(str(mat_path))
        return data["S"][:, 0, 1].ravel()
    # Fallback: use the helper at a default 400s length
    return _make_nedc_perturbation(int(400.0 / tstep), tstep)


# ── Performance metrics ───────────────────────────────────────────────


def fuel_consumption_rate(v: float, a: float) -> float:
    """Instantaneous fuel consumption rate (mL/s) for a single vehicle.

    From paper eq. (in Section VI.A.2, ref [51] Bowyer 1985):
        R_i = 0.333 + 0.00108*v² + 1.200*a
        f_i = 0.444 + 0.090*R*v + max(0.054*a²*v, 0)  if R > 0
        f_i = 0.444                                    if R <= 0
    """
    R = 0.333 + 0.00108 * v**2 + 1.200 * a
    if R > 0:
        accel_term = 0.054 * a**2 * v if a > 0 else 0.0
        return 0.444 + 0.090 * R * v + accel_term
    return 0.444


def compute_metrics(
    qp_vel: dict[str, np.ndarray],
    head_vel: np.ndarray,
    config: DeepLCCConfig,
    full_state: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute performance metrics from a closed-loop simulation."""
    n_steps = len(head_vel)
    dt = config.Tstep

    # MSVE: mean squared velocity error vs head vehicle (for CAVs)
    msve_cav0 = np.mean((qp_vel["cav_0"] - head_vel) ** 2)
    msve_cav1 = np.mean((qp_vel["cav_1"] - head_vel) ** 2)
    msve_avg = 0.5 * (msve_cav0 + msve_cav1)

    metrics = {
        "duration_s": n_steps * dt,
        "n_steps": n_steps,
        "msve_cav0": msve_cav0,
        "msve_cav1": msve_cav1,
        "msve_avg": msve_avg,
        "max_cav0_vel": float(qp_vel["cav_0"].max()),
        "min_cav0_vel": float(qp_vel["cav_0"].min()),
        "max_cav1_vel": float(qp_vel["cav_1"].max()),
        "min_cav1_vel": float(qp_vel["cav_1"].min()),
    }

    # Fuel consumption requires full state (all 8 followers, with accelerations)
    if full_state is not None:
        # full_state shape: (n_steps, n_vehicle+1, 3)
        # Vehicles 3-8 in 1-indexed = indices 3..8 in (n_vehicle+1) array
        # (since index 0 is head vehicle)
        total_fuel = 0.0
        for i in range(3, 9):  # vehicles 3 through 8 (1-indexed)
            v_i = full_state[:, i, 1]
            a_i = full_state[:, i, 2]
            fuel_i = sum(fuel_consumption_rate(v_i[k], a_i[k]) * dt for k in range(n_steps))
            total_fuel += fuel_i
        metrics["total_fuel_mL"] = total_fuel
        metrics["fuel_per_vehicle_mL"] = total_fuel / 6.0

        s_cav0 = full_state[:, 2, 0] - full_state[:, 3, 0]
        s_cav1 = full_state[:, 5, 0] - full_state[:, 6, 0]

        collision_count = int(np.sum((s_cav0 <= 0.0) | (s_cav1 <= 0.0)))
        violation_count = int(np.sum((s_cav0 < 5.0) | (s_cav1 < 5.0)))

        a_cav0 = full_state[:, 3, 2]
        a_cav1 = full_state[:, 6, 2]
        aeb_cav0 = int(np.sum(np.isclose(a_cav0, -5.0)))
        aeb_cav1 = int(np.sum(np.isclose(a_cav1, -5.0)))

        metrics["collision_count"] = collision_count
        metrics["violation_count"] = violation_count
        metrics["aeb_trigger_count"] = aeb_cav0 + aeb_cav1
        metrics["min_spacing"] = float(min(s_cav0.min(), s_cav1.min()))
        metrics["max_spacing"] = float(max(s_cav0.max(), s_cav1.max()))
        metrics["failure_flag"] = collision_count > 0

    return metrics


# ── Closed-loop runner that also returns full state ──────────────────


def run_with_state(
    config: DeepLCCConfig,
    ovm_config: OVMConfig,
    Q: np.ndarray,
    R: np.ndarray,
    head_vel_abs: np.ndarray,
    controller_fn=None,
    seed: int = 999,
    noise_seed: int = 42,
    enable_aeb: bool = True,
    update_s_star: bool = True,
    collect_dataset: bool = False,
    ovm_resampler: Callable[[float], OVMConfig] | None = None,
    comm_delay_ms: float = 0.0,
    acel_noise: float | None = None,
    hdv_dynamics_fn: Callable[..., np.ndarray] | None = None,
) -> tuple[float, dict[str, np.ndarray], np.ndarray, dict[str, list] | None]:
    """Run closed-loop simulation and return cost, velocities, and full state.

    Args:
        controller_fn: Callable(uini, yini, eini) → cav_accel (m_ctr,).
            If None, uses the QP solver (builds Hankel matrices + solver internally).
        collect_dataset: If True, also returns a dict with lists of
            (uini, yini, eini, u_opt) pairs collected at each control step.

    Returns:
        (total_cost, velocities, full_state, dataset_pairs)
        dataset_pairs is None if collect_dataset is False.
    """
    from rl_mixed_traffic.deep_lcc.measurement import measure_mixed_traffic
    from rl_mixed_traffic.deep_lcc.ovm import hdv_dynamics as _default_hdv_dynamics
    import math

    n_vehicle = ovm_config.n_vehicle
    ID = ovm_config.ID
    pos_cav = np.where(np.array(ID) == 1)[0]
    m_ctr = len(pos_cav)
    p_ctr = n_vehicle + m_ctr
    total_steps = len(head_vel_abs)
    T_ini = config.T_ini
    delay_steps = int(round(comm_delay_ms / 1000.0 / config.Tstep))
    # Allow eval-time override of HDV acceleration noise (independent of precollect).
    acel_noise_level = config.acel_noise if acel_noise is None else acel_noise

    np.random.seed(noise_seed)

    if hdv_dynamics_fn is None:
        hdv_dynamics_fn = _default_hdv_dynamics

    # Build QP solver if no external controller provided
    if controller_fn is None:
        print("  Pre-collecting Hankel matrices...")
        Up, Uf, Ep, Ef, Yp, Yf = precollect(config, ovm_config, seed=seed)
        print("  Building QP solver...")
        solver = CachedDeepLCCSolver(
            Up, Yp, Uf, Yf, Ep, Ef, Q, R,
            config.lambda_g, config.lambda_y,
            u_limit=(config.dcel_max, config.acel_max),
            s_limit=(
                config.spacing_min - config.s_star,
                config.spacing_max - config.s_star,
            ),
        )

        def controller_fn(uini, yini, eini):
            u_opt, _, status = solver(uini, yini, eini)
            if status in ("optimal", "optimal_inaccurate"):
                return u_opt[:m_ctr]
            return np.zeros(m_ctr)

    v_star = config.v_star
    s_star = config.s_star

    S = np.zeros((total_steps, n_vehicle + 1, 3))
    S[0, 0, 0] = 0.0
    for i in range(1, n_vehicle + 1):
        S[0, i, 0] = S[0, i - 1, 0] - s_star
    S[0, :, 1] = v_star

    u_hist = np.zeros((m_ctr, total_steps))
    y_hist = np.zeros((p_ctr, total_steps))

    total_cost = 0.0

    if collect_dataset:
        ds_uini, ds_yini, ds_eini, ds_uopt = [], [], [], []

    for k in range(total_steps - 1):
        # Resolve OVM config for this step (time-varying if resampler provided).
        step_ovm = ovm_resampler(k * config.Tstep) if ovm_resampler is not None else ovm_config
        # HDV dynamics for all followers (CAVs will be overwritten below)
        acel = hdv_dynamics_fn(S[k], step_ovm)
        if acel_noise_level > 0.0:
            noise = -acel_noise_level + 2.0 * acel_noise_level * np.random.rand(n_vehicle)
            acel = acel + noise
        acel = np.clip(acel, ovm_config.dcel_max, ovm_config.acel_max)

        S[k, 0, 2] = 0.0
        S[k, 1:, 2] = acel

        # Record current step's measurement (with current equilibrium)
        y_hist[:, k] = measure_mixed_traffic(
            S[k, 1:, 1], S[k, :, 0], ID, v_star, s_star, config.measure_type
        )

        if k >= T_ini + delay_steps:
            # V2V delay: controller sees state from delay_steps ago.
            # k_eff is the "perceived present" from the controller's perspective.
            k_eff = k - delay_steps

            # Update equilibrium from past T_ini head velocities (not including k_eff)
            v_star = float(np.mean(S[k_eff - T_ini : k_eff, 0, 1]))
            if update_s_star:
                v_ratio = float(np.clip(v_star / ovm_config.v_max * 2, 0.0, 2.0))
                s_star = (
                    math.acos(1.0 - v_ratio) / math.pi
                    * (ovm_config.s_go - ovm_config.s_st)
                    + ovm_config.s_st
                )

            # Re-compute past T_ini-1 measurements with updated equilibrium
            for k_past in range(k_eff - T_ini + 1, k_eff):
                y_hist[:, k_past] = measure_mixed_traffic(
                    S[k_past, 1:, 1], S[k_past, :, 0], ID,
                    v_star, s_star, config.measure_type,
                )

            # Past data ending at k_eff-1 (delayed window)
            uini = u_hist[:, k_eff - T_ini : k_eff].flatten(order="F")
            yini = y_hist[:, k_eff - T_ini : k_eff].flatten(order="F")
            eini = S[k_eff - T_ini : k_eff, 0, 1] - v_star

            cav_accel = controller_fn(uini, yini, eini)

            if collect_dataset:
                ds_uini.append(uini.copy())
                ds_yini.append(yini.copy())
                ds_eini.append(eini.copy())
                ds_uopt.append(cav_accel[:m_ctr].copy())

            # Apply CAV control: overwrite HDV-computed accel for CAVs
            for j, cav_idx in enumerate(pos_cav):
                S[k, cav_idx + 1, 2] = float(cav_accel[j])

            # AEB override: if HDV dynamics commanded dcel_max for a CAV, force brake
            if enable_aeb:
                brake_ids = np.where(acel.ravel() == ovm_config.dcel_max)[0]
                brake_cavs = np.intersect1d(brake_ids, pos_cav)
                for cav_idx in brake_cavs:
                    S[k, cav_idx + 1, 2] = ovm_config.dcel_max
                    j = int(np.where(pos_cav == cav_idx)[0][0])
                    cav_accel[j] = ovm_config.dcel_max

            # Record the actually-applied CAV accelerations
            u_hist[:, k] = S[k, pos_cav + 1, 2]

            # Cost for this step
            y_k = y_hist[:, k]
            total_cost += float(y_k @ Q @ y_k + cav_accel @ R @ cav_accel)
        else:
            # Pre-control phase: CAVs follow HDV dynamics
            u_hist[:, k] = S[k, pos_cav + 1, 2]

        # Integrate dynamics
        S[k + 1, :, 1] = S[k, :, 1] + config.Tstep * S[k, :, 2]
        S[k + 1, 0, 1] = head_vel_abs[k + 1]
        S[k + 1, :, 0] = S[k, :, 0] + config.Tstep * S[k, :, 1]

    velocities = {"head": S[:, 0, 1].copy()}
    for j, cav_idx in enumerate(pos_cav):
        velocities[f"cav_{j}"] = S[:, cav_idx + 1, 1].copy()

    dataset_pairs = None
    if collect_dataset:
        dataset_pairs = {
            "uini": ds_uini,
            "yini": ds_yini,
            "eini": ds_eini,
            "u_opt": ds_uopt,
        }

    return total_cost, velocities, S, dataset_pairs


# ── Plotting ──────────────────────────────────────────────────────────


def plot_scenario(
    name: str,
    velocities: dict[str, np.ndarray],
    full_state: np.ndarray,
    config: DeepLCCConfig,
) -> None:
    """Plot velocities, accelerations, and spacing for a scenario."""
    n_steps = len(velocities["head"])
    t = np.arange(n_steps) * config.Tstep

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Velocities
    ax = axes[0]
    ax.plot(t, velocities["head"], color="gray", linewidth=1.5,
            linestyle="--", label="Head vehicle", alpha=0.8)
    ax.plot(t, velocities["cav_0"], color="#1f77b4", linewidth=1.5,
            label="CAV 3 (DeeP-LCC)")
    ax.plot(t, velocities["cav_1"], color="#d62728", linewidth=1.5,
            label="CAV 6 (DeeP-LCC)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"{name}: Classical DeeP-LCC Performance")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Accelerations of head vehicle and CAVs
    ax = axes[1]
    # Head vehicle acceleration: numerical derivative of velocity
    a_head = np.zeros(n_steps)
    a_head[1:] = (velocities["head"][1:] - velocities["head"][:-1]) / config.Tstep
    ax.plot(t, a_head, color="gray", linewidth=1.5, linestyle="--",
            label="Head vehicle", alpha=0.8)
    a_cav0 = full_state[:, 3, 2]
    a_cav1 = full_state[:, 6, 2]
    ax.plot(t, a_cav0, color="#1f77b4", linewidth=1.0, label="CAV 3")
    ax.plot(t, a_cav1, color="#d62728", linewidth=1.0, label="CAV 6")
    ax.axhline(config.acel_max, color="black", linestyle=":", alpha=0.4)
    ax.axhline(config.dcel_max, color="black", linestyle=":", alpha=0.4)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Spacing of CAVs (distance to vehicle in front)
    ax = axes[2]
    s_cav0 = full_state[:, 2, 0] - full_state[:, 3, 0]
    s_cav1 = full_state[:, 5, 0] - full_state[:, 6, 0]
    ax.plot(t, s_cav0, color="#1f77b4", linewidth=1.0, label="CAV 3 spacing")
    ax.plot(t, s_cav1, color="#d62728", linewidth=1.0, label="CAV 6 spacing")
    ax.axhline(config.spacing_min, color="black", linestyle=":", alpha=0.4,
               label="min/max")
    ax.axhline(config.spacing_max, color="black", linestyle=":", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spacing (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = name.lower().replace(" ", "_")
    out_path = f"deep_lcc_results/classical_{filename}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    config = DeepLCCConfig()
    config.acel_max = 3.0
    ovm_config = OVMConfig()

    n_vehicle = ovm_config.n_vehicle
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)

    Q_v = config.weight_v * np.eye(n_vehicle)
    Q_s = config.weight_s * np.eye(m_ctr)
    Q = np.block(
        [
            [Q_v, np.zeros((n_vehicle, m_ctr))],
            [np.zeros((m_ctr, n_vehicle)), Q_s],
        ]
    )
    R = config.weight_u * np.eye(m_ctr)

    print("=" * 60)
    print("Classical DeeP-LCC Evaluation on Extreme Scenarios")
    print("=" * 60)
    print(f"Config: lambda_g={config.lambda_g}, lambda_y={config.lambda_y}")
    print(f"        T_ini={config.T_ini}, N={config.N}, Tstep={config.Tstep}")
    print(f"        s_go={ovm_config.s_go}, s_st={ovm_config.s_st}")
    print()

    # ── Scenario 1: Extreme Braking ──
    print("─" * 60)
    print("Scenario 1: EXTREME BRAKING")
    print("─" * 60)
    print("  Head: 5s settle → -5 m/s² for 2s → 5s hold → +2 m/s² for 5s → cruise")
    print("  (matches reference brake_amp=10, fixed s_star=20, heterogeneous HDVs)")
    brake_steps = int(30.0 / config.Tstep)
    brake_head = make_extreme_brake(brake_steps, config.Tstep, config.v_star)
    print(f"  Steps: {brake_steps}, duration: {brake_steps * config.Tstep}s")

    # Use heterogeneous HDV parameters matching reference hdv_ovm_2.mat
    brake_ovm = get_heterogeneous_ovm_config()

    t0 = time.time()
    cost, vels, state, _ = run_with_state(
        config, brake_ovm, Q, R, brake_head, update_s_star=False
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    metrics = compute_metrics(vels, brake_head, config, full_state=state)
    metrics["total_cost"] = cost
    print(f"\n  Performance metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:25s} {v:12.3f}")
        else:
            print(f"    {k:25s} {v}")
    plot_scenario("extreme_brake", vels, state, config)
    print()

    # ── Scenario 2: Sinusoidal ──
    print("─" * 60)
    print("Scenario 2: SINUSOIDAL PERTURBATION")
    print("─" * 60)
    print("  Head: 5s settle → sine wave (amp=2 m/s, period=10s)")
    print("  (heterogeneous HDVs, fixed s_star)")
    sine_steps = int(40.0 / config.Tstep)
    sine_head = make_sinusoidal(
        sine_steps, config.Tstep, config.v_star, amplitude=2.0
    )
    print(f"  Steps: {sine_steps}, duration: {sine_steps * config.Tstep}s")

    sine_ovm = get_heterogeneous_ovm_config()

    t0 = time.time()
    cost, vels, state, _ = run_with_state(
        config, sine_ovm, Q, R, sine_head, update_s_star=False
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    metrics = compute_metrics(vels, sine_head, config, full_state=state)
    metrics["total_cost"] = cost
    print(f"\n  Performance metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:25s} {v:12.3f}")
        else:
            print(f"    {k:25s} {v}")
    plot_scenario("sinusoidal", vels, state, config)
    print()

    # ── Scenario 3: NEDC ──
    print("─" * 60)
    print("Scenario 3: NEDC (from reference .mat)")
    print("─" * 60)
    nedc_head = make_nedc(config.Tstep)  # length determined by .mat file
    nedc_steps = len(nedc_head)
    print(f"  Steps: {nedc_steps}, duration: {nedc_steps * config.Tstep}s")

    t0 = time.time()
    cost, vels, state, _ = run_with_state(config, ovm_config, Q, R, nedc_head)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    metrics = compute_metrics(vels, nedc_head, config, full_state=state)
    metrics["total_cost"] = cost
    print(f"\n  Performance metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:25s} {v:12.3f}")
        else:
            print(f"    {k:25s} {v}")
    plot_scenario("nedc", vels, state, config)
    print()

    print("=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
