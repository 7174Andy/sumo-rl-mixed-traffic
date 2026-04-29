from dataclasses import dataclass, field
from typing import Literal

from rl_mixed_traffic.configs.ppo_config import PPOConfig


@dataclass
class RLMPCConfig:
    """Configuration for RLMPC training (warm-start RL or RL+NNMPC residual)."""

    # Mode
    mode: Literal["warm_start", "residual"] = "warm_start"

    # Paths
    nnmpc_path: str = "deep_lcc_results/nnmpc.pth"
    sumocfg_path: str = "configs/ring/platoon_9.sumocfg"
    out_dir: str = "deep_lcc_results/rlmpc_{mode}/"

    # SUMO
    use_gui: bool = False

    # Episode
    episode_length_s: float = 30.0
    Tstep: float = 0.05  # must match step-length in sumocfg

    # Platoon
    n_vehicle: int = 9          # 1 head + 8 followers
    n_followers: int = 8
    cav_positions: tuple[int, ...] = (3, 6)  # 1-indexed within platoon
    v_star: float = 15.0
    s_star: float = 20.0
    v_max: float = 30.0
    accel_max: float = 3.0
    accel_min: float = -5.0

    # Cost weights (match DeepLCCConfig)
    weight_v: float = 5.0
    weight_s: float = 0.1
    weight_u: float = 0.1
    spacing_min: float = 5.0
    spacing_max: float = 40.0

    # NNMPC architecture
    T_ini: int = 20
    obs_dim: int = 260   # T_ini * (m_ctr + p_ctr + 1) = 20 * (2 + 10 + 1)
    action_dim: int = 2  # m_ctr (number of CAVs)

    # Action bounds by mode (residual is ±2 m/s² added to NNMPC, then clipped)
    residual_max: float = 2.0

    # Lagrangian penalty weight
    lambda_violation: float = 1.0

    # PPO hyperparams (delegated)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Training loop
    total_steps: int = 1_000_000
    rollout_steps: int = 4096
    save_freq: int = 100_000

    # Warm-start specifics
    log_std_init_warm: float = -2.302585  # log(0.1)
    log_std_init_residual: float = -0.693147  # log(0.5)
    final_layer_gain_warm: float = 1.0
    final_layer_gain_residual: float = 0.01

    # Optional critic-only warm-up updates (paper risk mitigation).
    # Field reserved; implementation deferred until divergence is observed.
    critic_warmup_updates: int = 0

    # HDV randomization (per-vehicle, per-episode), applied via traci.vehicle.set*
    hdv_tau_range: tuple[float, float] = (0.8, 1.5)
    hdv_accel_range: tuple[float, float] = (1.5, 2.5)
    hdv_decel_range: tuple[float, float] = (2.5, 3.5)
    hdv_minGap_range: tuple[float, float] = (2.0, 3.0)
    hdv_sigma_range: tuple[float, float] = (0.3, 0.6)

    # Seed
    seed: int = 42

    @property
    def max_steps(self) -> int:
        return int(self.episode_length_s / self.Tstep)

    @property
    def m_ctr(self) -> int:
        return len(self.cav_positions)

    @property
    def p_ctr(self) -> int:
        # measurement_type=3: n_followers velocity errors + m_ctr spacing errors
        return self.n_followers + self.m_ctr
