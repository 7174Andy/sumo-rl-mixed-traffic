from dataclasses import dataclass, field


@dataclass
class OVMConfig:
    """Optimal Velocity Model parameters (homogeneous case, data_str=3)."""

    alpha: float = 0.6
    beta: float = 0.9
    v_max: float = 30.0
    s_st: float = 5.0
    s_go: float = 25.0
    n_vehicle: int = 8
    ID: list[int] = field(default_factory=lambda: [0, 0, 1, 0, 0, 1, 0, 0])
    acel_max: float = 2.0
    dcel_max: float = -5.0


@dataclass
class DeepLCCConfig:
    """DeeP-LCC algorithm and data generation parameters."""

    # Pre-collection
    T: int = 2000
    acel_noise: float = 0.1

    # Horizons
    T_ini: int = 20
    N: int = 50

    # Equilibrium
    v_star: float = 15.0
    s_star: float = 20.0

    # Cost weights
    weight_v: float = 1.0
    weight_s: float = 0.5
    weight_u: float = 0.1

    # Regularisation
    lambda_g: float = 1.0
    lambda_y: float = 1e3

    # Constraints
    acel_max: float = 2.0
    dcel_max: float = -5.0
    spacing_min: float = 5.0
    spacing_max: float = 40.0

    # Measurement type (3 = all velocity errors + CAV spacing errors)
    measure_type: int = 3

    # Simulation
    Tstep: float = 0.05
    total_time: float = 40.0

    # Dataset generation
    num_episodes: int = 100
    output_path: str = "deep_lcc_dataset/dataset.npz"
    # Perturbation amplitude mix: list of (amplitude, fraction) pairs.
    # Each episode's head-vehicle perturbation is uniform in
    # [-amplitude, +amplitude] m/s around v_star.
    perturb_mix: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 0.5), (3.0, 0.3), (5.0, 0.2)]
    )

    def __post_init__(self) -> None:
        total = sum(frac for _, frac in self.perturb_mix)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"perturb_mix fractions must sum to 1.0, got {total}"
            )
