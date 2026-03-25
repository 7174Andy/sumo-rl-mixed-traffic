import numpy as np


def measure_mixed_traffic(
    vel: np.ndarray,
    pos: np.ndarray,
    ID: list[int],
    v_star: float,
    s_star: float,
    measure_type: int = 3,
) -> np.ndarray:
    """Measure the output vector in mixed traffic flow.

    Port of measure_mixed_traffic.m from the DeeP-LCC MATLAB implementation.

    Args:
        vel: Velocities of following vehicles (excluding head), shape (n_vehicle,).
        pos: Positions of ALL vehicles (including head), shape (n_vehicle + 1,).
        ID: Vehicle type IDs (1=CAV, 0=HDV), length n_vehicle.
        v_star: Equilibrium velocity.
        s_star: Equilibrium spacing.
        measure_type:
            1 — velocity errors only
            2 — all velocity + all spacing errors
            3 — all velocity errors + CAV spacing errors only

    Returns:
        y: Output vector. Shape depends on measure_type:
            1 → (n_vehicle,)
            2 → (2 * n_vehicle,)
            3 → (n_vehicle + n_cav,)
    """
    pos_cav = np.where(np.array(ID) == 1)[0]

    if measure_type == 1:
        y = vel - v_star
    elif measure_type == 2:
        spacing = pos[:-1] - pos[1:]
        y = np.concatenate([vel - v_star, spacing - s_star])
    elif measure_type == 3:
        spacing = pos[:-1] - pos[1:]
        y = np.concatenate([vel - v_star, spacing[pos_cav] - s_star])
    else:
        raise ValueError(f"Unknown measure_type: {measure_type}")

    return y
