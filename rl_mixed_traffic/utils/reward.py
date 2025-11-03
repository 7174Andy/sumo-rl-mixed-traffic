import numpy as np

def desired_velocity_state(obs: np.ndarray, *, v_max: float, target_v: float) -> float:
    """
    State-only version of Flow's desired_velocity.
    obs: [v_norm(0..n-1), p_norm(0..n-1)]
    Reward in [0, 1] where 1 when all speeds == target_v.
    """
    n = obs.size // 2
    v = obs[:n].astype(np.float32) * v_max
    if n == 0:
        return 0.0
    target = float(target_v)
    # Flow uses ||v - target||_2 relative to max deviation ||[target,...,target]||_2
    cost = np.linalg.norm(v - target)
    max_cost = np.linalg.norm(np.full(n, target))
    eps = np.finfo(np.float32).eps
    return float(max(max_cost - cost, 0.0) / (max_cost + eps))

def average_velocity_state(obs: np.ndarray, *, v_max: float) -> float:
    """
    State-only average velocity (m/s).
    """
    n = obs.size // 2
    v = obs[:n].astype(np.float32) * v_max
    return float(v.mean()) if n > 0 else 0.0

def penalize_standstill_state(obs: np.ndarray, *, v_max: float, thresh: float = 0.0, gain: float = 1.0) -> float:
    """
    Penalize vehicles with speed <= thresh (m/s). Returns negative penalty.
    """
    n = obs.size // 2
    v = obs[:n].astype(np.float32) * v_max
    num = int((v <= thresh).sum())
    return -gain * num

def min_delay_state(obs: np.ndarray, *, v_top: float, dt: float) -> float:
    """
    Rough state-only analog of min_delay: reward higher when speeds close to v_top.
    Returns a value in [0, 1] if v_top>0.
    """
    n = obs.size // 2
    if n == 0 or v_top <= 0:
        return 0.0
    v = obs[:n].astype(np.float32) * v_top  # treat v_max≈v_top for scaling
    eps = np.finfo(np.float32).eps
    max_cost = dt * n
    cost = dt * np.sum((v_top - v) / v_top)
    return float(max((max_cost - cost) / (max_cost + eps), 0.0))
