"""Hard-constraint safety layer that clips unsafe accelerations.

Uses a simple physics model to predict spacing after applying an acceleration,
and restricts the acceleration if it would violate the minimum spacing threshold.

Pure math — no SUMO dependency — so it's easily unit-testable.
"""

import numpy as np


class SafetyLayer:
    """Clips accelerations that would violate minimum spacing constraints.

    Physics model (constant-acceleration over one timestep):
        predicted_spacing = spacing + relative_vel * dt - 0.5 * accel * dt^2

    If predicted_spacing < s_min, compute the maximum safe acceleration:
        safe_accel = 2 * (spacing - s_min + relative_vel * dt) / dt^2

    The filter only restricts (never accelerates more aggressively), so:
        filtered_accel = min(accel, safe_accel)

    Args:
        s_min: Minimum allowed spacing in meters (bumper-to-bumper).
        dt: Simulation timestep in seconds.
    """

    def __init__(self, s_min: float = 5.0, dt: float = 0.1):
        self.s_min = s_min
        self.dt = dt

    def filter(
        self,
        accel: float,
        spacing: float,
        relative_vel: float,
    ) -> tuple[float, bool]:
        """Filter an acceleration command to maintain minimum spacing.

        Args:
            accel: Requested acceleration (m/s^2). Positive = speed up.
            spacing: Current bumper-to-bumper gap to leader (m).
            relative_vel: v_leader - v_ego (m/s). Negative means closing.

        Returns:
            (safe_accel, was_clipped): The (possibly reduced) acceleration
            and whether clipping occurred.
        """
        dt = self.dt

        # Predict spacing after one timestep if we apply the requested accel.
        # Ego moves: 0.5 * accel * dt^2 extra relative to constant-speed baseline.
        # Leader moves: relative_vel * dt relative to ego (positive = opening).
        predicted_spacing = spacing + relative_vel * dt - 0.5 * accel * dt**2

        if predicted_spacing >= self.s_min:
            return accel, False

        # Compute the maximum accel that keeps predicted_spacing == s_min:
        #   s_min = spacing + relative_vel * dt - 0.5 * safe_accel * dt^2
        #   safe_accel = 2 * (spacing - s_min + relative_vel * dt) / dt^2
        safe_accel = 2.0 * (spacing - self.s_min + relative_vel * dt) / dt**2

        # Only restrict — never allow more aggressive accel than requested.
        filtered = min(accel, safe_accel)
        was_clipped = filtered != accel

        return filtered, was_clipped
