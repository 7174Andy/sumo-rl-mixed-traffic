"""Hard-constraint safety layer that clips unsafe accelerations.

Uses a simple physics model to predict spacing after applying an acceleration,
and restricts the acceleration if it would violate spacing thresholds.

Two constraints are enforced:
1. s_min (too close to physical leader): caps max accel — clips DOWN
2. s_max (too far from head vehicle):   caps min accel — clips UP

When both constraints conflict (s_min wants lower accel, s_max wants higher),
s_min takes priority (collision avoidance > formation keeping).

Pure math — no SUMO dependency — so it's easily unit-testable.
"""



class SafetyLayer:
    """Clips accelerations that would violate spacing constraints.

    Physics model (constant-acceleration over one timestep):
        predicted_spacing = spacing + relative_vel * dt - 0.5 * accel * dt^2

    s_min constraint (too close to leader):
        If predicted_spacing < s_min → cap accel from above.
        safe_max_accel = 2 * (spacing - s_min + relative_vel * dt) / dt^2

    s_max constraint (too far from head vehicle):
        If predicted_gap_to_head > s_max → cap accel from below.
        safe_min_accel = 2 * (gap_to_head - s_max + rel_vel_head * dt) / dt^2

    Args:
        s_min: Minimum allowed spacing in meters (bumper-to-bumper).
        s_max: Maximum allowed gap to head vehicle in meters. None to disable.
        dt: Simulation timestep in seconds.
    """

    def __init__(self, s_min: float = 5.0, s_max: float | None = None, dt: float = 0.1):
        self.s_min = s_min
        self.s_max = s_max
        self.dt = dt

    def filter(
        self,
        accel: float,
        spacing: float,
        relative_vel: float,
        gap_to_head: float | None = None,
        rel_vel_head: float | None = None,
    ) -> tuple[float, bool]:
        """Filter an acceleration command to maintain spacing constraints.

        Args:
            accel: Requested acceleration (m/s^2). Positive = speed up.
            spacing: Current bumper-to-bumper gap to physical leader (m).
            relative_vel: v_leader - v_ego (m/s). Negative means closing.
            gap_to_head: Current gap to head vehicle (m). None to skip s_max check.
            rel_vel_head: v_head - v_ego (m/s). Required if gap_to_head is given.

        Returns:
            (safe_accel, was_clipped): The (possibly adjusted) acceleration
            and whether clipping occurred.
        """
        dt = self.dt
        was_clipped = False
        filtered = accel

        # --- s_max: too far from head vehicle → force acceleration (clip UP) ---
        # Apply s_max first so s_min can override if they conflict.
        if (
            self.s_max is not None
            and gap_to_head is not None
            and rel_vel_head is not None
        ):
            predicted_gap_head = gap_to_head + rel_vel_head * dt - 0.5 * filtered * dt**2
            if predicted_gap_head > self.s_max:
                # Minimum accel to keep predicted_gap_head == s_max:
                #   s_max = gap_to_head + rel_vel_head * dt - 0.5 * a * dt^2
                #   a = 2 * (gap_to_head - s_max + rel_vel_head * dt) / dt^2
                min_accel = 2.0 * (gap_to_head - self.s_max + rel_vel_head * dt) / dt**2
                if filtered < min_accel:
                    filtered = min_accel
                    was_clipped = True

        # --- s_min: too close to leader → restrict acceleration (clip DOWN) ---
        # s_min takes priority: if both conflict, we prevent collision.
        predicted_spacing = spacing + relative_vel * dt - 0.5 * filtered * dt**2
        if predicted_spacing < self.s_min:
            safe_max_accel = 2.0 * (spacing - self.s_min + relative_vel * dt) / dt**2
            if filtered > safe_max_accel:
                filtered = safe_max_accel
                was_clipped = True

        return filtered, was_clipped
