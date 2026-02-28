"""Pure-math tests for SafetyLayer (no SUMO required)."""

import pytest
import math

from rl_mixed_traffic.env.safety_layer import SafetyLayer


@pytest.fixture
def layer():
    return SafetyLayer(s_min=5.0, dt=0.1)


class TestNoClip:
    """Cases where the requested accel should pass through unchanged."""

    def test_large_gap_no_clip(self, layer):
        """With a large gap, even strong acceleration should not be clipped."""
        accel, clipped = layer.filter(accel=3.0, spacing=50.0, relative_vel=0.0)
        assert accel == 3.0
        assert clipped is False

    def test_opening_gap_no_clip(self, layer):
        """Leader pulling away (positive relative_vel) — no clip needed."""
        accel, clipped = layer.filter(accel=2.0, spacing=6.0, relative_vel=5.0)
        assert accel == 2.0
        assert clipped is False

    def test_braking_passes_through(self, layer):
        """Strong braking with large gap should pass through unchanged."""
        # spacing=20 >> s_min=5, so even with closing, predicted spacing stays safe
        accel, clipped = layer.filter(accel=-5.0, spacing=20.0, relative_vel=-2.0)
        assert accel == -5.0
        assert clipped is False


class TestClipping:
    """Cases where the safety layer must intervene."""

    def test_closing_fast_clips(self, layer):
        """Closing gap with acceleration should be clipped."""
        # spacing=6, relative_vel=-10 (closing fast), accel=3 (accelerating into leader)
        accel, clipped = layer.filter(accel=3.0, spacing=6.0, relative_vel=-10.0)
        assert clipped is True
        assert accel < 3.0

    def test_small_gap_accelerating_clips(self, layer):
        """Near s_min gap with positive accel should clip.

        With dt=0.1, 0.5*a*dt^2 is small, so use a gap very close to s_min.
        predicted = 5.005 + 0 - 0.5*3*0.01 = 5.005 - 0.015 = 4.99 < 5.0
        """
        accel, clipped = layer.filter(accel=3.0, spacing=5.005, relative_vel=0.0)
        assert clipped is True
        assert accel < 3.0

    def test_clipped_accel_preserves_s_min(self, layer):
        """The clipped acceleration should produce predicted_spacing == s_min."""
        spacing = 6.0
        relative_vel = -10.0
        dt = layer.dt

        safe_accel, clipped = layer.filter(accel=3.0, spacing=spacing, relative_vel=relative_vel)
        assert clipped is True

        # Verify physics: predicted spacing with safe_accel should be >= s_min
        predicted = spacing + relative_vel * dt - 0.5 * safe_accel * dt**2
        assert predicted == pytest.approx(layer.s_min, abs=1e-9)


class TestPhysicsCorrectness:
    """Verify the physics model produces exact s_min when clipping."""

    def test_safe_accel_produces_exact_s_min(self, layer):
        """safe_accel should produce predicted_spacing == s_min exactly.

        With dt=0.1, need spacing close enough to s_min that large accel causes violation.
        spacing=5.5, rel_vel=-5 → predicted = 5.5 - 0.5 - 0.5*a*0.01
        For a=100: predicted = 5.0 - 0.5 = 4.5 < 5.0 → clips.
        """
        spacing = 5.5
        relative_vel = -5.0
        dt = layer.dt

        # Manually compute safe_accel
        safe_accel = 2.0 * (spacing - layer.s_min + relative_vel * dt) / dt**2

        # Use a large requested accel to force clipping
        filtered, clipped = layer.filter(accel=100.0, spacing=spacing, relative_vel=relative_vel)
        assert clipped is True
        assert filtered == pytest.approx(safe_accel, abs=1e-9)

        # Verify predicted spacing
        predicted = spacing + relative_vel * dt - 0.5 * filtered * dt**2
        assert predicted == pytest.approx(layer.s_min, abs=1e-9)

    def test_no_clip_when_exactly_at_threshold(self, layer):
        """If predicted spacing == s_min exactly, no clipping should occur."""
        dt = layer.dt
        spacing = 6.0
        relative_vel = 0.0
        # Compute accel that yields predicted_spacing == s_min exactly
        accel_boundary = 2.0 * (spacing - layer.s_min) / dt**2

        filtered, clipped = layer.filter(accel=accel_boundary, spacing=spacing, relative_vel=relative_vel)
        assert clipped is False
        assert filtered == pytest.approx(accel_boundary)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_spacing_already_below_s_min(self, layer):
        """When spacing is already below s_min, even zero accel may be clipped."""
        # spacing=3 < s_min=5, relative_vel=0 → predicted = 3 - 0 = 3 < 5
        # safe_accel = 2*(3-5+0)/0.01 = -400 → very strong braking required
        accel, clipped = layer.filter(accel=0.0, spacing=3.0, relative_vel=0.0)
        assert clipped is True
        assert accel < 0.0  # must brake

    def test_custom_s_min(self):
        """SafetyLayer should work with custom s_min values."""
        layer = SafetyLayer(s_min=10.0, dt=0.1)
        # predicted = 10.005 - 0.5*3*0.01 = 10.005 - 0.015 = 9.99 < 10.0
        accel, clipped = layer.filter(accel=3.0, spacing=10.005, relative_vel=0.0)
        assert clipped is True

    def test_custom_dt(self):
        """SafetyLayer should work with custom dt values."""
        layer = SafetyLayer(s_min=5.0, dt=0.5)
        # With larger dt, predictions look further ahead
        accel, clipped = layer.filter(accel=3.0, spacing=6.0, relative_vel=-2.0)
        assert clipped is True

    def test_only_restricts_never_increases(self, layer):
        """filtered accel should always be <= requested accel."""
        test_cases = [
            (3.0, 6.0, -10.0),
            (-1.0, 4.0, -5.0),
            (0.0, 5.0, 0.0),
            (1.0, 20.0, 0.0),
        ]
        for req_accel, spacing, rel_vel in test_cases:
            filtered, _ = layer.filter(accel=req_accel, spacing=spacing, relative_vel=rel_vel)
            assert filtered <= req_accel, (
                f"Safety layer increased accel: {filtered} > {req_accel} "
                f"(spacing={spacing}, rel_vel={rel_vel})"
            )
