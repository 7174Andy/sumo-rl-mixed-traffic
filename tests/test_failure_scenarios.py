from typing import cast

import numpy as np
import pytest

from rl_mixed_traffic.deep_lcc.failure_scenarios import SEVERITY_RANGES


class TestSeverityRanges:
    def test_mild_ranges(self):
        r = SEVERITY_RANGES["mild"]
        assert r["alpha"] == (0.4, 0.8)
        assert r["beta"] == (0.6, 1.0)
        assert r["s_go"] == (30.0, 40.0)

    def test_medium_ranges(self):
        r = SEVERITY_RANGES["medium"]
        assert r["alpha"] == (0.3, 0.9)
        assert r["beta"] == (0.5, 1.2)
        assert r["s_go"] == (25.0, 45.0)

    def test_severe_ranges(self):
        r = SEVERITY_RANGES["severe"]
        assert r["alpha"] == (0.2, 1.0)
        assert r["beta"] == (0.4, 1.5)
        assert r["s_go"] == (20.0, 50.0)

    def test_all_severities_present(self):
        assert set(SEVERITY_RANGES.keys()) == {"mild", "medium", "severe"}


from rl_mixed_traffic.deep_lcc.config import OVMConfig
from rl_mixed_traffic.deep_lcc.failure_scenarios import make_ovm_resampler


class TestMakeOvmResampler:
    def test_returns_callable(self):
        resampler = make_ovm_resampler("mild", seed=0)
        assert callable(resampler)

    def test_returns_ovm_config(self):
        resampler = make_ovm_resampler("mild", seed=0)
        cfg = resampler(0.0)
        assert isinstance(cfg, OVMConfig)

    def test_cav_positions_stay_nominal(self):
        """CAVs at ID positions 3 and 6 (1-indexed) = indices 2, 5 (0-indexed)
        must always carry nominal (0.6, 0.9, 35)."""
        resampler = make_ovm_resampler("severe", seed=0)
        for t in [0.0, 5.0, 17.3, 42.7, 95.0]:
            cfg = resampler(t)
            alpha, beta, s_go = cfg.alpha, cfg.beta, cfg.s_go
            assert isinstance(alpha, list)
            assert isinstance(beta, list)
            assert isinstance(s_go, list)
            assert len(alpha) == 8
            for cav_idx in (2, 5):
                assert alpha[cav_idx] == pytest.approx(0.6)
                assert beta[cav_idx] == pytest.approx(0.9)
                assert s_go[cav_idx] == pytest.approx(35.0)

    def test_hdv_positions_within_severity_range(self):
        """Non-CAV HDVs have parameters drawn from the severity distribution."""
        resampler = make_ovm_resampler("mild", seed=123)
        hdv_indices = [0, 1, 3, 4, 6, 7]  # non-CAV positions
        # Sample many times to cover multiple resample events
        for t in np.linspace(0.0, 200.0, 50):
            cfg = resampler(t)
            alpha, beta, s_go = cfg.alpha, cfg.beta, cfg.s_go
            assert isinstance(alpha, list)
            assert isinstance(beta, list)
            assert isinstance(s_go, list)
            for i in hdv_indices:
                assert 0.4 <= alpha[i] <= 0.8
                assert 0.6 <= beta[i] <= 1.0
                assert 30.0 <= s_go[i] <= 40.0

    def test_unknown_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            make_ovm_resampler("catastrophic", seed=0)


class TestResamplerCadence:
    def test_params_change_over_time(self):
        """Parameters should change as t advances past resample boundaries."""
        resampler = make_ovm_resampler("severe", seed=0)
        alpha_0 = cast(list[float], resampler(0.0).alpha)
        # After 30s (> max resample period of 25s), every HDV has resampled at least once
        alpha_30 = cast(list[float], resampler(30.0).alpha)
        hdv_indices = [0, 1, 3, 4, 6, 7]
        # At least one HDV's parameters changed (statistically, essentially all of them)
        any_changed = any(alpha_0[i] != alpha_30[i] for i in hdv_indices)
        assert any_changed

    def test_independent_clocks(self):
        """Different HDVs resample at different times (clocks are independent)."""
        resampler = make_ovm_resampler("severe", seed=0)
        # Walk t in 0.5s ticks and record when each HDV's alpha changes.
        hdv_indices = [0, 1, 3, 4, 6, 7]
        prev_alpha = cast(list[float], resampler(0.0).alpha)
        change_times: dict[int, list[float]] = {i: [] for i in hdv_indices}
        for step in range(1, 400):  # 0 .. 200s
            t = step * 0.5
            alpha = cast(list[float], resampler(t).alpha)
            for i in hdv_indices:
                if alpha[i] != prev_alpha[i]:
                    change_times[i].append(t)
            prev_alpha = alpha
        # Each HDV should have 5-13 resample events over 200s
        # (mean period ~20s → ~10 events)
        for i in hdv_indices:
            assert 5 <= len(change_times[i]) <= 15, (
                f"HDV {i} changed {len(change_times[i])} times in 200s"
            )
        # Flatten all change times — they should not all be identical
        all_times = {t for times in change_times.values() for t in times}
        assert len(all_times) > 20, "HDVs appear to share a clock"

    def test_inter_resample_gap_in_range(self):
        """Gaps between consecutive resamples of the same HDV are in [15, 25] s."""
        resampler = make_ovm_resampler("medium", seed=7)
        # Sample at fine time resolution
        prev_alpha = cast(list[float], resampler(0.0).alpha)
        change_times: dict[int, list[float]] = {i: [0.0] for i in [0, 1, 3, 4, 6, 7]}
        for step in range(1, 2000):  # 0.1s ticks, 0..200s
            t = step * 0.1
            alpha = cast(list[float], resampler(t).alpha)
            for i in [0, 1, 3, 4, 6, 7]:
                if alpha[i] != prev_alpha[i]:
                    change_times[i].append(t)
            prev_alpha = alpha
        for i, times in change_times.items():
            if len(times) < 2:
                continue
            gaps = np.diff(times[1:])  # skip t=0 initial sample
            # Each gap in [15, 25] s (allow 0.1s tolerance for the sampling grid)
            assert np.all((gaps >= 14.9) & (gaps <= 25.1)), (
                f"HDV {i} gaps out of range: {gaps}"
            )

    def test_reproducibility_with_same_seed(self):
        """Same seed → identical trajectories."""
        r1 = make_ovm_resampler("medium", seed=42)
        r2 = make_ovm_resampler("medium", seed=42)
        for t in np.linspace(0.0, 100.0, 20):
            cfg1 = r1(t)
            cfg2 = r2(t)
            assert cfg1.alpha == cfg2.alpha
            assert cfg1.beta == cfg2.beta
            assert cfg1.s_go == cfg2.s_go
