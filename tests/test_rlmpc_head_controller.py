"""Tests for PerturbMixHeadController."""

import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

# Stub SUMO_HOME and traci so the module imports without SUMO installed.
os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")
_fake_traci = types.ModuleType("traci")
_fake_traci.vehicle = types.SimpleNamespace(
    getIDList=lambda: ["car0"],
    setSpeed=lambda vid, speed: None,
)
sys.modules.setdefault("traci", _fake_traci)


from rl_mixed_traffic.deep_lcc.rlmpc_head_controller import (
    PerturbMixHeadController,
)


class TestSamplePerturbation:
    def test_random_perturbation_within_amplitude(self):
        ctrl = PerturbMixHeadController(
            head_id="car0",
            tstep=0.05,
            episode_length_s=10.0,
            v_star=15.0,
            seed=1,
        )
        trace = ctrl.sample_perturbation(("random", 1.0, 1.0), seed=1)
        assert trace.shape == (200,)
        assert (trace >= 14.0 - 1e-6).all()
        assert (trace <= 16.0 + 1e-6).all()

    def test_brake_perturbation_dips_then_recovers(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=30.0,
            v_star=15.0, seed=2,
        )
        trace = ctrl.sample_perturbation(("brake", 0.0, 1.0), seed=2)
        # Trace should reach v_low ≤ 5 m/s during the brake phase
        assert trace.min() <= 5.0
        # And recover toward v_star at the end
        assert abs(trace[-1] - 15.0) < 2.0

    def test_sinusoidal_perturbation_amplitude(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=20.0,
            v_star=15.0, seed=3,
        )
        trace = ctrl.sample_perturbation(("sinusoidal", 5.0, 1.0), seed=3)
        # Amplitude ≈ 5 m/s
        assert abs(trace.max() - 20.0) < 1.5
        assert abs(trace.min() - 10.0) < 1.5

    def test_unknown_type_raises(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=10.0,
            v_star=15.0, seed=1,
        )
        with pytest.raises(ValueError):
            ctrl.sample_perturbation(("garbage", 1.0, 1.0), seed=1)


class TestResetTraceSelection:
    def test_reset_picks_one_type_from_mix(self):
        # Mix with only "random" entries → reset must pick "random".
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=10.0,
            v_star=15.0, seed=42,
            perturb_mix=[("random", 1.0, 0.5), ("random", 3.0, 0.5)],
        )
        ctrl.reset()
        assert ctrl.trace is not None
        assert ctrl.trace.shape == (200,)
        # All values in [12, 18] (worst case amp=3)
        assert (ctrl.trace >= 11.99).all()
        assert (ctrl.trace <= 18.01).all()


class TestSetRandomHeadSpeed:
    def test_calls_setspeed_with_trace_value(self):
        ctrl = PerturbMixHeadController(
            head_id="car0", tstep=0.05, episode_length_s=2.0,
            v_star=15.0, seed=1,
            perturb_mix=[("random", 0.0, 1.0)],   # constant trace at v_star
        )
        ctrl.reset()

        with patch("traci.vehicle.setSpeed") as mock_set:
            ctrl.set_random_head_speed()  # called every step in update_every_step mode
            mock_set.assert_called_once()
            args, _ = mock_set.call_args
            assert args[0] == "car0"
            assert abs(args[1] - 15.0) < 1e-6
