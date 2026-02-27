"""Tests for make_head_controller() factory function.

Verifies correct controller creation for each scenario type.
"""

import pytest
import types
import sys
import os

os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")

_fake_traci = types.ModuleType("traci")
_fake_traci.vehicle = types.SimpleNamespace(
    getIDList=lambda: [],
    setSpeed=lambda *a, **kw: None,
)
sys.modules.setdefault("traci", _fake_traci)

from rl_mixed_traffic.env.scenario import make_head_controller
from rl_mixed_traffic.env.head_vehicle_controller import EmergencyBrakingController


class _Cfg:
    """Minimal config object mimicking Hydra DictConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestRandomScenario:
    def test_returns_none(self):
        ctrl = make_head_controller(_Cfg(type="random"))
        assert ctrl is None

    def test_default_type_is_random(self):
        """If config has no type attribute, defaults to random -> None."""
        ctrl = make_head_controller(_Cfg())
        assert ctrl is None


class TestEmergencyBrakingScenario:
    def test_returns_emergency_braking_controller(self):
        ctrl = make_head_controller(_Cfg(type="emergency_braking"))
        assert isinstance(ctrl, EmergencyBrakingController)

    def test_default_parameters(self):
        ctrl = make_head_controller(_Cfg(type="emergency_braking"))
        assert ctrl.v_cruise == 15.0
        assert ctrl.v_low == 2.0
        assert ctrl.t_brake == 50.0
        assert ctrl.brake_duration == 3.0
        assert ctrl.hold_duration == 10.0
        assert ctrl.recover_duration == 5.0

    def test_custom_parameters(self):
        ctrl = make_head_controller(
            _Cfg(
                type="emergency_braking",
                v_cruise=20.0,
                v_low=1.0,
                t_brake=30.0,
                brake_duration=5.0,
                hold_duration=8.0,
                recover_duration=4.0,
            )
        )
        assert ctrl.v_cruise == 20.0
        assert ctrl.v_low == 1.0
        assert ctrl.t_brake == 30.0
        assert ctrl.brake_duration == 5.0
        assert ctrl.hold_duration == 8.0
        assert ctrl.recover_duration == 4.0

    def test_custom_head_id(self):
        ctrl = make_head_controller(
            _Cfg(type="emergency_braking"), head_id="leader"
        )
        assert ctrl.head_id == "leader"

    def test_custom_step_length(self):
        ctrl = make_head_controller(
            _Cfg(type="emergency_braking"), step_length=0.05
        )
        assert ctrl.step_length == 0.05


class TestUnknownScenario:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown scenario type"):
            make_head_controller(_Cfg(type="highway_merge"))

    def test_error_message_includes_type(self):
        with pytest.raises(ValueError, match="'nonexistent'"):
            make_head_controller(_Cfg(type="nonexistent"))
