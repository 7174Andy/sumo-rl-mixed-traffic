"""Pure-math tests for EmergencyBrakingController.get_target_speed().

No SUMO needed — only validates the four-phase speed profile.
"""

import pytest
import math

# Patch out traci before importing the controller module, since the module
# does a top-level ``import traci`` which requires SUMO_HOME.
import types
import sys
import os

# Set a dummy SUMO_HOME so the module can be imported without SUMO installed.
os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")

_fake_traci = types.ModuleType("traci")
_fake_traci.vehicle = types.SimpleNamespace(
    getIDList=lambda: [],
    setSpeed=lambda *a, **kw: None,
)
sys.modules.setdefault("traci", _fake_traci)

from rl_mixed_traffic.env.head_vehicle_controller import EmergencyBrakingController


# ---------- defaults used in all tests unless overridden ----------
DEFAULTS = dict(
    v_cruise=15.0,
    v_low=2.0,
    t_brake=50.0,
    brake_duration=3.0,
    hold_duration=10.0,
    recover_duration=5.0,
)


def _ctrl(**overrides):
    kw = {**DEFAULTS, **overrides}
    return EmergencyBrakingController(head_id="car0", step_length=0.1, **kw)


# ---- Phase 1: cruise ----

def test_cruise_at_t0():
    c = _ctrl()
    assert c.get_target_speed(0.0) == pytest.approx(15.0)


def test_cruise_just_before_brake():
    c = _ctrl()
    assert c.get_target_speed(49.9) == pytest.approx(15.0)


# ---- Phase 2: braking ----

def test_brake_start():
    c = _ctrl()
    # At t=50.0, frac=0 → still v_cruise
    assert c.get_target_speed(50.0) == pytest.approx(15.0)


def test_brake_midpoint():
    c = _ctrl()
    # t=51.5 → frac=0.5 → midpoint between 15 and 2 = 8.5
    assert c.get_target_speed(51.5) == pytest.approx(8.5)


def test_brake_end():
    c = _ctrl()
    # t=52.999... → frac≈1 → ~v_low
    assert c.get_target_speed(53.0 - 1e-9) == pytest.approx(2.0, abs=0.01)


# ---- Phase 3: hold ----

def test_hold_start():
    c = _ctrl()
    # t=53.0 → hold phase begins
    assert c.get_target_speed(53.0) == pytest.approx(2.0)


def test_hold_midpoint():
    c = _ctrl()
    assert c.get_target_speed(58.0) == pytest.approx(2.0)


def test_hold_end():
    c = _ctrl()
    # t=62.999... → still hold
    assert c.get_target_speed(63.0 - 1e-9) == pytest.approx(2.0)


# ---- Phase 4: recover ----

def test_recover_start():
    c = _ctrl()
    # t=63.0 → frac=0 → v_low
    assert c.get_target_speed(63.0) == pytest.approx(2.0)


def test_recover_midpoint():
    c = _ctrl()
    # t=65.5 → frac=0.5 → midpoint between 2 and 15 = 8.5
    assert c.get_target_speed(65.5) == pytest.approx(8.5)


def test_recover_end():
    c = _ctrl()
    # t=67.999... → frac≈1 → ~v_cruise
    assert c.get_target_speed(68.0 - 1e-9) == pytest.approx(15.0, abs=0.01)


# ---- After recovery: cruise indefinitely ----

def test_post_recovery_cruise():
    c = _ctrl()
    assert c.get_target_speed(68.0) == pytest.approx(15.0)
    assert c.get_target_speed(100.0) == pytest.approx(15.0)
    assert c.get_target_speed(1000.0) == pytest.approx(15.0)


# ---- Properties and reset ----

def test_update_every_step_is_true():
    c = _ctrl()
    assert c.update_every_step is True


def test_reset_zeros_sim_time():
    c = _ctrl()
    c._sim_time = 99.0
    c.reset()
    assert c._sim_time == 0.0


# ---- Custom parameters ----

def test_custom_parameters():
    c = _ctrl(v_cruise=20.0, v_low=5.0, t_brake=10.0,
              brake_duration=2.0, hold_duration=5.0, recover_duration=3.0)
    # Phase 1
    assert c.get_target_speed(5.0) == pytest.approx(20.0)
    # Phase 2 midpoint: t=11.0, frac=0.5 → 20 + 0.5*(5-20) = 12.5
    assert c.get_target_speed(11.0) == pytest.approx(12.5)
    # Phase 3
    assert c.get_target_speed(14.0) == pytest.approx(5.0)
    # Phase 4 midpoint: t3=17, t=18.5, frac=0.5 → 5 + 0.5*(20-5) = 12.5
    assert c.get_target_speed(18.5) == pytest.approx(12.5)
    # After recovery (t4=20)
    assert c.get_target_speed(25.0) == pytest.approx(20.0)


# ---- Base class defaults ----

def test_base_controller_update_every_step_false():
    from rl_mixed_traffic.env.head_vehicle_controller import HeadVehicleController
    c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
    assert c.update_every_step is False


def test_base_controller_reset_noop():
    from rl_mixed_traffic.env.head_vehicle_controller import HeadVehicleController
    c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
    c.reset()  # should not raise
