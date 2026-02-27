"""Tests for the base HeadVehicleController class.

Patches traci.vehicle at the module level so no SUMO connection is required.
"""

from unittest.mock import patch, MagicMock

import pytest

from rl_mixed_traffic.env.head_vehicle_controller import HeadVehicleController


@pytest.fixture(autouse=True)
def mock_traci_vehicle():
    """Patch traci.vehicle in the head_vehicle_controller module for every test."""
    mock_vehicle = MagicMock()
    with patch("rl_mixed_traffic.env.head_vehicle_controller.traci.vehicle", mock_vehicle):
        yield mock_vehicle


class TestInit:
    def test_stores_head_id(self):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        assert c.head_id == "car0"

    def test_stores_speed_limits(self):
        c = HeadVehicleController(head_id="car0", head_speed_min=3, head_speed_max=25)
        assert c.head_speed_min == 3
        assert c.head_speed_max == 25


class TestUpdateEveryStep:
    def test_returns_false(self):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        assert c.update_every_step is False


class TestReset:
    def test_is_noop(self):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.reset()  # should not raise


class TestSetHeadSpeed:
    def test_sets_speed_within_range(self, mock_traci_vehicle):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_head_speed(10.0)
        mock_traci_vehicle.setSpeed.assert_called_once_with("car0", 10.0)

    def test_clamps_speed_below_min(self, mock_traci_vehicle):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_head_speed(2.0)
        mock_traci_vehicle.setSpeed.assert_called_once_with("car0", 5)

    def test_clamps_speed_above_max(self, mock_traci_vehicle):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_head_speed(30.0)
        mock_traci_vehicle.setSpeed.assert_called_once_with("car0", 20)

    def test_clamps_at_exact_min(self, mock_traci_vehicle):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_head_speed(5.0)
        mock_traci_vehicle.setSpeed.assert_called_once_with("car0", 5.0)

    def test_clamps_at_exact_max(self, mock_traci_vehicle):
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_head_speed(20.0)
        mock_traci_vehicle.setSpeed.assert_called_once_with("car0", 20.0)


class TestSetRandomHeadSpeed:
    def test_sets_speed_when_vehicle_present(self, mock_traci_vehicle):
        mock_traci_vehicle.getIDList.return_value = ["car0", "car1"]
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_random_head_speed()
        mock_traci_vehicle.setSpeed.assert_called_once()
        call_args = mock_traci_vehicle.setSpeed.call_args
        assert call_args[0][0] == "car0"
        speed = call_args[0][1]
        assert 5 <= speed <= 20

    def test_noop_when_vehicle_absent(self, mock_traci_vehicle):
        mock_traci_vehicle.getIDList.return_value = ["car1", "car2"]
        c = HeadVehicleController(head_id="car0", head_speed_min=5, head_speed_max=20)
        c.set_random_head_speed()
        mock_traci_vehicle.setSpeed.assert_not_called()

    def test_speed_varies_across_calls(self, mock_traci_vehicle):
        """Multiple calls should generally produce different speeds (not deterministic)."""
        mock_traci_vehicle.getIDList.return_value = ["car0"]
        c = HeadVehicleController(head_id="car0", head_speed_min=0, head_speed_max=100)
        speeds = set()
        for _ in range(20):
            mock_traci_vehicle.setSpeed.reset_mock()
            c.set_random_head_speed()
            speed = mock_traci_vehicle.setSpeed.call_args[0][1]
            speeds.add(round(speed, 2))
        # With range [0, 100] and 20 calls, we'd expect multiple distinct values
        assert len(speeds) > 1
