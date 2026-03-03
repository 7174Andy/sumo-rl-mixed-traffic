"""Unit tests for non-negative DeeP-LCC reward functions and spacing violations.

The reward is r = max(J_max - J, 0) / J_max  where J >= 0 is the raw cost.
At equilibrium (J=0) reward is 1.0; at worst case (J=J_max) reward is 0.0.

Mocks traci calls to test the reward math in isolation (no SUMO needed).
"""

import pytest
import numpy as np
from unittest.mock import patch

from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.configs.sumo_config import SumoConfig


# ---------------------------------------------------------------------------
# Fake traci state
# ---------------------------------------------------------------------------
_vehicle_data = {}  # vid -> {speed, distance, length}
_leader_data = {}   # vid -> (leader_id, gap) or None
_active_ids = []


def _setup_vehicles(vehicles: dict, leaders: dict):
    """Populate fake traci state."""
    global _active_ids
    _vehicle_data.clear()
    _leader_data.clear()
    _vehicle_data.update(vehicles)
    _leader_data.update(leaders)
    _active_ids = list(vehicles.keys())


@pytest.fixture(autouse=True)
def _mock_traci():
    """Patch all traci.vehicle calls used by the reward methods."""
    with (
        patch("traci.vehicle.getIDList", side_effect=lambda: list(_active_ids)),
        patch("traci.vehicle.getSpeed", side_effect=lambda vid: _vehicle_data[vid]["speed"]),
        patch("traci.vehicle.getDistance", side_effect=lambda vid: _vehicle_data[vid]["distance"]),
        patch("traci.vehicle.getLength", side_effect=lambda vid: _vehicle_data[vid]["length"]),
        patch("traci.vehicle.getLeader", side_effect=lambda vid: _leader_data.get(vid)),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RING_LENGTH = 230.0
CAR_LENGTH = 5.0


def _make_env(**overrides):
    """Create a RingRoadEnv without connecting to SUMO."""
    defaults = dict(
        v_star=15.0,
        s_star=20.0,
        weight_v=0.8,
        weight_s=0.7,
        weight_u=0.1,
        num_vehicles=4,
        num_agents=1,
    )
    defaults.update(overrides)

    cfg = SumoConfig(sumocfg_path="configs/ring/simulation.sumocfg", use_gui=False)
    env = RingRoadEnv(sumo_config=cfg, gui=False, **defaults)
    env.ring_length = RING_LENGTH
    return env


# ---------------------------------------------------------------------------
# Tests: compute_lcc_reward (single-agent)
# ---------------------------------------------------------------------------
class TestSingleAgentLCCReward:

    def test_agent_absent_returns_zero(self):
        env = _make_env()
        _setup_vehicles(
            {"car0": {"speed": 15.0, "distance": 0.0, "length": 5.0}},
            {},
        )
        assert env.compute_lcc_reward() == 0.0

    def test_at_equilibrium_reward_is_one(self):
        """When all vehicles at v_star and gap == s_star, reward == 1.0."""
        env = _make_env()
        s_star = env.s_star  # 20.0
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 150.0, "length": 5.0},
                "car3": {"speed": 15.0, "distance": 200.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        assert reward == pytest.approx(1.0, abs=1e-6)

    def test_velocity_error_reduces_reward(self):
        """Vehicles deviating from v_star produce reward < 1."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 10.0, "distance": 150.0, "length": 5.0},
                "car3": {"speed": 20.0, "distance": 200.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        # J_vel = 0.8*(25+25) = 40.0
        expected = max(env.J_max - 40.0, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)
        assert 0.0 < reward < 1.0

    def test_spacing_error_reduces_reward(self):
        """Gap deviating from s_star reduces reward."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 0.0

        gap = s_star + 5.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        # J_spacing = 0.7 * 25 = 17.5
        expected = max(env.J_max - 17.5, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_control_penalty_reduces_reward(self):
        """Non-zero acceleration reduces reward."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 2.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        # J_control = 0.1 * 4 = 0.4
        expected = max(env.J_max - 0.4, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_combined_cost(self):
        """All three cost terms contribute together."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 1.0

        gap = s_star + 3.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 12.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        # J_vel = 0.8*9 = 7.2, J_spacing = 0.7*9 = 6.3, J_ctrl = 0.1*1 = 0.1
        J = 7.2 + 6.3 + 0.1
        expected = max(env.J_max - J, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_spacing_error_clipped(self):
        """Spacing error is clipped to [-20, 20] before squaring."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 0.0

        gap = s_star + 50.0  # error=50, clipped to +20

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        # J_spacing = 0.7 * 400 = 280
        expected = max(env.J_max - 280.0, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_reward_is_always_nonnegative(self):
        """Non-negative reward: always in [0, 1]."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 1.5

        gap = s_star + 8.0

        _setup_vehicles(
            {
                "car0": {"speed": 10.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 18.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 12.0, "distance": 150.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        assert 0.0 <= reward <= 1.0

    def test_reward_bounded_above_by_one(self):
        """Reward never exceeds 1.0."""
        env = _make_env()
        s_star = env.s_star
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        assert reward <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_multi_agent_lcc_reward
# ---------------------------------------------------------------------------
class TestMultiAgentLCCReward:

    def test_at_equilibrium_reward_is_one(self):
        env = _make_env(num_agents=2)
        s_star = env.s_star  # 20.0
        env.prev_accels = {"car1": 0.0, "car2": 0.0}

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 75.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car3": {"speed": 15.0, "distance": 150.0, "length": 5.0},
            },
            {
                "car1": ("car0", s_star),
                "car2": ("car1", s_star),
            },
        )

        reward = env.compute_multi_agent_lcc_reward()
        assert reward == pytest.approx(1.0, abs=1e-6)

    def test_sums_spacing_across_agents(self):
        """Spacing error summed across all CAVs."""
        env = _make_env(num_agents=2)
        s_star = env.s_star
        env.prev_accels = {"car1": 0.0, "car2": 0.0}

        gap1 = s_star + 4.0  # 24
        gap2 = s_star + 3.0  # 23

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 71.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 43.0, "length": 5.0},
            },
            {
                "car1": ("car0", gap1),
                "car2": ("car1", gap2),
            },
        )

        reward = env.compute_multi_agent_lcc_reward()
        # J_spacing = 0.7 * (16 + 9) = 17.5
        expected = max(env.J_max_multi - 17.5, 0.0) / env.J_max_multi
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_sums_control_across_agents(self):
        """Control penalty summed across all CAVs."""
        env = _make_env(num_agents=2)
        s_star = env.s_star
        env.prev_accels = {"car1": 2.0, "car2": 1.0}

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 75.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {
                "car1": ("car0", s_star),
                "car2": ("car1", s_star),
            },
        )

        reward = env.compute_multi_agent_lcc_reward()
        # J_control = 0.1 * (4 + 1) = 0.5
        expected = max(env.J_max_multi - 0.5, 0.0) / env.J_max_multi
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_inactive_agent_skipped(self):
        """Agent not in active_ids doesn't contribute to reward."""
        env = _make_env(num_agents=2)
        s_star = env.s_star
        env.prev_accels = {"car1": 0.0, "car2": 3.0}

        # car2 absent
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_multi_agent_lcc_reward()
        assert reward == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: get_spacing_violation (s_min + s_max Lagrangian constraint)
# ---------------------------------------------------------------------------
class TestSpacingViolation:

    def test_no_violation_at_equilibrium(self):
        """Gap within both bounds → zero violation."""
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        # car1 at 50, car0 at 75 → gap_to_head = (75-50)%230 - 5 = 20
        # leader gap = 20 (from mock)
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 75.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 50.0, "length": CAR_LENGTH},
            },
            {"car1": ("car0", 20.0)},
        )

        assert env.get_spacing_violation() == pytest.approx(0.0)

    def test_smin_violation_only(self):
        """Physical leader too close → s_min violation."""
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        # leader gap = 3.0 (too close, violates s_min)
        # gap_to_head = (58-50)%230 - 5 = 3.0 (within s_max)
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 58.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 50.0, "length": CAR_LENGTH},
            },
            {"car1": ("car0", 3.0)},
        )

        violation = env.get_spacing_violation()
        # s_min violation: max(0, 5 - 3) / 5 = 0.4
        # s_max violation: max(0, 3 - 40) / 40 = 0.0
        assert violation == pytest.approx(0.4)

    def test_smax_violation_only(self):
        """Exploit scenario: HDV ahead at ~s_star, car0 far away."""
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        # car1 (CAV) at 10, car2 (HDV) at 35, car0 at 200
        # Physical leader is car2 at gap=20 (fine for s_min)
        # gap_to_head = (200-10)%230 - 5 = 185 (exceeds s_max)
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 200.0, "length": CAR_LENGTH},
                "car1": {"speed": 2.0, "distance": 10.0, "length": CAR_LENGTH},
                "car2": {"speed": 2.0, "distance": 35.0, "length": CAR_LENGTH},
            },
            {"car1": ("car2", 20.0)},
        )

        violation = env.get_spacing_violation()
        # s_min violation: max(0, 5 - 20) / 5 = 0
        # s_max violation: max(0, 185 - 40) / 40 = 3.625
        assert violation == pytest.approx(3.625)

    def test_combined_smin_and_smax(self):
        """Both constraints violated simultaneously (multi-agent)."""
        env = _make_env(num_agents=2, spacing_min=5.0, spacing_max=40.0)

        # car1: leader gap=3 (violates s_min), gap_to_head=3 (within s_max)
        # car2: leader gap=20 (fine), gap_to_head=180 (violates s_max)
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 58.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 50.0, "length": CAR_LENGTH},
                "car2": {"speed": 2.0, "distance": 100.0, "length": CAR_LENGTH},
                "car3": {"speed": 2.0, "distance": 125.0, "length": CAR_LENGTH},
            },
            {
                "car1": ("car0", 3.0),
                "car2": ("car3", 20.0),
            },
        )

        violation = env.get_spacing_violation()
        # car1: s_min=max(0,5-3)/5=0.4, s_max=max(0, 3-40)/40=0
        # car2: s_min=max(0,5-20)/5=0, s_max=max(0, 183-40)/40=3.575
        # total = 0.4 + 3.575 = 3.975
        assert violation == pytest.approx(3.975)

    def test_no_violation_at_smax_boundary(self):
        """Gap exactly at s_max → zero violation."""
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        # gap_to_head = (95-50)%230 - 5 = 40.0 (exactly s_max)
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 95.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 50.0, "length": CAR_LENGTH},
            },
            {"car1": ("car0", 20.0)},
        )

        violation = env.get_spacing_violation()
        # s_min: max(0, 5-20) = 0
        # s_max: max(0, 40-40) = 0
        assert violation == pytest.approx(0.0)

    def test_multi_agent_violations_sum(self):
        """Violations sum across all agents."""
        env = _make_env(num_agents=2, spacing_min=5.0, spacing_max=40.0)

        # Both agents too far from car0
        # car1 at 10, car0 at 100: gap_to_head = (100-10)%230 - 5 = 85
        # car2 at 50, car0 at 100: gap_to_head = (100-50)%230 - 5 = 45
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 10.0, "distance": 10.0, "length": CAR_LENGTH},
                "car2": {"speed": 10.0, "distance": 50.0, "length": CAR_LENGTH},
            },
            {
                "car1": ("car2", 20.0),
                "car2": ("car0", 20.0),
            },
        )

        violation = env.get_spacing_violation()
        # car1: s_min=max(0,5-20)/5=0, s_max=max(0, 85-40)/40=1.125
        # car2: s_min=max(0,5-20)/5=0, s_max=max(0, 45-40)/40=0.125
        # total = 1.25
        assert violation == pytest.approx(1.25)

    def test_inactive_agent_skipped(self):
        """Absent agent doesn't contribute to violation."""
        env = _make_env(num_agents=2, spacing_min=5.0, spacing_max=40.0)

        # car2 absent from simulation
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 75.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 50.0, "length": CAR_LENGTH},
            },
            {"car1": ("car0", 20.0)},
        )

        violation = env.get_spacing_violation()
        # Only car1: s_min=0, s_max=max(0, 20-40)=0
        assert violation == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: CAV overtakes car0 and becomes the "leader" on the ring
# ---------------------------------------------------------------------------
class TestCAVAsLeader:
    """Test behaviour when the CAV is positioned ahead of car0 on the ring.

    On a one-directional ring, being 10 m "ahead" of car0 means the
    *forward* distance to car0 wraps around the entire ring (~220 m on
    a 230 m ring).  Both `_get_gap_to_leader()` and `_get_gap_to_head()`
    measure forward distance, so they report a near-full-ring gap.
    """

    # -- low-level gap helpers --

    def test_find_ring_leader_fallback_selects_car0(self):
        """When getLeader returns None, fallback finds car0 via positions.

        CAV at 110, car0 at 100 (CAV is 10 m ahead).
        Forward distance = (100 - 110) % 230 = 220 m.
        Bumper gap = 220 - 5 = 215 m.
        """
        env = _make_env()

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},  # no getLeader data → forces fallback path
        )

        lead_id, gap = env._find_ring_leader("car1")
        assert lead_id == "car0"
        assert gap == pytest.approx(RING_LENGTH - 10.0 - CAR_LENGTH)  # 215.0

    def test_get_gap_to_leader_wraps_when_cav_ahead(self):
        """_get_gap_to_leader returns the wrap-around gap to car0."""
        env = _make_env()

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},  # fallback path
        )

        gap = env._get_gap_to_leader("car1")
        assert gap == pytest.approx(215.0)

    def test_get_gap_to_head_wraps_when_cav_ahead(self):
        """_get_gap_to_head also wraps: forward from CAV to car0."""
        env = _make_env()

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},
        )

        gap = env._get_gap_to_head("car1")
        # (100 - 110) % 230 - 5 = 215
        assert gap == pytest.approx(215.0)

    def test_leader_is_closer_hdv_not_car0(self):
        """With an HDV between CAV and car0 (forward), leader is the HDV.

        Ring layout (positions on 230 m ring):
            car1 (CAV) at 110, car2 (HDV) at 150, car0 at 100
        Forward distances from car1:
            to car2: (150 - 110) % 230 = 40 m   ← closer
            to car0: (100 - 110) % 230 = 220 m
        """
        env = _make_env()

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
                "car2": {"speed": 15.0, "distance": 150.0, "length": CAR_LENGTH},
            },
            {},  # fallback path
        )

        lead_id, gap = env._find_ring_leader("car1")
        assert lead_id == "car2"
        assert gap == pytest.approx(40.0 - CAR_LENGTH)  # 35.0

    # -- reward --

    def test_reward_drops_when_cav_overtakes_car0(self):
        """CAV ahead of car0 → large wrap-around leader gap → reduced reward.

        Leader gap ≈ 215 m, spacing error = 215 - 20 = 195, clipped to +20.
        J_spacing = 0.7 * 20² = 280. Reward drops but stays positive because
        the ±20 clip limits the spacing penalty.  The real deterrent is the
        Lagrangian s_max violation (tested separately).
        """
        env = _make_env()
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},  # fallback: leader = car0, gap ≈ 215
        )

        reward = env.compute_lcc_reward()
        # J_spacing = 0.7 * clip(215 - 20, -20, 20)² = 0.7 * 400 = 280
        J_spacing = env.weight_s * 20.0 ** 2  # 280
        expected = max(env.J_max - J_spacing, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)
        assert reward < 1.0, "reward drops below equilibrium"
        assert reward > 0.0, "but base reward stays positive (clip limits penalty)"

    def test_reward_with_hdv_between_cav_and_car0(self):
        """CAV ahead of car0 but an HDV sits between them in forward dir.

        Leader is the HDV at gap 35 m.  Spacing error = 35 - 20 = 15.
        Reward penalised but less severely than the no-HDV case.
        """
        env = _make_env()
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
                "car2": {"speed": 15.0, "distance": 150.0, "length": CAR_LENGTH},
            },
            {},  # fallback: leader = car2, gap = 35
        )

        reward = env.compute_lcc_reward()
        # J_spacing = 0.7 * (35 - 20)² = 0.7 * 225 = 157.5
        J_spacing = env.weight_s * 15.0 ** 2  # 157.5
        expected = max(env.J_max - J_spacing, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_reward_still_nonneg_when_cav_is_leader(self):
        """Even with max spacing penalty + velocity errors, reward >= 0."""
        env = _make_env()
        env.prev_accel = 3.0  # max accel → control penalty too

        _setup_vehicles(
            {
                "car0": {"speed": 5.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 25.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},  # leader = car0 via wrap-around
        )

        reward = env.compute_lcc_reward()
        assert 0.0 <= reward <= 1.0

    # -- spacing violation --

    def test_smax_violation_when_cav_overtakes(self):
        """CAV barely ahead of car0 → gap_to_head wraps → large s_max violation.

        CAV at 110, car0 at 100: gap_to_head = (100-110)%230 - 5 = 215.
        s_max violation = max(0, 215 - 40) / 40 = 4.375.
        """
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},  # fallback: leader = car0, gap = 215 (fine for s_min)
        )

        violation = env.get_spacing_violation()
        # s_min: max(0, 5 - 215) / 5 = 0
        # s_max: max(0, 215 - 40) / 40 = 4.375
        assert violation == pytest.approx(4.375)

    def test_smax_violation_with_hdv_between(self):
        """HDV between CAV and car0 in forward dir: leader gap is small,
        but gap_to_head still wraps → s_max fires.

        car1 at 110, car2 at 150, car0 at 100.
        leader = car2, gap = 35 (no s_min issue).
        gap_to_head = (100-110)%230 - 5 = 215 → s_max = 4.375.
        """
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
                "car2": {"speed": 15.0, "distance": 150.0, "length": CAR_LENGTH},
            },
            {},  # fallback: leader = car2 at gap 35
        )

        violation = env.get_spacing_violation()
        # s_min: max(0, 5 - 35) / 5 = 0
        # s_max: max(0, 215 - 40) / 40 = 4.375
        assert violation == pytest.approx(4.375)

    def test_lagrangian_penalty_makes_overtake_strongly_negative(self):
        """Demonstrate the Lagrangian penalty crushes the base reward.

        Base reward ≈ 0.66 (spacing error clipped to 20).
        Normalised violation = 4.375.
        lambda = 1.0 → augmented = 0.66 - 1.0 * 4.375 = -3.72.
        """
        env = _make_env(spacing_min=5.0, spacing_max=40.0)
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 110.0, "length": CAR_LENGTH},
            },
            {},
        )

        r_base = env.compute_lcc_reward()
        violation = env.get_spacing_violation()
        lambda_val = 1.0

        r_augmented = r_base - lambda_val * violation

        assert r_base > 0.0, "base reward is still positive"
        assert violation > 4.0, "normalised violation is significant"
        assert r_augmented < 0.0, "augmented reward is strongly negative"

    def test_no_violation_when_cav_just_behind_car0(self):
        """CAV behind car0 by 25 m — normal following, no overtake.

        car1 at 75, car0 at 100: gap_to_head = (100-75)%230 - 5 = 20.
        This is within s_max (40), so no s_max violation.
        """
        env = _make_env(spacing_min=5.0, spacing_max=40.0)

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 100.0, "length": CAR_LENGTH},
                "car1": {"speed": 15.0, "distance": 75.0, "length": CAR_LENGTH},
            },
            {"car1": ("car0", 20.0)},
        )

        violation = env.get_spacing_violation()
        assert violation == pytest.approx(0.0)
