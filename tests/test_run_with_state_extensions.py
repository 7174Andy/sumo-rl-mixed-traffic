import numpy as np
import pytest

from rl_mixed_traffic.deep_lcc.config import DeepLCCConfig, OVMConfig
from rl_mixed_traffic.deep_lcc.eval_classical import compute_metrics, run_with_state


@pytest.fixture
def small_config():
    """Tiny config so the test runs fast."""
    return DeepLCCConfig(T=300, T_ini=5, N=10, total_time=2.0, Tstep=0.05)


@pytest.fixture
def ovm_config():
    return OVMConfig()


@pytest.fixture
def weights(small_config, ovm_config):
    n_vehicle = ovm_config.n_vehicle
    pos_cav = np.where(np.array(ovm_config.ID) == 1)[0]
    m_ctr = len(pos_cav)
    Q_v = small_config.weight_v * np.eye(n_vehicle)
    Q_s = small_config.weight_s * np.eye(m_ctr)
    Q = np.block(
        [
            [Q_v, np.zeros((n_vehicle, m_ctr))],
            [np.zeros((m_ctr, n_vehicle)), Q_s],
        ]
    )
    R = small_config.weight_u * np.eye(m_ctr)
    return Q, R


def _zero_controller(uini, yini, eini):
    """Stub controller: always return zero acceleration."""
    return np.zeros(2)


class TestOvmResamplerParam:
    def test_default_none_matches_static_config(
        self, small_config, ovm_config, weights
    ):
        """Without ovm_resampler, behavior must match the pre-existing path."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star)
        cost_a, vel_a, state_a, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
        )
        cost_b, vel_b, state_b, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            ovm_resampler=None,  # explicit None
        )
        np.testing.assert_allclose(state_a, state_b)
        assert cost_a == cost_b

    def test_resampler_called_per_step(self, small_config, ovm_config, weights):
        """When ovm_resampler is provided, it is invoked each step."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star)
        call_times: list[float] = []

        def spy_resampler(t: float) -> OVMConfig:
            call_times.append(t)
            return ovm_config

        run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            ovm_resampler=spy_resampler,
        )
        # Called roughly once per step (40 steps → >30 calls accounting for
        # the t=0 init step)
        assert len(call_times) >= 35
        # Time values monotonically increase
        assert all(call_times[i] <= call_times[i + 1] for i in range(len(call_times) - 1))

    def test_resampler_affects_dynamics(self, small_config, ovm_config, weights):
        """Swapping OVM params mid-run changes the HDV trajectory."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star + 1.0)  # small perturbation

        aggressive = OVMConfig(alpha=0.3, beta=1.5, s_go=25.0)

        def aggressive_resampler(t: float) -> OVMConfig:
            return aggressive

        _, vel_static, _, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
        )
        _, vel_dyn, _, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            ovm_resampler=aggressive_resampler,
        )
        # At least one HDV's velocity trajectory differs
        assert not np.allclose(vel_static["cav_0"], vel_dyn["cav_0"])


class TestCommDelayParam:
    def test_default_zero_matches_no_delay(self, small_config, ovm_config, weights):
        """comm_delay_ms=0 must match the no-delay path exactly."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star)
        cost_a, _, state_a, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
        )
        cost_b, _, state_b, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            comm_delay_ms=0.0,
        )
        np.testing.assert_allclose(state_a, state_b)
        assert cost_a == cost_b

    def test_delay_postpones_first_control_step(
        self, small_config, ovm_config, weights
    ):
        """With delay=2 steps, the first controller call happens 2 steps later."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star)

        call_counts = {"no_delay": 0, "delay": 0}

        def ctrl_no_delay(uini, yini, eini):
            call_counts["no_delay"] += 1
            return np.zeros(2)

        def ctrl_delay(uini, yini, eini):
            call_counts["delay"] += 1
            return np.zeros(2)

        run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=ctrl_no_delay, enable_aeb=False,
        )
        run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=ctrl_delay, enable_aeb=False,
            comm_delay_ms=100.0,  # 2 steps at Tstep=0.05s
        )
        # Delay should reduce total controller calls by delay_steps=2
        assert call_counts["no_delay"] - call_counts["delay"] == 2

    def test_delay_affects_dynamics_under_control(
        self, small_config, ovm_config, weights
    ):
        """With a non-trivial controller, delay changes the resulting trajectory."""
        Q, R = weights
        head_vel = np.linspace(
            small_config.v_star, small_config.v_star + 2.0, 40
        )

        # A controller that reacts to yini (not just zero) — simple proportional
        def prop_ctrl(uini, yini, eini):
            # React to mean velocity error (first n_vehicle entries of each yini block)
            return np.full(2, -0.5 * float(np.mean(eini)))

        _, _, state_nodelay, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=prop_ctrl, enable_aeb=False,
        )
        _, _, state_delayed, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=prop_ctrl, enable_aeb=False,
            comm_delay_ms=150.0,
        )
        assert not np.allclose(state_nodelay, state_delayed)


class TestComputeMetricsExtensions:
    def _synth_state(
        self, n_steps: int, n_vehicle: int, spacings: np.ndarray, accels: np.ndarray
    ) -> np.ndarray:
        S = np.zeros((n_steps, n_vehicle + 1, 3))
        for k in range(n_steps):
            S[k, 0, 0] = 0.0
            S[k, 0, 1] = 15.0
            pos = 0.0
            for i in range(1, n_vehicle + 1):
                gap = 20.0
                if i == 3:
                    gap = spacings[k, 0]
                elif i == 6:
                    gap = spacings[k, 1]
                pos -= gap
                S[k, i, 0] = pos
                S[k, i, 1] = 15.0
            S[k, 1:, 2] = accels[k]
        return S

    def _vels(self, n_steps: int) -> dict[str, np.ndarray]:
        return {
            "head": np.full(n_steps, 15.0),
            "cav_0": np.full(n_steps, 15.0),
            "cav_1": np.full(n_steps, 15.0),
        }

    def test_collision_count_zero_for_safe_run(self, small_config):
        n_steps, n_vehicle = 40, 8
        spacings = np.full((n_steps, 2), 20.0)
        accels = np.zeros((n_steps, n_vehicle))
        state = self._synth_state(n_steps, n_vehicle, spacings, accels)
        vels = self._vels(n_steps)
        m = compute_metrics(vels, vels["head"], small_config, full_state=state)
        assert m["collision_count"] == 0
        assert m["violation_count"] == 0
        assert m["aeb_trigger_count"] == 0
        assert m["failure_flag"] is False

    def test_collision_count_nonzero(self, small_config):
        n_steps, n_vehicle = 40, 8
        spacings = np.full((n_steps, 2), 20.0)
        spacings[10:15, 0] = -0.5
        accels = np.zeros((n_steps, n_vehicle))
        state = self._synth_state(n_steps, n_vehicle, spacings, accels)
        vels = self._vels(n_steps)
        m = compute_metrics(vels, vels["head"], small_config, full_state=state)
        assert m["collision_count"] == 5
        assert m["violation_count"] >= 5
        assert m["failure_flag"] is True

    def test_violation_count_spacing_below_5(self, small_config):
        n_steps, n_vehicle = 40, 8
        spacings = np.full((n_steps, 2), 20.0)
        spacings[20:30, 1] = 3.0
        accels = np.zeros((n_steps, n_vehicle))
        state = self._synth_state(n_steps, n_vehicle, spacings, accels)
        vels = self._vels(n_steps)
        m = compute_metrics(vels, vels["head"], small_config, full_state=state)
        assert m["collision_count"] == 0
        assert m["violation_count"] == 10

    def test_aeb_trigger_count(self, small_config):
        n_steps, n_vehicle = 40, 8
        spacings = np.full((n_steps, 2), 20.0)
        accels = np.zeros((n_steps, n_vehicle))
        accels[5:8, 2] = -5.0
        accels[15:17, 5] = -5.0
        state = self._synth_state(n_steps, n_vehicle, spacings, accels)
        vels = self._vels(n_steps)
        m = compute_metrics(vels, vels["head"], small_config, full_state=state)
        assert m["aeb_trigger_count"] == 5


class TestHdvDynamicsFnParam:
    def test_default_none_matches_current_ovm_path(
        self, small_config, ovm_config, weights
    ):
        """hdv_dynamics_fn=None must produce byte-identical output to the
        pre-change OVM path."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star)
        cost_a, vel_a, state_a, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
        )
        cost_b, vel_b, state_b, _ = run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            hdv_dynamics_fn=None,
        )
        np.testing.assert_allclose(state_a, state_b)
        assert cost_a == cost_b

    def test_custom_hdv_fn_is_called(self, small_config, ovm_config, weights):
        """A provided hdv_dynamics_fn must be invoked at every step."""
        Q, R = weights
        head_vel = np.full(40, small_config.v_star)
        calls = {"n": 0}

        def spy_fn(S, cfg):
            calls["n"] += 1
            return np.zeros(S.shape[0] - 1)

        run_with_state(
            small_config, ovm_config, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            hdv_dynamics_fn=spy_fn,
        )
        assert calls["n"] >= 35

    def test_idm_fn_produces_finite_trajectory(
        self, small_config, weights
    ):
        """Running with idm_dynamics_fn produces a valid (finite, positive-
        spacing) trajectory on a short flat-head run."""
        from rl_mixed_traffic.deep_lcc.idm import (
            get_default_idm_config,
            idm_dynamics,
        )

        Q, R = weights
        head_vel = np.full(40, small_config.v_star)
        idm_cfg = get_default_idm_config()

        _, _, state, _ = run_with_state(
            small_config, idm_cfg, Q, R, head_vel,
            controller_fn=_zero_controller, enable_aeb=False,
            hdv_dynamics_fn=idm_dynamics,
        )
        assert np.all(np.isfinite(state))
        gaps = state[:, :-1, 0] - state[:, 1:, 0]
        assert gaps.min() > 0.0
