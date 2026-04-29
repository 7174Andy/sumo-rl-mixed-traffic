import numpy as np
import pytest

from rl_mixed_traffic.deep_lcc.config import OVMConfig
from rl_mixed_traffic.deep_lcc.idm import (
    IDMConfig,
    get_default_idm_config,
    idm_dynamics,
)
from rl_mixed_traffic.deep_lcc.ovm import hdv_dynamics


@pytest.fixture
def idm_config() -> IDMConfig:
    return get_default_idm_config()


class TestIDMDynamics:
    def test_output_shape(self, idm_config):
        n = idm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[:, 1] = 15.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 20.0
        acel = idm_dynamics(S, idm_config)
        assert acel.shape == (n,)

    def test_shape_parity_with_ovm(self, idm_config):
        """IDM output shape must match OVM for the same state array."""
        n = idm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[:, 1] = 15.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 20.0
        ovm_cfg = OVMConfig()
        assert idm_dynamics(S, idm_config).shape == hdv_dynamics(S, ovm_cfg).shape

    def test_free_flow_limit(self, idm_config):
        """With a huge gap, acceleration approaches a * (1 - (v/v0)^delta)."""
        n = idm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[0, 0] = 0.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 10_000.0  # enormous gap
        v = idm_config.v0 / 2.0
        S[:, 1] = v

        acel = idm_dynamics(S, idm_config)
        expected = idm_config.a * (1.0 - (v / idm_config.v0) ** idm_config.delta)
        np.testing.assert_allclose(acel, expected, atol=1e-3)

    def test_equilibrium_near_zero(self, idm_config):
        """At IDM equilibrium spacing at v_star=15, accelerations are ~0."""
        n = idm_config.n_vehicle
        v_star = 15.0
        s_eq = (idm_config.s0 + v_star * idm_config.T) / np.sqrt(
            1.0 - (v_star / idm_config.v0) ** idm_config.delta
        )
        S = np.zeros((n + 1, 3))
        S[0, 0] = 0.0
        for i in range(1, n + 1):
            S[i, 0] = S[i - 1, 0] - s_eq
        S[:, 1] = v_star

        acel = idm_dynamics(S, idm_config)
        np.testing.assert_allclose(acel, 0.0, atol=1e-3)

    def test_safety_braking_clamp(self, idm_config):
        """If closing-speed criterion exceeds |dcel_max|, clamp to dcel_max."""
        n = idm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[0, 0] = 0.0
        S[0, 1] = 5.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 2.0
            S[i, 1] = 25.0

        acel = idm_dynamics(S, idm_config)
        assert np.any(acel == idm_config.dcel_max)

    def test_accel_saturation(self, idm_config):
        """All accelerations lie within [dcel_max, acel_max]."""
        n = idm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[0, 0] = 0.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 50.0  # large gap → big free-flow accel
        S[:, 1] = 1.0

        acel = idm_dynamics(S, idm_config)
        assert np.all(acel >= idm_config.dcel_max)
        assert np.all(acel <= idm_config.acel_max)
