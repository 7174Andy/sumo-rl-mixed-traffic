import numpy as np
import pytest

from rl_mixed_traffic.deep_lcc.config import OVMConfig
from rl_mixed_traffic.deep_lcc.ovm import hdv_dynamics


@pytest.fixture
def ovm_config():
    return OVMConfig()


class TestHDVDynamics:
    def test_equilibrium_near_zero(self, ovm_config):
        """At OVM equilibrium spacing, accelerations should be near zero."""
        n = ovm_config.n_vehicle
        v_star = 15.0
        # OVM equilibrium: s = acos(1 - 2*v_star/v_max)/pi * (s_go - s_st) + s_st
        s_eq = (
            np.arccos(1 - 2 * v_star / ovm_config.v_max) / np.pi
            * (ovm_config.s_go - ovm_config.s_st)
            + ovm_config.s_st
        )  # = 15.0 for default params

        S = np.zeros((n + 1, 3))
        S[0, 0] = 0.0
        for i in range(1, n + 1):
            S[i, 0] = S[i - 1, 0] - s_eq
        S[:, 1] = v_star

        acel = hdv_dynamics(S, ovm_config)
        np.testing.assert_allclose(acel, 0.0, atol=0.1)

    def test_output_shape(self, ovm_config):
        n = ovm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[:, 1] = 15.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 20.0

        acel = hdv_dynamics(S, ovm_config)
        assert acel.shape == (n,)

    def test_acceleration_saturation(self, ovm_config):
        """Accelerations should be clipped to [dcel_max, acel_max]."""
        n = ovm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        # Very large spacing → large desired velocity → large acceleration
        S[0, 0] = 0.0
        for i in range(1, n + 1):
            S[i, 0] = -i * 100.0
        S[:, 1] = 1.0  # very slow

        acel = hdv_dynamics(S, ovm_config)
        assert np.all(acel >= ovm_config.dcel_max)
        assert np.all(acel <= ovm_config.acel_max)

    def test_safety_braking(self, ovm_config):
        """When gap is very small and follower is faster, safety braking activates."""
        n = ovm_config.n_vehicle
        S = np.zeros((n + 1, 3))
        S[0, 0] = 0.0
        S[0, 1] = 5.0  # slow leader
        for i in range(1, n + 1):
            S[i, 0] = -i * 2.0  # very close spacing
            S[i, 1] = 25.0  # fast follower

        acel = hdv_dynamics(S, ovm_config)
        # At least some vehicles should get emergency braking
        assert np.any(acel == ovm_config.dcel_max)
