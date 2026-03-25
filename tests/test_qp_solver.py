import numpy as np
import pytest

from rl_mixed_traffic.deep_lcc.config import DeepLCCConfig, OVMConfig
from rl_mixed_traffic.deep_lcc.precollect import precollect
from rl_mixed_traffic.deep_lcc.qp_solver import solve_deep_lcc


@pytest.fixture
def small_config():
    """Smaller config for fast tests."""
    return DeepLCCConfig(T=300, T_ini=5, N=10)


@pytest.fixture
def ovm_config():
    return OVMConfig()


@pytest.fixture
def hankel_matrices(small_config, ovm_config):
    return precollect(small_config, ovm_config, seed=42)


class TestSolveDeepLCC:
    def test_equilibrium_returns_near_zero(self, small_config, ovm_config, hankel_matrices):
        """At equilibrium, optimal control should be near zero."""
        Up, Uf, Ep, Ef, Yp, Yf = hankel_matrices

        ID = ovm_config.ID
        pos_cav = np.where(np.array(ID) == 1)[0]
        m_ctr = len(pos_cav)
        n_vehicle = ovm_config.n_vehicle
        p_ctr = n_vehicle + m_ctr  # measure_type=3

        T_ini = small_config.T_ini
        N = small_config.N

        Q_v = small_config.weight_v * np.eye(n_vehicle)
        Q_s = small_config.weight_s * np.eye(m_ctr)
        Q = np.block([
            [Q_v, np.zeros((n_vehicle, m_ctr))],
            [np.zeros((m_ctr, n_vehicle)), Q_s],
        ])
        R = small_config.weight_u * np.eye(m_ctr)
        r = np.zeros(p_ctr * N)

        # Equilibrium: zero errors
        uini = np.zeros(m_ctr * T_ini)
        yini = np.zeros(p_ctr * T_ini)
        eini = np.zeros(T_ini)

        u_limit = (small_config.dcel_max, small_config.acel_max)
        s_limit = (
            small_config.spacing_min - small_config.s_star,
            small_config.spacing_max - small_config.s_star,
        )

        u_opt, y_opt, status = solve_deep_lcc(
            Up, Yp, Uf, Yf, Ep, Ef,
            uini, yini, eini,
            Q, R, r,
            small_config.lambda_g, small_config.lambda_y,
            u_limit=u_limit,
            s_limit=s_limit,
        )

        assert status in ("optimal", "optimal_inaccurate")
        # At equilibrium, optimal control should be small
        np.testing.assert_allclose(u_opt[:m_ctr], 0.0, atol=0.5)

    def test_acceleration_bounds_respected(self, small_config, ovm_config, hankel_matrices):
        """Optimal control should respect acceleration bounds."""
        Up, Uf, Ep, Ef, Yp, Yf = hankel_matrices

        ID = ovm_config.ID
        pos_cav = np.where(np.array(ID) == 1)[0]
        m_ctr = len(pos_cav)
        n_vehicle = ovm_config.n_vehicle
        p_ctr = n_vehicle + m_ctr

        T_ini = small_config.T_ini
        N = small_config.N

        Q_v = small_config.weight_v * np.eye(n_vehicle)
        Q_s = small_config.weight_s * np.eye(m_ctr)
        Q = np.block([
            [Q_v, np.zeros((n_vehicle, m_ctr))],
            [np.zeros((m_ctr, n_vehicle)), Q_s],
        ])
        R = small_config.weight_u * np.eye(m_ctr)
        r = np.zeros(p_ctr * N)

        # Non-equilibrium state with perturbation
        rng = np.random.default_rng(123)
        uini = rng.standard_normal(m_ctr * T_ini) * 0.5
        yini = rng.standard_normal(p_ctr * T_ini) * 2.0
        eini = rng.standard_normal(T_ini) * 0.5

        u_limit = (small_config.dcel_max, small_config.acel_max)

        u_opt, y_opt, status = solve_deep_lcc(
            Up, Yp, Uf, Yf, Ep, Ef,
            uini, yini, eini,
            Q, R, r,
            small_config.lambda_g, small_config.lambda_y,
            u_limit=u_limit,
        )

        if status in ("optimal", "optimal_inaccurate"):
            assert np.all(u_opt >= small_config.dcel_max - 1e-4)
            assert np.all(u_opt <= small_config.acel_max + 1e-4)

    def test_unconstrained_solves(self, small_config, ovm_config, hankel_matrices):
        """QP should solve without constraints."""
        Up, Uf, Ep, Ef, Yp, Yf = hankel_matrices

        ID = ovm_config.ID
        pos_cav = np.where(np.array(ID) == 1)[0]
        m_ctr = len(pos_cav)
        n_vehicle = ovm_config.n_vehicle
        p_ctr = n_vehicle + m_ctr

        T_ini = small_config.T_ini
        N = small_config.N

        Q = np.eye(p_ctr)
        R = 0.1 * np.eye(m_ctr)
        r = np.zeros(p_ctr * N)

        uini = np.zeros(m_ctr * T_ini)
        yini = np.zeros(p_ctr * T_ini)
        eini = np.zeros(T_ini)

        u_opt, y_opt, status = solve_deep_lcc(
            Up, Yp, Uf, Yf, Ep, Ef,
            uini, yini, eini,
            Q, R, r,
            small_config.lambda_g, small_config.lambda_y,
        )

        assert status in ("optimal", "optimal_inaccurate")
        assert u_opt.shape == (m_ctr * N,)
        assert y_opt.shape == (p_ctr * N,)
