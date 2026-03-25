import numpy as np
import pytest

from rl_mixed_traffic.deep_lcc.hankel import build_hankel_matrices, hankel_matrix


class TestHankelMatrix:
    def test_shape_1d_signal(self):
        u = np.arange(10).reshape(1, 10).astype(float)
        H = hankel_matrix(u, L=3)
        assert H.shape == (3, 8)  # (1*3, 10-3+1)

    def test_shape_2d_signal(self):
        u = np.arange(20).reshape(2, 10).astype(float)
        H = hankel_matrix(u, L=4)
        assert H.shape == (8, 7)  # (2*4, 10-4+1)

    def test_values_1d(self):
        u = np.array([[1, 2, 3, 4, 5]], dtype=float)
        H = hankel_matrix(u, L=3)
        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ])
        np.testing.assert_array_equal(H, expected)

    def test_values_2d(self):
        u = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        H = hankel_matrix(u, L=2)
        # L=2, T=4, cols=3
        # Block row 0: u[:, 0:3] = [[1,2,3],[5,6,7]]
        # Block row 1: u[:, 1:4] = [[2,3,4],[6,7,8]]
        expected = np.array([
            [1, 2, 3],
            [5, 6, 7],
            [2, 3, 4],
            [6, 7, 8],
        ])
        np.testing.assert_array_equal(H, expected)

    def test_L_equals_T(self):
        u = np.array([[1, 2, 3]], dtype=float)
        H = hankel_matrix(u, L=3)
        assert H.shape == (3, 1)
        np.testing.assert_array_equal(H.ravel(), [1, 2, 3])


class TestBuildHankelMatrices:
    def test_shapes(self):
        T = 100
        m_ctr = 2
        p = 10
        T_ini = 5
        N = 10

        ud = np.random.randn(m_ctr, T)
        ed = np.random.randn(1, T)
        yd = np.random.randn(p, T)

        Up, Uf, Ep, Ef, Yp, Yf = build_hankel_matrices(ud, ed, yd, T_ini, N)

        cols = T - T_ini - N + 1  # 86

        assert Up.shape == (m_ctr * T_ini, cols)
        assert Uf.shape == (m_ctr * N, cols)
        assert Ep.shape == (T_ini, cols)
        assert Ef.shape == (N, cols)
        assert Yp.shape == (p * T_ini, cols)
        assert Yf.shape == (p * N, cols)

    def test_past_future_contiguous(self):
        """Past and future blocks should stack to the full Hankel matrix."""
        T = 50
        m_ctr = 1
        T_ini = 3
        N = 5

        ud = np.random.randn(m_ctr, T)
        ed = np.random.randn(1, T)
        yd = np.random.randn(2, T)

        Up, Uf, Ep, Ef, Yp, Yf = build_hankel_matrices(ud, ed, yd, T_ini, N)

        U_full = hankel_matrix(ud, T_ini + N)
        np.testing.assert_array_equal(np.vstack([Up, Uf]), U_full)

        E_full = hankel_matrix(ed, T_ini + N)
        np.testing.assert_array_equal(np.vstack([Ep, Ef]), E_full)

        Y_full = hankel_matrix(yd, T_ini + N)
        np.testing.assert_array_equal(np.vstack([Yp, Yf]), Y_full)
