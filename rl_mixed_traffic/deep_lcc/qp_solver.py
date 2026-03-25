"""DeeP-LCC QP solver using cvxopt's dense interior-point method.

Uses cvxopt.solvers.qp which is a primal-dual interior-point solver
designed for dense problems (converges in ~10-30 iterations vs OSQP's
~225 for this fully dense 1931-variable QP).

The P, A_eq, and G matrices are constant for a given set of Hankel
matrices. CachedDeepLCCSolver pre-builds these once per episode and
only updates q, b, h (which depend on uini, yini, eini) each step.
"""

from __future__ import annotations

import numpy as np
from cvxopt import matrix, solvers

# Suppress cvxopt iteration output
solvers.options["show_progress"] = False


class CachedDeepLCCSolver:
    """Pre-built cvxopt QP for DeeP-LCC.

    Build once per episode (when Hankel matrices change), then call
    repeatedly with different (uini, yini, eini) values.
    """

    def __init__(
        self,
        Up: np.ndarray,
        Yp: np.ndarray,
        Uf: np.ndarray,
        Yf: np.ndarray,
        Ep: np.ndarray,
        Ef: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        lambda_g: float,
        lambda_y: float,
        u_limit: tuple[float, float] | None = None,
        s_limit: tuple[float, float] | None = None,
    ):
        T_ini = Ep.shape[0]
        N = Ef.shape[0]
        n = Up.shape[1]

        m_ctr = Up.shape[0] // T_ini
        p_ctr = Yp.shape[0] // T_ini

        self._Uf = Uf
        self._Yf = Yf
        self._N_m = Uf.shape[0]
        self._N_p = Yf.shape[0]
        self._lambda_y = lambda_y

        # --- P matrix (constant quadratic cost) ---
        Q_blk = np.kron(np.eye(N), Q)
        R_blk = np.kron(np.eye(N), R)

        P_dense = (
            Yf.T @ Q_blk @ Yf
            + Uf.T @ R_blk @ Uf
            + lambda_g * np.eye(n)
            + lambda_y * Yp.T @ Yp
        )
        P_dense = 0.5 * (P_dense + P_dense.T)
        self._P = matrix(P_dense, tc="d")

        # --- q helper (linear cost depends on yini) ---
        self._YpT = Yp.T

        # --- A (equality constraints): [Up; Ep; Ef] g = [uini; eini; 0] ---
        A_eq = np.vstack([Up, Ep, Ef])
        self._A = matrix(A_eq, tc="d")
        self._n_up = Up.shape[0]
        self._n_ep = Ep.shape[0]
        self._n_ef = Ef.shape[0]

        # --- G, h (inequality constraints): G g <= h ---
        self._has_ineq = u_limit is not None or s_limit is not None
        if self._has_ineq:
            G_blocks = []
            h_list = []

            if u_limit is not None:
                # Uf g <= u_max  and  -Uf g <= -u_min
                G_blocks.append(Uf)
                G_blocks.append(-Uf)
                h_list.append(np.full(self._N_m, u_limit[1]))
                h_list.append(np.full(self._N_m, -u_limit[0]))

            if s_limit is not None:
                Sf = np.hstack([np.zeros((m_ctr, p_ctr - m_ctr)), np.eye(m_ctr)])
                Sf_blk = np.kron(np.eye(N), Sf)
                SfYf = Sf_blk @ Yf
                n_s = SfYf.shape[0]
                # SfYf g <= s_max  and  -SfYf g <= -s_min
                G_blocks.append(SfYf)
                G_blocks.append(-SfYf)
                h_list.append(np.full(n_s, s_limit[1]))
                h_list.append(np.full(n_s, -s_limit[0]))

            self._G = matrix(np.vstack(G_blocks), tc="d")
            self._h = matrix(np.concatenate(h_list).reshape(-1, 1), tc="d")
        else:
            self._G = None
            self._h = None

    def __call__(
        self,
        uini: np.ndarray,
        yini: np.ndarray,
        eini: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        # Linear cost (changes each step)
        q_np = -self._lambda_y * (self._YpT @ yini)
        q = matrix(q_np.reshape(-1, 1), tc="d")

        # Equality RHS (changes each step)
        b_np = np.concatenate([uini, eini, np.zeros(self._n_ef)])
        b = matrix(b_np.reshape(-1, 1), tc="d")

        sol = solvers.qp(self._P, q, self._G, self._h, self._A, b)

        if sol["status"] == "optimal":
            g_val = np.array(sol["x"]).ravel()
            u_opt = self._Uf @ g_val
            y_opt = self._Yf @ g_val
            return u_opt, y_opt, "optimal"
        else:
            return np.zeros(self._N_m), np.zeros(self._N_p), sol["status"]


def solve_deep_lcc(
    Up: np.ndarray,
    Yp: np.ndarray,
    Uf: np.ndarray,
    Yf: np.ndarray,
    Ep: np.ndarray,
    Ef: np.ndarray,
    uini: np.ndarray,
    yini: np.ndarray,
    eini: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    r: np.ndarray,
    lambda_g: float,
    lambda_y: float,
    u_limit: tuple[float, float] | None = None,
    s_limit: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve the DeeP-LCC QP for one time step.

    Convenience wrapper that builds a CachedDeepLCCSolver and calls it once.
    For repeated solves with the same Hankel matrices, use CachedDeepLCCSolver
    directly to avoid re-canonicalization overhead.
    """
    solver = CachedDeepLCCSolver(
        Up, Yp, Uf, Yf, Ep, Ef, Q, R,
        lambda_g, lambda_y,
        u_limit=u_limit, s_limit=s_limit,
    )
    return solver(uini, yini, eini)
