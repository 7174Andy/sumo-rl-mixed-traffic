import numpy as np


def hankel_matrix(u: np.ndarray, L: int) -> np.ndarray:
    """Generate a block Hankel matrix of order L.

    Args:
        u: Signal array of shape (m, T) where m is the signal dimension.
        L: Number of block rows.

    Returns:
        H: Hankel matrix of shape (m * L, T - L + 1).
    """
    m, T = u.shape
    H = np.zeros((m * L, T - L + 1))
    for i in range(L):
        H[i * m : (i + 1) * m, :] = u[:, i : i + T - L + 1]
    return H


def build_hankel_matrices(
    ud: np.ndarray,
    ed: np.ndarray,
    yd: np.ndarray,
    T_ini: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all six Hankel sub-matrices split into past and future blocks.

    Args:
        ud: CAV control inputs, shape (m_ctr, T).
        ed: Head vehicle disturbance, shape (1, T).
        yd: Measured outputs, shape (p, T).
        T_ini: Past horizon length.
        N: Future prediction horizon length.

    Returns:
        (Up, Uf, Ep, Ef, Yp, Yf) — past and future Hankel blocks.
    """
    m_ctr = ud.shape[0]
    p = yd.shape[0]
    L = T_ini + N

    U = hankel_matrix(ud, L)
    Up = U[: T_ini * m_ctr, :]
    Uf = U[T_ini * m_ctr :, :]

    E = hankel_matrix(ed, L)
    Ep = E[:T_ini, :]
    Ef = E[T_ini:, :]

    Y = hankel_matrix(yd, L)
    Yp = Y[: T_ini * p, :]
    Yf = Y[T_ini * p :, :]

    return Up, Uf, Ep, Ef, Yp, Yf
