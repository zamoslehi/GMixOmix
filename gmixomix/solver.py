"""ADMM solver for graph matching with a peak-to-gene prior.

The relaxation and ADMM scheme follow:
Robust Multimodal Graph Matching: Sparse Coding Meets Graph Matching.
Advances in Neural Information Processing Systems 26 (NIPS 2013).
M. Fiori, P. Sprechmann, J. Vogelstein, P. Muse, G. Sapiro.
"""
from typing import Tuple

import numba
import numpy as np
from numba import njit, set_num_threads
from scipy.optimize import linear_sum_assignment


@njit(parallel=True)
def normF(M):
    """Frobenius norm of M."""
    cumulant = 0.0
    for i in numba.prange(M.shape[0]):
        for j in range(M.shape[1]):
            cumulant += M[i, j] ** 2
    return np.sqrt(cumulant)


@njit(parallel=True)
def normF2(M):
    """Squared Frobenius norm of M."""
    cumulant = 0.0
    for i in numba.prange(M.shape[0]):
        for j in range(M.shape[1]):
            cumulant += M[i, j] ** 2
    return cumulant


def lapjv(X):
    """Solve the linear assignment problem and return the column assignment."""
    return linear_sum_assignment(X)[1]


def perm2mat(p: np.ndarray) -> np.ndarray:
    """Convert a permutation vector to a float permutation matrix."""
    n = len(p)
    P = np.zeros((n, n))
    for k in range(n):
        P[k, p[k]] = 1
    return P


@njit
def vector_th_alpha_beta(a, b, Lambda):
    """Group soft-thresholding of the paired entries of a and b."""
    dims = a.shape
    dimy = dims[0]
    dimx = dims[1]

    c = np.zeros((dimy, dimx), dtype=np.float32)
    d = np.zeros((dimy, dimx), dtype=np.float32)

    for i in range(dimx):
        for j in range(dimy):
            norm = np.sqrt(a[i, j] ** 2 + b[i, j] ** 2)
            if norm > 0:
                c[i, j] = np.maximum(0.0, 1.0 - Lambda / norm) * a[i, j]
                d[i, j] = np.maximum(0.0, 1.0 - Lambda / norm) * b[i, j]
    return c, d


def admom_sub_solve_for_alpha_beta(A, B, P, U, W, c):
    """Solve the ADMM subproblem for alpha and beta."""
    C = A @ P - U
    D = P @ B - W
    alpha, beta = vector_th_alpha_beta(C.astype(float), D.astype(float), 1.0 / (c + 1e-10))
    return alpha, beta


@numba.njit
def project_DS(Ti: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Project a matrix onto the set of doubly stochastic matrices."""
    n = Ti.shape[0]

    var = 1
    Jn = 1 / n
    xk = Ti

    iter_ = 0
    max_iter = 200
    while (var > tol) and (iter_ < max_iter):
        iter_ = iter_ + 1
        xkold = xk
        yk = xk - (xk.sum(0) / xk.shape[1])
        yk = yk - (yk.sum(1).reshape((-1, 1)) / xk.shape[0] - Jn)
        xk = np.maximum(0, yk)
        var = np.sum(np.abs(xk - xkold))
    return xk


@numba.njit(parallel=True)
def admom_sub_solve_for_P_linearized_withD(c1, c2, D_GA, X_A, X_B, W, alpha, beta, A, B, U, V, tau, Pk, tole=1e-5):
    """Update P with a linearized, projected step."""
    w = np.diag(W)
    C = alpha + U
    D = beta + V
    E = X_B * w @ D_GA

    CmAPk = C - A @ Pk
    DmPkB = D - Pk @ B
    XAmPkE = X_A - Pk @ E
    set_num_threads(6)
    f = c1 / 2 * normF2(CmAPk) + c1 / 2 * normF2(DmPkB) + c2 * normF2(XAmPkE)
    f2 = f + 1

    k = 1
    plus1 = tau * A.T @ (CmAPk)
    plus2 = tau * (DmPkB) @ B.T
    plus3 = tau * (XAmPkE) @ E.T

    while (f2 > f) and (k < 11):
        R1 = Pk + plus1 / k
        R2 = Pk + plus2 / k
        R3 = Pk + plus3 / k
        P = project_DS((c1 / 2 * R1 + c1 / 2 * R2 + c2 * R3) / (c1 + c2), tole)
        f2 = c1 / 2 * np.sum((C - A @ P) ** 2) + c1 / 2 * np.sum((D - P @ B) ** 2) + c2 * np.sum((X_A - P @ E) ** 2)
        k = k + 1

    return P


def admom_sub_solve_for_P_onestep_withD(c1, c2, D_GA, X_A, X_B, W, alpha, beta, A, B, U, V, Pk):
    """Update P with a line search, used when the iterates zigzag."""
    C = alpha + U
    D = beta + V
    E = X_B @ W @ D_GA

    CmAPk = C - A @ Pk
    DmPkB = D - Pk @ B
    XAmPkF = X_A - Pk @ E
    set_num_threads(6)
    f = c1 / 2 * normF2(CmAPk) + c1 / 2 * normF2(DmPkB) + c2 * normF2(XAmPkF)

    eta = 0.7
    max_iter_in = 50
    N = 5
    step = 1
    grad = c1 * A.T @ (CmAPk) + c1 * (DmPkB) @ B.T + 2 * c2 * (XAmPkF) @ E.T

    notGoodEnough = True
    iter_in = 0

    while notGoodEnough and (iter_in < max_iter_in):
        iter_in = iter_in + 1
        P1 = Pk + step * grad
        P2 = project_DS(P1)

        for i in range(1, N + 1):
            a = 1.0 / i
            Pc = (1 - a) * Pk + a * P2
            set_num_threads(6)
            fc = c1 / 2 * normF2(C - A @ Pc) + c1 / 2 * normF2(D - Pc @ B) + c2 * normF2(X_A - Pc @ E)
            if fc < f:
                P2 = Pc
                notGoodEnough = False
                break
        step = step * eta

    return P2


def admom_sub_solve_for_w_onestep_withD(c2, c3, c4, D, X_A, X_B, W, P):
    """Update the peak weights w with a line search."""
    w = np.diag(W)
    E = X_B * w @ D
    set_num_threads(6)
    f = c2 * normF2(X_A - P @ E) + c3 * np.sum(w ** 2) - c4 * np.sum(np.log(np.maximum(w, 1e-15)))
    eta = 0.7
    max_iter_in = 100
    N = 5
    step = 0.00001

    # Only the diagonal of the gradient matrix is needed, so the last matrix
    # multiplications are replaced by element-wise sums.
    P_XB = P @ X_B
    D_DT = D @ D.T
    XA_DT = X_A @ D.T
    diagPXBT_XADT = np.sum(P_XB * XA_DT, 0)
    PXBT_PXB = (P_XB).T @ P_XB
    WD_DT = D_DT * w[:, None]
    diagPXBTPXB_WDDT = np.sum(PXBT_PXB.T * WD_DT, 0)
    diagGrad_matrix = -2 * diagPXBT_XADT + 2 * diagPXBTPXB_WDDT

    grad_w = c2 * diagGrad_matrix + 2 * c3 * w + c4 / np.maximum(w, 1e-15)

    notGoodEnough = True
    iter_in = 0

    while notGoodEnough and (iter_in < max_iter_in):
        iter_in = iter_in + 1
        w1 = w - step * grad_w

        for i in range(1, N + 1):
            a = 1.0 / i
            wc = (1 - a) * w + a * w1
            Ec = X_B * wc @ D
            set_num_threads(6)
            fc = c2 * normF2(X_A - P @ Ec) + c3 * np.sum(wc ** 2) - c4 * np.sum(np.log(np.maximum(wc, 1e-15)))
            if fc < f:
                w1 = wc
                notGoodEnough = False
                break
        step = step * eta
    return w1


def stop_condition(normAPPB, normAPPB1, barrier_term, iter_, tol, tolbar, maxIter):
    """Return True when the solver should stop."""
    condition = False
    condition |= (normAPPB <= tol) and (barrier_term <= tolbar)
    condition |= (normAPPB1 <= tol) and (barrier_term <= tolbar)
    condition |= iter_ >= maxIter
    return condition


def graph_matching_withD(
    D_GA: np.ndarray,
    X_A: np.ndarray,
    X_B: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    c1: float = 2.0,
    c2: float = 1.0,
    c3: float = 1.0,
    c4: float = 1.0,
    tol: float = 2e-2,
    tolbar: float = 1e-4,
    tol2: float = 9e-9,
    N: int = 100,
    maxIter: int = 30000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the cells of two graphs using a peak-to-gene prior.

    Parameters
    ----------
    D_GA : np.ndarray
        Gene activity matrix of shape (peaks, genes).
    X_A : np.ndarray
        RNA matrix of shape (cells, genes).
    X_B : np.ndarray
        ATAC matrix of shape (cells, peaks).
    A, B : np.ndarray
        Adjacency matrices of the two graphs, of shape (cells, cells).
    c1 : float
        Weight of the graph matching term.
    c2 : float
        Weight of the RNA reconstruction term. With c2=0, w is not updated.
    c3 : float
        Weight of the L2 regularization of w.
    c4 : float
        Weight of the log barrier on w.
    tol : float
        Tolerance on the objective used for stopping.
    tolbar : float
        Tolerance on the barrier term used for stopping.
    tol2 : float
        Tolerance on the change between iterations.
    N : int
        Check the closest permutation matrix every N iterations.
    maxIter : int
        Maximum number of iterations.
    verbose : bool
        Print the objective terms during optimization.

    Returns
    -------
    P : np.ndarray
        Doubly stochastic matrix at the last iteration.
    Pp : np.ndarray
        Best permutation matrix found.
    w : np.ndarray
        Learned peak weights.
    """
    D_GA = D_GA.astype(float)

    p = A.shape
    peaks = D_GA.shape[0]

    def forig(P: np.ndarray, w: np.ndarray) -> float:
        return (
            c1 * np.sum(np.sqrt((A @ P) ** 2 + (P @ B) ** 2))
            + c2 * np.sum((X_A - P @ X_B * w @ D_GA) ** 2)
            + c3 * np.sum(w ** 2)
        )

    P = np.ones(p) / p[0]
    U = np.zeros(p)
    V = np.zeros(p)
    w = np.ones(peaks)

    iter_ = 0

    rhoA = np.max(np.linalg.eigh(A.T @ A)[0])
    rhoB = np.max(np.linalg.eigh(B.T @ B)[0])

    var = 1.0
    normAPPB = 1.0
    normAPPB1 = 1.0

    Pmin = P
    fmin = 1e10
    zigzag1 = 0
    Nzigzag = 10.0
    barrier_term = +np.inf

    forigi = []
    while not stop_condition(normAPPB, normAPPB1, barrier_term, iter_, tol, tolbar, maxIter):
        iter_ = iter_ + 1

        alpha, beta = admom_sub_solve_for_alpha_beta(A, B, P, U, V, c1)

        tole = 1e-4
        if zigzag1 > Nzigzag:
            P = admom_sub_solve_for_P_onestep_withD(c1, c2, D_GA, X_A, X_B, np.diag(w), alpha, beta, A, B, U, V, P)
            zigzag1 = 0
            Nzigzag = Nzigzag * 1.2
        else:
            E = X_B @ np.diag(w) @ D_GA
            rhoE = np.max(np.linalg.eigh(E.T @ E)[0])
            rho = max(rhoA, rhoB, rhoE)
            tau = 0.8 / rho
            set_num_threads(6)
            P = admom_sub_solve_for_P_linearized_withD(c1, c2, D_GA, X_A, X_B, np.diag(w), alpha, beta, A, B, U, V, tau, P, tole)

        if c2 > 0:
            w = admom_sub_solve_for_w_onestep_withD(c2, c3, c4, D_GA, X_A, X_B, np.diag(w), P)

        U = U + alpha - A @ P
        V = V + beta - P @ B
        fori = forig(P, w)

        # Every N iterations, check the closest permutation matrix.
        if (iter_ - 1) % N == 0:
            if verbose:
                F1 = c1 * np.sum(np.sqrt((A @ P) ** 2 + (P @ B) ** 2))
                F2 = c2 * np.sum((X_A - P @ X_B @ np.diag(w) @ D_GA) ** 2)
                F3 = c3 * np.sum(np.diag(w) ** 2)
                F4 = -c4 * np.sum(np.log(np.maximum(w, 1e-15)))
                print(f"Iteration #{iter_},F1={F1},F2={F2},F3={F3},F4={F4}")

            corr = lapjv(-P)
            P1 = perm2mat(corr)
            FirstTerm1 = np.sqrt(np.sum((A @ P1 - P1 @ B) ** 2))
            SecondTerm1 = np.sum((X_A - P1 @ X_B @ np.diag(w) @ D_GA) ** 2)
            normAPPB1 = c1 * FirstTerm1 + c2 * SecondTerm1 + c3 * np.sum(w ** 2)
            barrier_term = -c4 * np.sum(np.log(np.maximum(w, 1e-15)))
            fau = forig(P1, w)
            if fau < fmin:
                fmin = fau
                Pmin = P1

        forigi.append(fori)

        if iter_ > 5:
            set_num_threads(6)
            FirstTerm = normF(A @ P - P @ B)
            SecondTerm = normF2(X_A - P @ X_B * w @ D_GA)
            normAPPB = c1 * FirstTerm + c2 * SecondTerm + c3 * np.sum(w ** 2)
            barrier_term = -c4 * np.sum(np.log(np.maximum(w, 1e-15)))

            var = 100
            zigzag = (
                (forigi[iter_ - 3] < forigi[iter_ - 2])
                * (forigi[iter_ - 3] < forigi[iter_ - 4])
                * (forigi[iter_ - 1] < forigi[iter_ - 2])
                * (forigi[iter_ - 5] < forigi[iter_ - 4])
                * (forigi[iter_ - 5] < forigi[iter_ - 6])
            )
            zigzag1 = zigzag1 + zigzag

            if iter_ == 6 and var < 1e-7:
                var = 1
                P = np.eye(p)

        if verbose and stop_condition(normAPPB, normAPPB1, barrier_term, iter_, tol, tolbar, maxIter):
            if not (normAPPB > tol):
                print(f"Terminating at {iter_} for tol (APPB) at {normAPPB}")
            if not (normAPPB1 > tol):
                print(f"Terminating at {iter_} for tol (APPB1)")
            if not (var > tol2):
                print(f"Terminating at {iter_} for tol2 = {var}")
            if not (iter_ < maxIter):
                print(f"Terminating at {iter_} for maxiter")

    # Keep the better of the last permutation and the best one seen.
    corr = lapjv(-P)
    Pp = perm2mat(corr)
    fau = forig(Pp, w)
    if fmin < fau:
        fau = fmin
        Pp = Pmin

    return P, Pp, w
