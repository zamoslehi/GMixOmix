"""Functions that generate the simulated data."""
import numpy as np
from scipy.linalg import svd


def randPSD(rng, size=2, eigrange=(0.2, 5)):
    """Draw a random positive semi-definite matrix with eigenvalues in eigrange."""
    L = rng.normal(0, 1, size=(size, size))
    _, _, eigvects = np.linalg.svd(L, full_matrices=False)
    eigvals = rng.uniform(eigrange[0], eigrange[1], size=size)
    return eigvects.T * eigvals @ eigvects


def perm2P(p):
    """Convert a permutation vector to an integer permutation matrix."""
    shape = len(p)
    P = np.zeros((shape, shape), dtype=int)
    for i in range(shape):
        P[i, p[i]] = 1
    return P


def make_random_P(rng, shape):
    """Draw a random permutation matrix."""
    return perm2P(rng.permutation(shape))


def make_latent_space(rng, means_list, covs_list, ncells_list):
    """Generate points in a latent space as anisotropic gaussians.

    Parameters
    ----------
    means_list : list of array-like
        The cluster centers.
    covs_list : list of np.ndarray
        The covariance matrix of each gaussian.
    ncells_list : list of int
        The number of cells in each gaussian.

    Returns
    -------
    Y : np.ndarray
        The points in the latent space.
    labels : np.ndarray
        The gaussian each point was generated from.
    """
    labels = np.concatenate([np.ones(n, dtype=int) * i for i, n in enumerate(ncells_list)])
    Y = np.concatenate(
        [
            rng.multivariate_normal(mean=means_list[i], cov=covs_list[i], size=ncells_list[i])
            for i in range(len(means_list))
        ]
    )
    return Y, labels


def embed_in_higher_dimensions(rng, Y, dims, return_basis=False):
    """Embed points in a random basis of a higher dimensional space.

    Parameters
    ----------
    Y : np.ndarray
        The points in the latent space.
    dims : int
        The number of dimensions to embed the points into.
    return_basis : bool
        Also return the matrix UT such that X = Y @ UT.
    """
    L = rng.normal(0, 1, size=(dims, Y.shape[1]))
    U, _, _ = svd(L, full_matrices=False)
    X = Y @ U.T
    if return_basis:
        return X, U.T
    return X


def add_relative_noise(rng, X, SNR=5.0, to_zero_centered=True):
    """Add gaussian noise to each column so that the signal to noise ratio is SNR."""
    if to_zero_centered:
        center = X.mean(0)
        X = X - center
        X = X + rng.normal(0, (X.std(0) + 1e-12) / SNR, size=X.shape)
        X = X + center
        return X
    return X + rng.normal(0, (X.std(0) + 1e-12) / SNR, size=X.shape)


def add_colored_noise(rng, X, latentdims=2, nonflatdims=20, SNR=5, alpha=0.5, to_zero_centered=True):
    """Add correlated noise to a data matrix by changing its singular values.

    Parameters
    ----------
    X : np.ndarray
        The matrix to add noise to.
    latentdims : int
        The original dimensionality of the latent space of X.
    nonflatdims : int
        The number of components that are nonzero.
    SNR : float
        Signal to noise ratio.
    alpha : float
        The decay rate of the added singular values.
    """
    assert SNR >= 1, "SNR below 1 would change the top principal component"
    if to_zero_centered:
        center = X.mean(0)
        X = X - center
    X = X + rng.normal(0, 1e-6, X.shape)
    U, s, Vt = svd(X, full_matrices=False)
    s[latentdims:] = np.exp(-alpha * np.arange(latentdims, len(s)))

    snew = np.zeros_like(s)

    noise = s[latentdims:nonflatdims].sum()
    signal = s[:latentdims].sum() - noise

    snew[:latentdims] = s[:latentdims] * SNR / signal
    snew[latentdims:nonflatdims] = s[latentdims:nonflatdims] / noise
    snew = snew * (signal + noise) / snew.sum()
    X = U * snew @ Vt
    if to_zero_centered:
        X = X + center
    return X


def from_normal_to_poisson2(rng, X, perc=5, factor=(3, 8)):
    """Turn continuous values into Poisson counts."""
    X = np.clip(X - np.percentile(X, perc, axis=0), 0, np.inf)
    X = np.exp(X)
    X = X * rng.uniform(factor[0], factor[1], X.shape[1])
    return rng.poisson(X, X.shape)


def make_blockdiag_D(npeaks, ngenes):
    """Make a block diagonal peak-to-gene matrix made of ones."""
    D = np.zeros((npeaks, ngenes), dtype=int)
    ppg = int(npeaks // ngenes)
    for i in range(ngenes):
        D[ppg * i : ppg * i + ppg, i] = 1
    D[ppg * i + ppg :, i] = 1  # in case npeaks % ngenes != 0
    return D


def _choose(rng, n, fraction):
    return rng.choice(n, size=int(np.ceil(n * fraction)), replace=False)


def random_rows_to_zero(rng, D, fraction=0.3):
    """Return a copy of D with a fraction of the rows set to 0, and their indices."""
    Dnew = np.copy(D)
    idx = _choose(rng, D.shape[0], fraction)
    Dnew[idx, :] = 0
    return Dnew, idx


def random_rows_permuted(rng, D, fraction=0.3):
    """Return a copy of D with a fraction of the rows shuffled, and their indices."""
    Dnew = np.copy(D)
    idx = _choose(rng, D.shape[0], fraction)
    Dnew[idx, :] = Dnew[idx, :][rng.permutation(len(idx)), :]
    return Dnew, idx


def random_cols_to_zero(rng, D, fraction=0.3):
    """Return a copy of D with a fraction of the columns set to 0, and their indices."""
    Dnew = np.copy(D)
    idx = _choose(rng, D.shape[1], fraction)
    Dnew[:, idx] = 0
    return Dnew, idx


def random_cols_permuted(rng, D, fraction=0.3):
    """Return a copy of D with a fraction of the columns shuffled, and their indices."""
    Dnew = np.copy(D)
    idx = _choose(rng, D.shape[1], fraction)
    Dnew[:, idx] = Dnew[:, idx][:, rng.permutation(len(idx))]
    return Dnew, idx
