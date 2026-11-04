import numpy as np

from gmixomix import compute_adjaciency_matrix, graph_matching_withD
from gmixomix.solver import project_DS


def test_project_DS_gives_doubly_stochastic_matrix():
    rng = np.random.default_rng(0)
    P = project_DS(rng.random((20, 20)), 1e-8)
    assert (P >= 0).all()
    assert np.allclose(P.sum(0), 1, atol=1e-3)
    assert np.allclose(P.sum(1), 1, atol=1e-3)


def test_recovers_a_permutation_of_identical_graphs():
    rng = np.random.default_rng(0)
    Y = np.concatenate([rng.normal(c, 0.5, size=(20, 2)) for c in ((0, 0), (5, 0), (0, 5))])
    P0 = np.eye(60)[rng.permutation(60)]
    A = compute_adjaciency_matrix(P0 @ Y, n_neighbors=10)
    B = compute_adjaciency_matrix(Y, n_neighbors=10)
    D, Xa, Xb = np.zeros((10, 5), dtype=int), np.zeros((60, 5), dtype=int), np.zeros((60, 10), dtype=int)
    _, P, _ = graph_matching_withD(D, Xa, Xb, A, B, c1=1, c2=0, c3=0, c4=0, N=20, maxIter=300,
                                   tol=9e-17, tolbar=1e-5, tol2=9e-16)
    assert np.mean(np.all(P @ Y == P0 @ Y, axis=1)) > 0.9
