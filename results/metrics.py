"""Metrics that compare a matching with the ground truth."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from gmixomix.graph import compute_adjaciency_matrix
from gmixomix.solver import lapjv, perm2mat


def score_latent(P, YA, YB, k=10):
    """Average mass of P on the k nearest neighbours in the latent space."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(YB)
    _, neigh_ind = nn.kneighbors(YA)
    score = 0
    for i in range(neigh_ind.shape[0]):
        for j in neigh_ind[i, :]:
            score += P[i, j]
    return score / P.shape[0]


def score_neigh(Ya, Yb, P, P0, k=10):
    """Fraction of cells matched within k neighbours of their true match."""
    adj_b = compute_adjaciency_matrix(Yb, n_neighbors=k)
    count = 0.0
    for i in range(adj_b.shape[0]):
        count += np.any(P @ adj_b[i, :] * P0[:, i])
    return count / Ya.shape[0]


def calc_stats(Ya, Yb, labels_a, labels_b, P0, P, k=20):
    """Compute all metrics for a matching P, as a dictionary.

    Suffixes: `_input` is the matching before optimization, `_input_I` is the
    identity matching, `_groundtruth` is the true matching P0, and no suffix or
    `_solution` is the matching P found by the solver.
    """
    P_init = np.ones(P.shape) / P.shape[0]
    P_permut_init = perm2mat(lapjv(-P_init))
    identity = np.eye(P.shape[0])

    return dict(
        avg_dist_input=np.linalg.norm(Ya - P_permut_init @ Yb, axis=1).mean(),
        avg_dist_input_I=np.linalg.norm(Ya - Yb, axis=1).mean(),
        avg_dist_groundtruth=np.linalg.norm(Ya - P0 @ Yb, axis=1).mean(),
        avg_dist_solution=np.linalg.norm(Ya - P @ Yb, axis=1).mean(),
        fract_exact_matches=np.sum(np.logical_and(P0, P)) / P.shape[0],
        label_accuracy_input=np.mean(labels_a == P_permut_init @ labels_b),
        label_accuracy_input_I=np.mean(labels_a == labels_b),
        label_accuracy_groundtruth=np.mean(labels_a == P0 @ labels_b),
        label_accuracy=np.mean(labels_a == P @ labels_b),
        latent_score_input=score_latent(P_permut_init, Ya, Yb, k=k),
        latent_score_input_I=score_latent(identity, Ya, Yb, k=k),
        latent_score_groundtruth=score_latent(P0, Ya, Yb, k=k),
        latent_score_solution=score_latent(P, Ya, Yb, k=k),
        relaxed_match_accuracy_input=score_neigh(Ya, Yb, P_permut_init, P0, k=k),
        relaxed_match_accuracy_input_I=score_neigh(Ya, Yb, identity, P0, k=k),
        relaxed_match_accuracy_groundtruth=score_neigh(Ya, Yb, P0, P0, k=k),
        relaxed_match_accuracy=score_neigh(Ya, Yb, P, P0, k=k),
    )
