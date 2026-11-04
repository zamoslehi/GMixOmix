"""Figures of the simulated data and matchings."""
import matplotlib.pyplot as plt
import numpy as np


def plot_latent_space(Y, labels):
    """Scatter plot of the latent space coloured by cluster."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10")
    ax.axis("equal")
    ax.set_title("Latent space")
    return fig


def plot_matching(rng, Xa, Xb, P, labels_a=None, labels_b=None, show_fraction=1, alpha=0.5, title=None):
    """Plot both datasets side by side, with lines between matched cells."""
    fig, ax = plt.subplots(figsize=(15, 7))
    delta = 2 * np.sqrt(np.linalg.svd(Xa, compute_uv=False)[0])
    ax.scatter(Xa[:, 0], Xa[:, 1], c=labels_a, cmap="tab10" if labels_a is not None else None)
    ax.scatter(Xb[:, 0] + delta, Xb[:, 1], c=labels_b, cmap="tab10" if labels_b is not None else None)
    p = np.where(P)[1]
    n_shown = int(np.ceil(show_fraction * Xa.shape[0]))
    for i in rng.choice(Xa.shape[0], size=n_shown, replace=False):
        ax.plot([Xa[i, 0], Xb[p[i], 0] + delta], [Xa[i, 1], Xb[p[i], 1]], c=(0.7, 0.7, 0.7, alpha))
    ax.axis("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig
