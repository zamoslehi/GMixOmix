"""Run the simulations and save their results.

Usage, from the repository root:
    python -m simulations.run                     # the 11 simulations of the paper
    python -m simulations.run --sims 1 "8'" 10    # some simulations
    python -m simulations.run --repeats 2         # fewer repeats

Each simulation has its own file in results/runs/, so running one
simulation does not change the results of the others. A simulation that already
has all its repeats is skipped. results/summary.csv is then rebuilt
from all the files in that folder.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from gmixomix import compute_adjaciency_matrix, graph_matching_withD
from results.metrics import calc_stats
from results.plots import plot_latent_space, plot_matching

from . import configs, data

RESULTS = Path(__file__).resolve().parent.parent / "results"
LATENT_DIMS = len(configs.MEANS[0])


def make_generators():
    """Return the random generators of the latent space and of RNA and ATAC."""
    latent, omics = np.random.SeedSequence(configs.SEED).spawn(2)
    return np.random.default_rng(latent), np.random.default_rng(omics)


def make_covariances():
    """Return the covariance matrix of each cluster."""
    rng = np.random.default_rng(configs.SEED)
    return [data.randPSD(rng, eigrange=e) for e in configs.EIGRANGES]


def simulate(cfg, rng1, rng2, Y, labels):
    """Generate the data of one repeat of a simulation."""
    ncells, npeaks, ngenes = Y.shape[0], configs.NPEAKS, configs.NGENES
    P0 = data.make_random_P(rng1, ncells)
    Yb, labels_b = Y, labels
    Ya = P0 @ Yb if cfg["snr"] is None else data.add_relative_noise(rng1, P0 @ Yb, SNR=cfg["snr"])
    labels_a = P0 @ labels_b

    if cfg["omics"]:
        Xb = data.embed_in_higher_dimensions(rng2, Yb, npeaks)
        D = data.make_blockdiag_D(npeaks, ngenes)
        D_true = D
        if cfg["d_rows_zero"] is not None:
            D_true, _ = data.random_cols_to_zero(rng2, D_true, fraction=0.1)
            D_true, _ = data.random_cols_permuted(rng2, D_true, fraction=0.1)
            D_true, _ = data.random_rows_to_zero(rng2, D_true, fraction=cfg["d_rows_zero"])
            D_true, _ = data.random_rows_permuted(rng2, D_true, fraction=0.05)
        Xa = P0 @ Xb @ D_true
        if cfg["omics_noise"]:
            Xb = data.add_colored_noise(rng2, Xb, latentdims=LATENT_DIMS, nonflatdims=15, SNR=5, alpha=0.5)
            Xa = data.add_colored_noise(rng2, Xa, latentdims=LATENT_DIMS, nonflatdims=10, SNR=6, alpha=0.5)
        if cfg["poisson"]:
            Xa = data.from_normal_to_poisson2(rng2, Xa, factor=(4, 8))
            Xb = data.from_normal_to_poisson2(rng2, Xb, factor=(1, 2))
    else:
        D = np.zeros((npeaks, ngenes), dtype=int)
        Xa = np.zeros((ncells, ngenes), dtype=int)
        Xb = np.zeros((ncells, npeaks), dtype=int)

    pca_a, pca_b = None, None
    if cfg["graph_from_omics"]:
        pre = (lambda X: np.log(X + 1)) if cfg["poisson"] else (lambda X: X)
        seed_a, seed_b = (int(s) for s in rng2.integers(2 ** 31, size=2))
        pca_a = PCA(cfg["pca_dims"], random_state=seed_a).fit_transform(pre(Xa))
        pca_b = PCA(cfg["pca_dims"], random_state=seed_b).fit_transform(pre(Xb))
        A = compute_adjaciency_matrix(pca_a, n_neighbors=cfg["k"])
        B = compute_adjaciency_matrix(pca_b, n_neighbors=cfg["k"])
    else:
        A = compute_adjaciency_matrix(Ya, n_neighbors=cfg["k"])
        B = compute_adjaciency_matrix(Yb, n_neighbors=cfg["k"])

    return dict(P0=P0, Ya=Ya, Yb=Yb, labels_a=labels_a, labels_b=labels_b, D=D, Xa=Xa, Xb=Xb,
                A=A, B=B, pca_a=pca_a, pca_b=pca_b)


def slug(name):
    """Return a file name friendly name of a simulation."""
    return "sim_" + name.replace("'", "prime")


def save_figures(name, sim, P):
    """Save the matching found in a repeat of a simulation."""
    directory = RESULTS / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    pa, pb = (sim["pca_a"], sim["pca_b"]) if sim["pca_a"] is not None else (sim["Ya"], sim["Yb"])
    rng = np.random.default_rng(configs.SEED)
    for label, matching in (("ground_truth", sim["P0"]), ("solution", P)):
        fig = plot_matching(rng, pa[:, :2], pb[:, :2], matching, sim["labels_a"], sim["labels_b"],
                            show_fraction=0.25, title=f"Simulation {name}: {label.replace('_', ' ')}")
        fig.savefig(directory / f"{slug(name)}_{label}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def run_simulation(name, n_repeats=configs.N_REPEATS, verbose=False):
    """Run the repeats of one simulation and write one row per repeat."""
    path = RESULTS / "runs" / f"{slug(name)}.csv"
    if path.exists():
        with open(path) as handle:
            if len(list(csv.DictReader(handle))) == n_repeats:
                print(f"Simulation {name} already has {n_repeats} repeats, skipping", flush=True)
                return

    cfg = configs.SIMULATIONS[name]
    rng1, rng2 = make_generators()
    covariances = make_covariances()

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(n_repeats):
        start = time.time()
        Y, labels = data.make_latent_space(rng1, configs.MEANS, covariances, configs.NCELLS)
        sim = simulate(cfg, rng1, rng2, Y, labels)
        _, P, w = graph_matching_withD(sim["D"], sim["Xa"], sim["Xb"], sim["A"], sim["B"],
                                       verbose=verbose, **cfg["solver"])
        stats = calc_stats(sim["Ya"], sim["Yb"], sim["labels_a"], sim["labels_b"], sim["P0"], P, k=20)
        row = dict(simulation=name, repeat=repeat, **stats,
                   w_min=float(w.min()), w_max=float(w.max()), seconds=round(time.time() - start, 1))
        rows.append(row)
        # Written after every repeat so that an interrupted run keeps its results.
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Simulation {name} repeat {repeat}: label accuracy {stats['label_accuracy']:.4f}, "
              f"{row['seconds']} s", flush=True)
    save_figures(name, sim, P)


def summarize():
    """Rebuild results/summary.csv with the mean and SD of each simulation."""
    metrics = ["label_accuracy", "label_accuracy_input", "label_accuracy_groundtruth",
               "relaxed_match_accuracy", "relaxed_match_accuracy_input", "relaxed_match_accuracy_groundtruth",
               "avg_dist_solution", "avg_dist_input", "avg_dist_groundtruth",
               "fract_exact_matches", "w_min", "w_max", "seconds"]
    rows = []
    for name, cfg in configs.SIMULATIONS.items():
        path = RESULTS / "runs" / f"{slug(name)}.csv"
        if not path.exists():
            continue
        with open(path) as handle:
            repeats = list(csv.DictReader(handle))
        row = dict(simulation=name, description=cfg["description"], repeats=len(repeats))
        for metric in metrics:
            values = [float(r[metric]) for r in repeats]
            row[f"{metric}_mean"] = round(statistics.mean(values), 4)
            row[f"{metric}_sd"] = round(statistics.stdev(values), 4) if len(values) > 1 else ""
        rows.append(row)
    if rows:
        with open(RESULTS / "summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", nargs="+", choices=list(configs.SIMULATIONS),
                        default=list(configs.PAPER_SIMULATIONS.values()))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=configs.N_REPEATS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    plt.switch_backend("Agg")  # figures are saved to files, not shown
    threadpool_limits(limits=args.threads)
    rng = np.random.default_rng(configs.SEED)
    Y, labels = data.make_latent_space(rng, configs.MEANS, make_covariances(), configs.NCELLS)
    directory = RESULTS / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plot_latent_space(Y, labels).savefig(directory / "latent_space.png", dpi=100, bbox_inches="tight")

    for name in args.sims:
        run_simulation(name, args.repeats, args.verbose)
        summarize()


if __name__ == "__main__":
    main()
