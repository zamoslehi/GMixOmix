"""Settings of the simulated datasets and of the 12 simulations."""

SEED = 19900715
N_REPEATS = 10

MEANS = [[3.8, 2.5], [0, 2.1], [-3, 5], [-3, 0], [5.5, 6]]
EIGRANGES = [(0.3, 1.5), (0.3, 1), (0.5, 1.5), (0.5, 1), (0.3, 1.1)]
NCELLS = [90, 110, 120, 80, 100]
NPEAKS = 1000
NGENES = 100

_STRICT = dict(tol=9e-17, tolbar=1e-5)
_LOOSE = dict(tol=1e-5, tolbar=1e-5)
_NO_OMICS = dict(c1=1, c2=0, c3=0, c4=0)
_OMICS = dict(c1=1, c2=5, c3=0, c4=5)


def _sim(description, *, snr=3, k=100, omics=False, omics_noise=False, graph_from_omics=False,
         d_rows_zero=None, poisson=False, pca_dims=4, solver):
    return dict(
        description=description,
        snr=snr,                            # noise on the first latent space, None for no noise
        k=k,                                # neighbours in the adjacency matrices
        omics=omics,                        # simulate RNA and ATAC matrices
        omics_noise=omics_noise,            # add colored noise to RNA and ATAC
        graph_from_omics=graph_from_omics,  # build the graphs from PCA of RNA and ATAC
        d_rows_zero=d_rows_zero,            # generate RNA with a D that differs from the solver's D
        poisson=poisson,                    # convert RNA and ATAC to counts
        pca_dims=pca_dims,
        solver=solver,
    )


SIMULATIONS = {
    "1": _sim("Identical latent space, adjacency only", snr=None, k=50,
              solver=dict(_NO_OMICS, N=20, maxIter=500, **_STRICT)),
    "2": _sim("Latent space with tiny noise, adjacency only", snr=10, k=50,
              solver=dict(_NO_OMICS, N=20, maxIter=500, **_STRICT)),
    "3": _sim("Latent space with more noise, adjacency only", k=50,
              solver=dict(_NO_OMICS, N=50, maxIter=1000, **_STRICT)),
    "4": _sim("Same noise as 3 with more neighbours", k=100,
              solver=dict(_NO_OMICS, N=50, maxIter=1000, **_STRICT)),
    "5": _sim("Latent adjacency helped by noiseless RNA and ATAC", omics=True,
              solver=dict(_OMICS, N=80, maxIter=1000, **_STRICT)),
    "6": _sim("Latent adjacency helped by noisy RNA and ATAC", omics=True, omics_noise=True,
              solver=dict(_OMICS, N=50, maxIter=1000, **_STRICT)),
    "7": _sim("Graphs from noisy RNA and ATAC, latent space not accessible",
              omics=True, omics_noise=True, graph_from_omics=True,
              solver=dict(_OMICS, N=50, maxIter=1000, **_LOOSE)),
    "8": _sim("As 7, RNA generated with a different D (5% of rows zeroed)",
              omics=True, omics_noise=True, graph_from_omics=True, d_rows_zero=0.05,
              solver=dict(_OMICS, N=50, maxIter=1000, **_LOOSE)),
    "8'": _sim("As 8, with 70% of rows zeroed",
               omics=True, omics_noise=True, graph_from_omics=True, d_rows_zero=0.7,
               solver=dict(_OMICS, N=50, maxIter=1000, **_LOOSE)),
    "9": _sim("As 8, with regularization (c3=20)",
              omics=True, omics_noise=True, graph_from_omics=True, d_rows_zero=0.05,
              solver=dict(_OMICS, c3=20, N=50, maxIter=1000, **_LOOSE)),
    "9'": _sim("As 8', with stronger regularization (c3=2000)",
               omics=True, omics_noise=True, graph_from_omics=True, d_rows_zero=0.7,
               solver=dict(_OMICS, c3=2000, N=50, maxIter=1000, **_LOOSE)),
    "10": _sim("As 9, with RNA and ATAC converted to Poisson counts",
               omics=True, omics_noise=True, graph_from_omics=True, d_rows_zero=0.05,
               poisson=True, pca_dims=5,
               solver=dict(_OMICS, c3=20, N=50, maxIter=1000, **_LOOSE)),
}
