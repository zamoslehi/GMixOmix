# GMixOmix

Graph matching of single-cell RNA and ATAC data, using a peak-to-gene prior.

## Try it

```bash
conda env create -f environment.yml
conda activate gmixomix
jupyter lab
```

There are two notebooks in `notebooks/`. Open one and choose **Run > Run All Cells**.

| Notebook | What it does | Time |
|---|---|---|
| `run_demo.ipynb` | Runs any of the 11 simulations once and shows the matching plots and metrics | About a minute for S1, an hour or more for all 11 |
| `evaluation_demo.ipynb` | Shows the saved results of all 11 simulations: Table 1 and the matching plots | Seconds |

## Reproduce the results

From the repository root:

```bash
python -m simulations.run --sims 1 --repeats 2   # quick try
python -m simulations.run                        # all 11 simulations with 10 repeats, which takes many hours
python -m results.make_table                     # rebuild Table 1 from the saved results
```

A simulation that already has all its repeats is skipped, so delete its file in `results/runs/` to run it again.

## Which code does what

| File | Contents |
|---|---|
| `gmixomix/solver.py` | The method: `graph_matching_withD` |
| `gmixomix/graph.py` | Builds the adjacency matrices |
| `simulations/data.py` | Generates the simulated data |
| `simulations/configs.py` | The settings of each simulation |
| `simulations/run.py` | Runs the simulations and saves their results |
| `results/metrics.py` | Calculates Acc, RMS and LDE |
| `results/plots.py` | Draws the matching plots |
| `results/make_table.py` | Writes `results/table1.md` and `results/table1.tex` |
| `tests/` | Automatic checks, run with `pytest` |

## Saved results

| File | Contents |
|---|---|
| `results/table1.md`, `results/table1.tex` | Table 1, with the 2021 values alongside in the Markdown version |
| `results/summary.csv` | Mean and SD of each simulation |
| `results/runs/` | One row per repeat of each simulation |
| `results/figures/` | The matching plots |

## Simulations in the paper

| Paper | `simulations/configs.py` | Description |
|---|---|---|
| S1 | `1` | Identical latent space, adjacency only |
| S2 | `2` | Latent space with tiny noise, adjacency only |
| S3 | `3` | Latent space with more noise, adjacency only |
| S4 | `4` | Same noise as S3 with more neighbours |
| S5 | `5` | Latent adjacency helped by noiseless RNA and ATAC |
| S6 | `6` | Latent adjacency helped by noisy RNA and ATAC |
| S7 | `7` | Graphs from noisy RNA and ATAC, latent space not accessible |
| S8 | `8` | As S7, RNA generated with a different D (5% of rows zeroed) |
| S9 | `8'` | As S8, with 70% of rows zeroed |
| S10 | `9'` | As S9, with stronger regularization (c3=2000) |
| S11 | `10` | As S8 with c3=20, and RNA and ATAC converted to Poisson counts |

`9` in `simulations/configs.py` isn't in the paper.

## Differences from the 2021 results

The saved results come from this code, so they differ from the 2021 values:

- The peak weights `w` are updated with the objective including `-c4*log(w)`, with the sign of this term in the gradient fixed. `w` stays positive in S1 to S10, and goes slightly negative in S11.
- Each repeat draws new latent points, and the random generators and PCA are seeded independently.

## Versions

The original notebook and results, as uploaded in August 2021, are kept at the tag [`v0-original`](../../tree/v0-original).
