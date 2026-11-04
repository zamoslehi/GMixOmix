# GMixOmix

Graph matching of single-cell RNA and ATAC data, using a peak-to-gene prior.

## Try it

```bash
conda env create -f environment.yml
conda activate gmixomix
jupyter lab
```

Open `notebooks/demo.ipynb` and choose **Run > Run All Cells**. It simulates a dataset, matches the cells, and shows the result in about a minute.

## Reproduce the simulations

From the repository root:

```bash
python -m simulations.run --sims 1 --repeats 2   # quick try
python -m simulations.run                        # all simulations, which takes many hours
```

Results are written to `results/`.

## Layout

| Folder | Contents |
|---|---|
| `gmixomix/` | The method |
| `simulations/` | Data generation and the settings of each simulation |
| `results/` | Metrics, plots, and saved results |
| `notebooks/` | The demo |
| `tests/` | Automatic checks, run with `pytest` |

## Versions

The original notebook and results, as uploaded in August 2021, are kept at the tag [`v0-original`](../../tree/v0-original).
