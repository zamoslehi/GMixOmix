"""Write Table 1 of the paper from the saved results, and compare it with the paper.

Usage, from the repository root:
    python -m results.make_table barrier
"""
import csv
import sys
from pathlib import Path

from simulations import configs

RESULTS = Path(__file__).resolve().parent
ROWS = [("Acc", "label_accuracy", "label_accuracy"),
        ("RMS", "relaxed_match_accuracy", "relaxed_match_accuracy"),
        ("LDE", "avg_dist", "avg_dist_solution")]


def read_summary(version):
    with open(RESULTS / version / "summary.csv") as handle:
        return {row["simulation"]: row for row in csv.DictReader(handle)}


def read_paper():
    with open(RESULTS / "paper_table1.csv") as handle:
        return {(row["simulation"], row["metric"]): row for row in csv.DictReader(handle)}


def cell(row, metric):
    return f"{float(row[metric + '_mean']):.3f}$\\pm${float(row[metric + '_sd'] or 0):.3f}"


def plain(text):
    return text.replace("$\\pm$", " ± ")


def table_rows(summary):
    for label, name in configs.PAPER_SIMULATIONS.items():
        if name not in summary:
            continue
        row = summary[name]
        for metric, stem, method in ROWS:
            yield label, metric, cell(row, f"{stem}_input"), cell(row, f"{stem}_groundtruth"), cell(row, method)


def write_latex(version, summary):
    lines = [r"\begin{tabular}{llccc}", r"\hline", r" & & Input & Ground truth & GMixOmix \\", r"\hline"]
    last = None
    for label, metric, *values in table_rows(summary):
        if last and label != last:
            lines.append(r"\hline")
        first = label.replace("S", r"\textbf{S ") + "}" if label != last else ""
        lines.append(f"{first} & {metric} & " + " & ".join(values) + r" \\")
        last = label
    lines += [r"\hline", r"\end{tabular}"]
    (RESULTS / version / "table1.tex").write_text("\n".join(lines) + "\n")


def write_markdown(version, summary):
    paper = read_paper()
    lines = ["| Simulation | Metric | Input | Ground truth | GMixOmix | Paper (2021) |", "|---|---|---|---|---|---|"]
    for label, metric, *values in table_rows(summary):
        old = paper[(label, metric)]
        lines.append(f"| {label} | {metric} | " + " | ".join(plain(v) for v in values)
                     + f" | {float(old['method_mean']):.3f} ± {float(old['method_sd']):.3f} |")
    (RESULTS / version / "table1.md").write_text("\n".join(lines) + "\n")


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "barrier"
    summary = read_summary(version)
    write_latex(version, summary)
    write_markdown(version, summary)
    print((RESULTS / version / "table1.md").read_text())


if __name__ == "__main__":
    main()
