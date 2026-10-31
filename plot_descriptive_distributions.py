"""
plot_descriptive_distributions.py

Descriptive distributions of two MPNet CSD quantities on the deidentified
N=451 data, read from the all-mpnet-base-v2 rows of
results/crossmodel_csd_long.csv (written by analyze_crossmodel_agreement.py;
no embeddings are generated):

  aligned_normalized   normalized aligned CSD
  revision_magnitude   M = ||F - I||

Outputs:
  figures/aligned_normalized_distribution.{png,pdf}
  figures/revision_magnitude_distribution.{png,pdf}
  results/descriptive_distribution_summary.csv   (SD uses ddof=1)

Usage:
    python plot_descriptive_distributions.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from critique_permutation_test import BAR, GRID, INK, INK_MUTED, INK_SECONDARY, save

LONG_PATH = Path("results/crossmodel_csd_long.csv")
MODEL_NAME = "all-mpnet-base-v2"
RESULTS_DIR = Path("results")
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
KDE_COLOR = "#0b2e5c"

FIGURES = [
    ("aligned_normalized", "Normalized aligned CSD", None,
     "aligned_normalized_distribution", (0.0, 1.0), "upper right"),
    ("revision_magnitude", r"Revision magnitude $\Vert F - I \Vert$", "Distribution of Revision Magnitude",
     "revision_magnitude_distribution", None, "upper left"),
]


def summarize(name, x):
    row = {"variable": name, "N": len(x), "mean": x.mean(), "SD": x.std(ddof=1), "min": x.min()}
    row.update({f"p{p}": np.percentile(x, p) for p in PERCENTILES})
    row["max"] = x.max()
    row["median"] = row.pop("p50")
    return row


def plot(x, xlabel, title, stem, xlim, legend_loc):
    lo, hi = xlim if xlim else (0.0, np.ceil(x.max() * 10) / 10)
    bins = np.linspace(lo, hi, 26 if xlim else int(round((hi - lo) / 0.05)) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    counts, edges, _ = ax.hist(x, bins=bins, color=BAR, alpha=0.85, edgecolor="white", linewidth=1)
    grid = np.linspace(lo, hi, 400)
    width = edges[1] - edges[0]
    ax.plot(grid, gaussian_kde(x)(grid) * len(x) * width, color=KDE_COLOR, linewidth=1.5,
            label="Kernel density (scaled to counts)")
    mean, median = x.mean(), np.median(x)
    ax.axvline(mean, color=INK, linewidth=1.2, label=f"Mean = {mean:.3f}")
    ax.axvline(median, color=INK_SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)),
               label=f"Median = {median:.3f}")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Participants")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc, labelcolor=INK_SECONDARY)
    if title:
        ax.set_title(f"{title}\nN = {len(x)} participants, all-mpnet-base-v2",
                     fontsize=10, loc="left")
    save(fig, stem)


def main():
    df = pd.read_csv(LONG_PATH)
    df = df[df["model"] == MODEL_NAME].sort_values("row_id")
    if not (df["status"] == "ok").all():
        raise ValueError("Unexpected non-ok status")
    rows = []
    for col, xlabel, title, stem, xlim, legend_loc in FIGURES:
        x = df[col].to_numpy(dtype=float)
        if xlim and (x.min() < xlim[0] or x.max() > xlim[1]):
            raise ValueError(f"{col} outside plotted range")
        plot(x, xlabel, title, stem, xlim, legend_loc)
        rows.append(summarize(col, x))
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows)[["variable", "N", "mean", "SD", "min", "p1", "p5", "p10", "p25",
                                  "median", "p75", "p90", "p95", "p99", "max"]]
    summary.to_csv(RESULTS_DIR / "descriptive_distribution_summary.csv", index=False)
    print(summary.round(4).set_index("variable").T.to_string())


if __name__ == "__main__":
    main()
