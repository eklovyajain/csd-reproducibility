"""
critique_permutation_test.py

Shuffled-critique placebo test for CSD on the deidentified N=451 data.

Question: is each participant's revision more aligned with the critique they
actually received than with critiques received by other participants?

Pipeline (all-mpnet-base-v2, canonical csd_core.csd_decompose):
  1. Build the full 451 x 451 matrix
         A[i, j] = aligned_normalized(I_i, C_j, F_i)
     The diagonal holds each participant's observed score.
  2. Permutation placebo: 1,000 random derangements of critiques (no
     participant keeps their own critique); for each, the mean and median
     of A[i, perm(i)] across participants. The observed mean/median is
     compared with these null distributions.
  3. Participant-level check: each participant's observed score vs. their
     mean score across the 1,000 permutations (descriptive only).
  4. Harder check: for each participant, the percentile rank of the actual
     critique among all 451 possible critiques -- the share of the 450
     wrong critiques it beats (ties count half).

Embeddings come from data/embedding_cache/all-mpnet-base-v2.npz (written by
generate_crossmodel_embeddings.py); if a text is missing from the cache it
is embedded locally with embedding.embed().

Outputs:
  results/critique_permutation_summary.csv
  results/critique_actual_percentile_ranks.csv
  figures/critique_permutation_null.{png,pdf}
  figures/critique_actual_percentile.{png,pdf}

Usage:
    python critique_permutation_test.py
"""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

from csd_core import csd_decompose

INPUT_PATH = Path("data/ProlificData451_csd_only.xlsx")
CACHE_PATH = Path("data/embedding_cache/all-mpnet-base-v2.npz")
MODEL_NAME = "all-mpnet-base-v2"
RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")

SEED = 20261024
N_PERMUTATIONS = 1000
TOP_THRESHOLDS = [50, 25, 10, 5]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BAR = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rows():
    ws = openpyxl.load_workbook(INPUT_PATH, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if list(rows[0]) != ["row_id", "initial_reasoning", "critique", "final_reasoning"]:
        raise ValueError(f"Unexpected columns: {rows[0]!r}")
    return rows[1:]


def load_vectors(rows):
    """Return I, C, F arrays (n x dim) for the rows, using the MPNet cache."""
    cache = {}
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH, allow_pickle=False) as npz:
            cache = dict(zip(npz["hashes"].tolist(), npz["vectors"]))
    missing = sorted({t for r in rows for t in r[1:] if text_hash(t) not in cache})
    if missing:
        from embedding import embed
        cache.update(zip(map(text_hash, missing), embed(missing, model_name=MODEL_NAME)))
    return [np.stack([cache[text_hash(r[k])] for r in rows]) for k in (1, 2, 3)]


def alignment_matrix(I, C, F):
    n = len(I)
    A = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            r = csd_decompose(I[i], C[j], F[i])
            if r["status"] != "ok":
                raise ValueError(f"Undefined decomposition for I_{i}, C_{j}: {r['status']}")
            A[i, j] = r["aligned_normalized"]
    return A


def random_derangement(n, rng):
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def run_permutations(A, rng):
    n = len(A)
    idx = np.arange(n)
    perm_means, perm_medians = np.empty(N_PERMUTATIONS), np.empty(N_PERMUTATIONS)
    participant_sum = np.zeros(n)
    for k in range(N_PERMUTATIONS):
        perm = random_derangement(n, rng)
        scores = A[idx, perm]
        perm_means[k], perm_medians[k] = scores.mean(), np.median(scores)
        participant_sum += scores
    return perm_means, perm_medians, participant_sum / N_PERMUTATIONS


def actual_percentiles(A):
    n = len(A)
    observed = np.diag(A)
    wrong = A[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    below = (wrong < observed[:, None]).sum(axis=1)
    ties = (wrong == observed[:, None]).sum(axis=1)
    return 100 * (below + 0.5 * ties) / (n - 1)


def summary_row(statistic, observed, null):
    return {
        "statistic": statistic,
        "observed": observed,
        "permutation_mean": null.mean(),
        "permutation_sd": null.std(ddof=1),
        "observed_minus_permutation_mean": observed - null.mean(),
        "standardized_difference": (observed - null.mean()) / null.std(ddof=1),
        "permutation_p_value": float(np.mean(null >= observed)),
        "n_permutations_at_least_observed": int(np.sum(null >= observed)),
        "n_permutations": len(null),
        "seed": SEED,
    }


def save(fig, stem):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def plot_null(perm_means, observed_mean, n):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(perm_means, bins=30, color=BAR, edgecolor="white", linewidth=1)
    ax.axvline(observed_mean, color=INK, linewidth=1.5)
    ax.text(observed_mean, ax.get_ylim()[1] * 0.95, f" observed mean = {observed_mean:.3f}",
            ha="left" if observed_mean < perm_means.max() else "right",
            va="top", fontsize=8, color=INK)
    ax.set_xlabel("Mean aligned_normalized across participants")
    ax.set_ylabel(f"Permutations (of {len(perm_means):,})")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Shuffled-critique placebo test\n"
                 f"Null distribution of mean alignment, N = {n} participants, MPNet",
                 fontsize=10, loc="left")
    save(fig, "critique_permutation_null")


def plot_percentiles(pct, n):
    median = float(np.median(pct))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(pct, bins=np.arange(0, 101, 5), color=BAR, edgecolor="white", linewidth=1)
    ax.axvline(50, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(50, ax.get_ylim()[1] * 0.97, " chance (50th)", ha="left", va="top",
            fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile of actual critique among all 451 critiques")
    ax.set_ylabel("Participants")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Rank of each participant's actual critique\n"
                 f"aligned_normalized vs. 450 critiques not received; median = {median:.1f}th percentile",
                 fontsize=10, loc="left")
    save(fig, "critique_actual_percentile")


def main():
    rows = load_rows()
    n = len(rows)
    I, C, F = load_vectors(rows)
    A = alignment_matrix(I, C, F)
    observed = np.diag(A)

    rng = np.random.default_rng(SEED)
    perm_means, perm_medians, shuffled_mean = run_permutations(A, rng)
    pct = actual_percentiles(A)

    summary = pd.DataFrame([
        summary_row("mean_aligned_normalized", observed.mean(), perm_means),
        summary_row("median_aligned_normalized", float(np.median(observed)), perm_medians),
    ])
    participant = pd.DataFrame({
        "row_id": [r[0] for r in rows],
        "observed_aligned_normalized": observed,
        "mean_shuffled_aligned_normalized": shuffled_mean,
        "observed_minus_shuffled": observed - shuffled_mean,
        "actual_critique_percentile": pct,
        **{f"top_{t}pct": pct >= 100 - t for t in TOP_THRESHOLDS},
    })
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "critique_permutation_summary.csv", index=False)
    participant.to_csv(RESULTS_DIR / "critique_actual_percentile_ranks.csv", index=False)

    plot_null(perm_means, observed.mean(), n)
    plot_percentiles(pct, n)

    print(f"N = {n}, permutations = {N_PERMUTATIONS} derangements, seed = {SEED}")
    print(summary.drop(columns=["seed"]).round(4).to_string(index=False))
    diff = participant["observed_minus_shuffled"]
    print(f"\nParticipant-level (descriptive): observed > own shuffled mean for "
          f"{100 * (diff > 0).mean():.1f}%; mean diff {diff.mean():.4f}, median diff {diff.median():.4f}")
    print(f"\nActual-critique percentile among all {n} critiques: "
          f"median {np.median(pct):.1f}, mean {pct.mean():.1f}")
    for t in TOP_THRESHOLDS:
        print(f"  in top {t}%: {100 * (pct >= 100 - t).mean():.1f}% of participants")
    off = A[~np.eye(n, dtype=bool)]
    print(f"\nAll wrong-critique scores: mean {off.mean():.4f}, SD {off.std():.4f}; "
          f"observed: mean {observed.mean():.4f}, SD {observed.std():.4f}")


if __name__ == "__main__":
    main()
