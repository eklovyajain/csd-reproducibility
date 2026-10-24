"""
unrelated_placebo_test.py

Unrelated-text placebo test for CSD on the deidentified N=451 data
(all-mpnet-base-v2, canonical csd_core.csd_decompose).

For each participant i, aligned_normalized(I_i, C, F_i) is computed with
three kinds of reference text C:
  A. ACTUAL           C_i, the critique participant i received
  B. SAME-TASK WRONG  C_j for all 450 j != i (other participants' critiques)
  C. UNRELATED        U_k for all 50 texts in data/unrelated_placebo_texts.csv

Reports condition means/medians, paired differences with participant-level
bootstrap CIs, participant-level orderings, the actual critique's percentile
among the unrelated texts, and an axis-similarity diagnostic:
cos(C_i - I_i, C_j - I_i) and cos(C_i - I_i, U_k - I_i).

Participant embeddings come from data/embedding_cache/all-mpnet-base-v2.npz;
the unrelated texts are embedded with the same model and cached in
data/embedding_cache/unrelated_placebo__all-mpnet-base-v2.npz.

Outputs:
  results/unrelated_placebo_summary.csv
  results/unrelated_placebo_participant_scores.csv
  results/unrelated_axis_similarity.csv
  figures/unrelated_placebo_three_conditions.{png,pdf}
  figures/unrelated_axis_similarity.{png,pdf}

Usage:
    python unrelated_placebo_test.py
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

from csd_core import csd_decompose

INPUT_PATH = Path("data/ProlificData451_csd_only.xlsx")
PLACEBO_PATH = Path("data/unrelated_placebo_texts.csv")
CACHE_DIR = Path("data/embedding_cache")
PARTICIPANT_CACHE = CACHE_DIR / "all-mpnet-base-v2.npz"
PLACEBO_CACHE = CACHE_DIR / "unrelated_placebo__all-mpnet-base-v2.npz"
MODEL_NAME = "all-mpnet-base-v2"
RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")

BOOTSTRAP_SEED = 20261024
N_BOOTSTRAP = 10_000
TOP_THRESHOLDS = [50, 25, 10, 5]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2

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


def load_cache(path):
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as npz:
        return dict(zip(npz["hashes"].tolist(), npz["vectors"]))


def embed_with_cache(texts, path):
    """Vectors for texts, embedding (and caching) any not already cached."""
    cache = load_cache(path)
    missing = sorted({t for t in texts if text_hash(t) not in cache})
    if missing:
        from embedding import embed
        cache.update(zip(map(text_hash, missing),
                         np.asarray(embed(missing, model_name=MODEL_NAME), dtype=np.float32)))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hashes = sorted(cache)
        np.savez(path, hashes=np.array(hashes, dtype="<U64"),
                 vectors=np.stack([cache[h] for h in hashes]))
    return np.stack([cache[text_hash(t)] for t in texts])


def load_inputs():
    ws = openpyxl.load_workbook(INPUT_PATH, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if list(rows[0]) != ["row_id", "initial_reasoning", "critique", "final_reasoning"]:
        raise ValueError(f"Unexpected columns: {rows[0]!r}")
    rows = rows[1:]
    with open(PLACEBO_PATH, newline="", encoding="utf-8") as f:
        placebo = list(csv.DictReader(f))
    I = embed_with_cache([r[1] for r in rows], PARTICIPANT_CACHE)
    C = embed_with_cache([r[2] for r in rows], PARTICIPANT_CACHE)
    F = embed_with_cache([r[3] for r in rows], PARTICIPANT_CACHE)
    U = embed_with_cache([p["text"] for p in placebo], PLACEBO_CACHE)
    return [r[0] for r in rows], I, C, F, U, placebo


def alignment_matrix(I, refs, F):
    """M[i, k] = aligned_normalized(I_i, refs_k, F_i) via csd_decompose."""
    M = np.empty((len(I), len(refs)))
    for i in range(len(I)):
        for k in range(len(refs)):
            r = csd_decompose(I[i], refs[k], F[i])
            if r["status"] != "ok":
                raise ValueError(f"Undefined decomposition (i={i}, k={k}): {r['status']}")
            M[i, k] = r["aligned_normalized"]
    return M


def axis_cosines(I, refs, axis):
    """cos(axis_i, refs_k - I_i) for every participant i and reference k."""
    out = np.empty((len(I), len(refs)))
    for i in range(len(I)):
        dirs = refs - I[i]
        out[i] = dirs @ axis[i] / (np.linalg.norm(dirs, axis=1) * np.linalg.norm(axis[i]))
    return out


def percentile_among(observed, others):
    below = (others < observed[:, None]).sum(axis=1)
    ties = (others == observed[:, None]).sum(axis=1)
    return 100 * (below + 0.5 * ties) / others.shape[1]


def bootstrap_ci(values, rng):
    n = len(values)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def save(fig, stem):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def plot_conditions(scores):
    labels = ["Actual critique", "Same-task wrong\n(mean of 450)", "Unrelated text\n(mean of 50)"]
    cols = ["actual", "same_task_wrong_mean", "unrelated_mean"]
    data = scores[cols].values
    n = len(data)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = np.arange(3)
    for row in data:  # paired lines, one per participant
        ax.plot(x, row, color=INK_MUTED, alpha=0.06, linewidth=0.6, zorder=1)
    for k in range(3):
        jitter = rng.uniform(-0.07, 0.07, n)
        ax.scatter(np.full(n, x[k]) + jitter, data[:, k], s=6, color=SERIES[0],
                   alpha=0.35, linewidths=0, zorder=2)
        med = np.median(data[:, k])
        ax.plot([x[k] - 0.16, x[k] + 0.16], [med, med], color=INK, linewidth=2, zorder=3)
        ax.text(x[k] + 0.19, med, f"{med:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.4, 2.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("aligned_normalized")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Alignment with the actual critique, other participants' critiques,\n"
                 f"and unrelated texts (MPNet, N = {n}; grey lines link each participant, black bars = medians)",
                 fontsize=10, loc="left")
    save(fig, "unrelated_placebo_three_conditions")


def plot_axis_similarity(axis_df):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bins = np.linspace(min(axis_df.min()) - 0.02, 1.0, 50)
    series = [("same_task_axis_cosine_mean", "Other participants' critiques"),
              ("unrelated_axis_cosine_mean", "Unrelated texts")]
    for (col, label), color in zip(series, SERIES):
        ax.hist(axis_df[col], bins=bins, color=color, alpha=0.75, edgecolor="white",
                linewidth=0.8, label=label)
        med = axis_df[col].median()
        ax.axvline(med, color=color, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xlabel("Mean cosine between actual axis (C_i - I_i) and alternative axis (X - I_i)")
    ax.set_ylabel("Participants")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("How similar are the alternative reference directions to the actual one?\n"
                 f"Participant-level means, MPNet, N = {len(axis_df)} (dashed: medians)",
                 fontsize=10, loc="left")
    save(fig, "unrelated_axis_similarity")


def main():
    row_ids, I, C, F, U, placebo = load_inputs()
    n = len(row_ids)
    off = ~np.eye(n, dtype=bool)

    A = alignment_matrix(I, C, F)            # 451 x 451; diagonal = actual
    Z = alignment_matrix(I, U, F)            # 451 x 50 unrelated
    actual = np.diag(A)
    wrong = A[off].reshape(n, n - 1)

    scores = pd.DataFrame({
        "row_id": row_ids,
        "actual": actual,
        "same_task_wrong_mean": wrong.mean(axis=1),
        "same_task_wrong_median": np.median(wrong, axis=1),
        "unrelated_mean": Z.mean(axis=1),
        "unrelated_median": np.median(Z, axis=1),
        "actual_pctile_vs_same_task": percentile_among(actual, wrong),
        "actual_pctile_vs_unrelated": percentile_among(actual, Z),
    })

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = []
    for basis in ("mean", "median"):
        w, u = scores[f"same_task_wrong_{basis}"], scores[f"unrelated_{basis}"]
        for name, a, b in [("actual", actual, None), ("same_task_wrong", w, None), ("unrelated", u, None)]:
            summary.append({"section": f"condition_level ({basis} of comparison set)", "quantity": name,
                            "mean_across_participants": float(np.mean(a)),
                            "median_across_participants": float(np.median(a))})
        for name, a, b in [("actual_minus_same_task_wrong", actual, w),
                           ("actual_minus_unrelated", actual, u),
                           ("same_task_wrong_minus_unrelated", w, u)]:
            d = np.asarray(a) - np.asarray(b)
            lo, hi = bootstrap_ci(d, rng)
            summary.append({"section": f"paired_difference ({basis} of comparison set)", "quantity": name,
                            "mean_across_participants": d.mean(), "median_across_participants": np.median(d),
                            "sd_of_difference": d.std(ddof=1), "standardized_effect": d.mean() / d.std(ddof=1),
                            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                            "pct_participants_positive": 100 * (d > 0).mean()})
    w, u = scores["same_task_wrong_mean"], scores["unrelated_mean"]
    full = (actual > w) & (w > u)
    summary.append({"section": "ordering", "quantity": "actual > same_task_wrong > unrelated (means)",
                    "pct_participants_positive": 100 * full.mean()})
    for label, col in [("actual_pctile_vs_unrelated", "actual_pctile_vs_unrelated"),
                       ("actual_pctile_vs_same_task", "actual_pctile_vs_same_task")]:
        p = scores[col]
        summary.append({"section": "rank_check", "quantity": label,
                        "mean_across_participants": p.mean(), "median_across_participants": p.median(),
                        **{f"pct_top_{t}": 100 * (p >= 100 - t).mean() for t in TOP_THRESHOLDS}})

    same_axis = axis_cosines(I, C, C - I)
    same_axis_mean = np.where(off, same_axis, 0).sum(axis=1) / (n - 1)
    unrel_axis_mean = axis_cosines(I, U, C - I).mean(axis=1)
    axis_df = pd.DataFrame({"row_id": row_ids,
                            "same_task_axis_cosine_mean": same_axis_mean,
                            "unrelated_axis_cosine_mean": unrel_axis_mean})
    for col in ("same_task_axis_cosine_mean", "unrelated_axis_cosine_mean"):
        summary.append({"section": "axis_similarity", "quantity": col,
                        "mean_across_participants": axis_df[col].mean(),
                        "median_across_participants": axis_df[col].median()})
    diff = axis_df["same_task_axis_cosine_mean"] - axis_df["unrelated_axis_cosine_mean"]
    summary.append({"section": "axis_similarity", "quantity": "same_task_minus_unrelated",
                    "mean_across_participants": diff.mean(), "median_across_participants": diff.median(),
                    "pct_participants_positive": 100 * (diff > 0).mean()})

    summary = pd.DataFrame(summary)
    summary["bootstrap_seed"] = BOOTSTRAP_SEED
    summary["n_bootstrap"] = N_BOOTSTRAP
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "unrelated_placebo_summary.csv", index=False)
    scores.to_csv(RESULTS_DIR / "unrelated_placebo_participant_scores.csv", index=False)
    axis_df.to_csv(RESULTS_DIR / "unrelated_axis_similarity.csv", index=False)

    plot_conditions(scores)
    plot_axis_similarity(axis_df)

    meta = {"model": MODEL_NAME, "n_participants": n, "n_unrelated_texts": len(placebo),
            "bootstrap_seed": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP}
    print(json.dumps(meta))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns=["bootstrap_seed", "n_bootstrap"]).round(4).to_string(index=False))
    for a, b in [("actual", "same_task_wrong_mean"), ("actual", "unrelated_mean"),
                 ("same_task_wrong_mean", "unrelated_mean")]:
        print(f"{a} > {b}: {100 * (scores[a] > scores[b]).mean():.1f}%")


if __name__ == "__main__":
    main()
