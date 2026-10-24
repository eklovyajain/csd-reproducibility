"""
plot_crossmodel_robustness.py

Cross-model robustness figures and tables for aligned_normalized, built from
results/crossmodel_csd_long.csv (written by analyze_crossmodel_agreement.py).
No embeddings are computed here. residual_normalized is not plotted: within
each model it is sqrt(1 - aligned_normalized**2), so it carries no
independent information.

Outputs:
  figures/crossmodel_alignment_heatmap.{png,pdf}                  Pearson
  figures/crossmodel_alignment_heatmap_pearson_spearman.{png,pdf} split
  figures/crossmodel_alignment_scatter_raw_<model>.{png,pdf}      x3
  figures/crossmodel_alignment_scatter_percentile_<model>.{png,pdf} x3
  figures/crossmodel_alignment_percentile_3panel.{png,pdf}
  results/crossmodel_truncation_sensitivity.csv
  results/robust_high_alignment_candidates.csv

Truncation flags come from re-tokenizing the input texts with the two local
models' tokenizers (no truncation) and comparing against each model's
max_seq_length -- the same check generate_crossmodel_embeddings.py records
as counts.

Usage:
    python plot_crossmodel_robustness.py
"""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

LONG_PATH = Path("results/crossmodel_csd_long.csv")
INPUT_PATH = Path("data/ProlificData451_csd_only.xlsx")
FIG_DIR = Path("figures")
RESULTS_DIR = Path("results")

MODELS = {
    "all-mpnet-base-v2": "MPNet",
    "all-MiniLM-L6-v2": "MiniLM",
    "text-embedding-3-large": "OpenAI Large",
    "text-embedding-3-small": "OpenAI Small",
}
FILE_TAGS = {"MiniLM": "minilm", "OpenAI Large": "openai_large", "OpenAI Small": "openai_small"}
REFERENCE = "MPNet"
LOCAL_MODELS = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]
TEXT_FIELDS = ["initial_reasoning", "critique", "final_reasoning"]

# Restrained chart chrome: neutral ink, hairline grid, one blue for marks,
# and a single-hue light->dark blue ramp for correlation magnitude.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
POINT = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CORR_RANGE = (0.5, 1.0)

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


def save(fig, stem):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def load_aligned():
    long = pd.read_csv(LONG_PATH)
    long = long[long["model"].isin(MODELS)]
    if (long["status"] != "ok").any():
        raise ValueError("Non-ok rows present; agreement statistics assume all rows are ok")
    wide = long.pivot(index="row_id", columns="model", values="aligned_normalized")
    cos = long.pivot(index="row_id", columns="model", values="cos_IF")
    labels = list(MODELS.values())
    wide = wide[list(MODELS)].rename(columns=MODELS)[labels]
    cos = cos[list(MODELS)].rename(columns=MODELS)[labels]
    return wide, cos


def percentiles(wide):
    return wide.rank(pct=True) * 100


def draw_heatmap(ax, values, labels, annotate_mask=None):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    im = ax.imshow(values, cmap=cmap, vmin=CORR_RANGE[0], vmax=CORR_RANGE[1])
    n = len(labels)
    ax.set_xticks(range(n), labels, rotation=30, ha="right")
    ax.set_yticks(range(n), labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2px-style surface gap between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            v = values[i, j]
            shade = (v - CORR_RANGE[0]) / (CORR_RANGE[1] - CORR_RANGE[0])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if shade > 0.55 else INK)
    return im


def heatmaps(wide):
    labels = list(wide.columns)
    n = len(wide)
    pearson = wide.corr(method="pearson").values
    spearman = wide.corr(method="spearman").values

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = draw_heatmap(ax, pearson, labels)
    ax.set_title("Cross-model agreement in aligned orientation\n"
                 f"Pearson r of aligned_normalized, N = {n} participants",
                 fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_SECONDARY, length=0)
    cb.set_label("Correlation", color=INK_SECONDARY)
    save(fig, "crossmodel_alignment_heatmap")

    split = np.tril(pearson, -1) + np.triu(spearman, 1) + np.eye(len(labels))
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    im = draw_heatmap(ax, split, labels)
    ax.set_title("Cross-model agreement in aligned orientation\n"
                 f"aligned_normalized, N = {n} participants",
                 fontsize=10, loc="left")
    ax.text(0.0, -0.24, "Below diagonal: Pearson r     Above diagonal: Spearman ρ",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, ha="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_SECONDARY, length=0)
    cb.set_label("Correlation", color=INK_SECONDARY)
    save(fig, "crossmodel_alignment_heatmap_pearson_spearman")


def scatter_panel(ax, x, y, lims, xlabel, ylabel, stats_text, ticks=None):
    ax.plot(lims, lims, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(x, y, s=9, color=POINT, alpha=0.35, linewidths=0, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=INK_SECONDARY)


def scatterplots(wide):
    n = len(wide)
    pct = percentiles(wide)
    raw_hi = float(np.ceil(wide.values.max() * 10) / 10)
    raw_lims = (0.0, max(raw_hi, 0.9))
    for other in [c for c in wide.columns if c != REFERENCE]:
        x, y = wide[REFERENCE], wide[other]
        r = x.corr(y, method="pearson")
        rho = x.corr(y, method="spearman")

        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        scatter_panel(ax, x, y, raw_lims, f"{REFERENCE} aligned_normalized",
                      f"{other} aligned_normalized",
                      f"r = {r:.2f}\nρ = {rho:.2f}\nN = {n}",
                      ticks=np.arange(0, raw_lims[1] + 1e-9, 0.2))
        ax.set_title(f"{REFERENCE} vs {other}: raw scores", fontsize=10, loc="left")
        save(fig, f"crossmodel_alignment_scatter_raw_{FILE_TAGS[other]}")

        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        scatter_panel(ax, pct[REFERENCE], pct[other], (0, 100),
                      f"{REFERENCE} percentile", f"{other} percentile",
                      f"ρ = {rho:.2f}\nN = {n}", ticks=range(0, 101, 25))
        ax.set_title(f"{REFERENCE} vs {other}: within-model percentiles",
                     fontsize=10, loc="left")
        save(fig, f"crossmodel_alignment_scatter_percentile_{FILE_TAGS[other]}")

    others = [c for c in wide.columns if c != REFERENCE]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
    for ax, other in zip(axes, others):
        rho = wide[REFERENCE].corr(wide[other], method="spearman")
        scatter_panel(ax, pct[REFERENCE], pct[other], (0, 100),
                      f"{REFERENCE} percentile", f"Comparison-model percentile",
                      f"ρ = {rho:.2f}", ticks=range(0, 101, 25))
        ax.set_title(f"{REFERENCE} vs {other}", fontsize=10, loc="left")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle("Within-model percentile of aligned_normalized, "
                 f"N = {n} participants (dashed line: equal percentile)",
                 fontsize=10, x=0.01, ha="left", y=1.02, color=INK)
    fig.tight_layout()
    save(fig, "crossmodel_alignment_percentile_3panel")


def truncation_flags():
    """Per-row, per-field truncation by the two local models (token counts)."""
    from embedding import load_model

    ws = openpyxl.load_workbook(INPUT_PATH, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    records = []
    for model_id in LOCAL_MODELS:
        model = load_model(model_id)
        limit = model.max_seq_length
        for row in rows[1:]:
            for field, text in zip(TEXT_FIELDS, row[1:]):
                n_tokens = len(model.tokenizer(text, truncation=False)["input_ids"])
                records.append({"row_id": row[0], "model": MODELS[model_id], "field": field,
                                "tokens": n_tokens, "truncated": n_tokens > limit})
    return pd.DataFrame(records)


def truncation_sensitivity(wide):
    flags = truncation_flags()
    hit = flags[flags["truncated"]]
    affected = sorted(hit["row_id"].unique())
    kept = wide.drop(index=affected)

    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        fp, kp = wide[a].corr(wide[b]), kept[a].corr(kept[b])
        fs = wide[a].corr(wide[b], method="spearman")
        ks = kept[a].corr(kept[b], method="spearman")
        rows.append({
            "model_pair": f"{a} vs {b}",
            "full_sample_pearson": round(fp, 4),
            "no_truncation_pearson": round(kp, 4),
            "pearson_difference": round(kp - fp, 4),
            "full_sample_spearman": round(fs, 4),
            "no_truncation_spearman": round(ks, 4),
            "spearman_difference": round(ks - fs, 4),
        })
    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "crossmodel_truncation_sensitivity.csv", index=False)

    texts_by_model = hit.groupby("model").size().to_dict()
    by_field = hit.drop_duplicates(["row_id", "field"]).groupby("field")["row_id"].nunique().to_dict()
    return {
        "affected_participants": len(affected),
        "remaining_participants": len(kept),
        "truncated_texts_by_model": texts_by_model,
        "participants_by_field": by_field,
        "table": table,
    }


def robust_candidates(wide, cos):
    pct = percentiles(wide)
    summary = pd.DataFrame(index=wide.index)
    for m in wide.columns:
        summary[f"pctile_{m}"] = pct[m]
    pcols = [f"pctile_{m}" for m in wide.columns]
    summary["min_pctile"] = summary[pcols].min(axis=1)
    summary["mean_pctile"] = summary[pcols].mean(axis=1)
    summary["pctile_range"] = summary[pcols].max(axis=1) - summary["min_pctile"]
    for m in wide.columns:
        summary[f"aligned_normalized_{m}"] = wide[m]
    for m in wide.columns:
        summary[f"cos_IF_{m}"] = cos[m]
    top = summary.sort_values(["min_pctile", "mean_pctile", "pctile_range"],
                              ascending=[False, False, True]).head(10)
    top = top.reset_index().round(4)
    top.columns = [c.replace(" ", "_") for c in top.columns]
    top.to_csv(RESULTS_DIR / "robust_high_alignment_candidates.csv", index=False)
    return top, summary


def main():
    wide, cos = load_aligned()
    heatmaps(wide)
    scatterplots(wide)
    trunc = truncation_sensitivity(wide)
    top, summary = robust_candidates(wide, cos)

    print(f"Participants: {len(wide)} | models: {list(wide.columns)}")
    print("\nTruncation (MPNet or MiniLM, any field):")
    print(f"  affected participants: {trunc['affected_participants']}")
    print(f"  remaining after exclusion: {trunc['remaining_participants']}")
    print(f"  truncated texts by model: {trunc['truncated_texts_by_model']}")
    print(f"  affected participants by field: {trunc['participants_by_field']}")
    print(trunc["table"].to_string(index=False))
    print("\nTop 10 cross-model high-alignment candidates:")
    print(top.to_string(index=False))
    print("\nCurrent Case 1 (row 222) for comparison:")
    print(summary.loc[[222]].round(3).to_string())


if __name__ == "__main__":
    main()
