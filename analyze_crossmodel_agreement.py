"""
analyze_crossmodel_agreement.py

Participant-level cross-model agreement in CSD orientation for the
deidentified N=451 data, using the embeddings cached by
generate_crossmodel_embeddings.py (no embedding calls are made here).

For every row and every cached model, computes csd_core.csd_decompose and
writes results/crossmodel_csd_long.csv with:
    row_id, model, aligned_normalized, residual_normalized,
    revision_magnitude, cos_IF, status

Then reports, on the unstandardized normalized scores:
  - Pearson and Spearman correlation matrices across models for
    aligned_normalized (primary) and residual_normalized (complementary:
    aligned_normalized**2 + residual_normalized**2 == 1, so the two are
    mechanically linked);
  - mean, lowest and highest pairwise agreement;
  - the proposal showcase rows (222, 25, 13) across models, with each
    row's within-model percentile rank as a secondary, scale-free view;
  - truncation / input-length checks recorded in the cache metadata.

The report is printed and also written to
results/crossmodel_agreement_summary.txt.

Usage:
    python analyze_crossmodel_agreement.py
"""

import csv
import hashlib
import io
import itertools
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from csd_core import csd_decompose

INPUT_PATH = Path("data/ProlificData451_csd_only.xlsx")
CACHE_DIR = Path("data/embedding_cache")
LONG_OUTPUT = Path("results/crossmodel_csd_long.csv")
SUMMARY_OUTPUT = Path("results/crossmodel_agreement_summary.txt")

MODELS = [
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "text-embedding-3-large",
    "text-embedding-3-small",
]
SHORT_NAMES = {
    "all-mpnet-base-v2": "mpnet",
    "all-MiniLM-L6-v2": "MiniLM",
    "text-embedding-3-large": "oai-large",
    "text-embedding-3-small": "oai-small",
}
SHOWCASE_ROWS = {222: "Case 1 (strongly aligned)", 25: "Case 2 (intermediate)",
                 13: "Case 3 (strong residual)"}
COLUMNS = ["row_id", "model", "aligned_normalized", "residual_normalized",
           "revision_magnitude", "cos_IF", "status"]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_stem(model_id):
    return model_id.replace("/", "__")


def load_rows(path=INPUT_PATH):
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if list(rows[0]) != ["row_id", "initial_reasoning", "critique", "final_reasoning"]:
        raise ValueError(f"Unexpected columns: {rows[0]!r}")
    return rows[1:]


def load_model(model_id):
    npz_path = CACHE_DIR / f"{cache_stem(model_id)}.npz"
    meta_path = CACHE_DIR / f"{cache_stem(model_id)}.meta.json"
    if not npz_path.exists():
        return None, None
    with np.load(npz_path, allow_pickle=False) as npz:
        cache = dict(zip(npz["hashes"].tolist(), npz["vectors"]))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return cache, meta


def score_model(model_id, cache, rows):
    out = []
    for row_id, I_text, C_text, F_text in rows:
        try:
            I, C, F = (cache[text_hash(t)] for t in (I_text, C_text, F_text))
        except KeyError:
            raise RuntimeError(f"{model_id}: row {row_id} has text missing from the cache")
        r = csd_decompose(I, C, F)
        out.append({"row_id": row_id, "model": model_id,
                    **{k: r[k] for k in COLUMNS[2:]}})
    return out


def pairwise(matrix):
    pairs = {(a, b): matrix.loc[a, b] for a, b in itertools.combinations(matrix.columns, 2)}
    lo = min(pairs, key=pairs.get)
    hi = max(pairs, key=pairs.get)
    return float(np.mean(list(pairs.values()))), (lo, pairs[lo]), (hi, pairs[hi])


def report_agreement(wide, label):
    for method in ("pearson", "spearman"):
        m = wide.corr(method=method)
        mean, (lo, lo_v), (hi, hi_v) = pairwise(m)
        print(f"\n{label} -- {method.capitalize()} correlation matrix")
        print(m.round(3).to_string())
        print(f"  mean pairwise {method}: {mean:.3f}")
        print(f"  lowest pair:  {lo[0]} vs {lo[1]} = {lo_v:.3f}")
        print(f"  highest pair: {hi[0]} vs {hi[1]} = {hi_v:.3f}")


def main():
    rows = load_rows()
    available, missing, metas, records = [], [], {}, []
    for model_id in MODELS:
        cache, meta = load_model(model_id)
        if cache is None:
            missing.append(model_id)
            continue
        available.append(model_id)
        metas[model_id] = meta
        records.extend(score_model(model_id, cache, rows))

    LONG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LONG_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    df = pd.DataFrame(records)
    df["model"] = df["model"].map(SHORT_NAMES)
    order = [SHORT_NAMES[m] for m in available]

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"Rows: {len(rows)} | models scored: {len(available)} of {len(MODELS)}")
        for m in available:
            meta = metas[m]
            print(f"  {SHORT_NAMES[m]:<10} = {m} (dim {meta.get('output_dimension')}, "
                  f"provider model {meta.get('provider_reported_models') or meta.get('provider_model_version') or meta.get('hf_snapshot_revision')})")
        if missing:
            print(f"  NOT AVAILABLE (no cached embeddings): {missing}")
        print("\nStatus counts by model:")
        print(df.groupby("model")["status"].value_counts().unstack(fill_value=0).loc[order].to_string())

        ok = df[df["status"] == "ok"]
        aligned = ok.pivot(index="row_id", columns="model", values="aligned_normalized")[order]
        residual = ok.pivot(index="row_id", columns="model", values="residual_normalized")[order]
        print(f"\nRows with status ok in every model: {aligned.dropna().shape[0]}")

        print("\nPer-model distribution of aligned_normalized:")
        print(aligned.describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

        report_agreement(aligned.dropna(), "aligned_normalized")
        print("\n(residual_normalized is complementary: aligned^2 + residual^2 = 1 within each model)")
        report_agreement(residual.dropna(), "residual_normalized")

        print("\nShowcase rows across models (percentile = within-model rank of aligned_normalized):")
        pct = aligned.rank(pct=True) * 100
        cos = ok.pivot(index="row_id", columns="model", values="cos_IF")[order]
        for row_id, label in SHOWCASE_ROWS.items():
            print(f"\n  row_id {row_id}: {label}")
            table = pd.DataFrame({
                "aligned_normalized": aligned.loc[row_id],
                "residual_normalized": residual.loc[row_id],
                "cos_IF": cos.loc[row_id],
                "aligned_pctile": pct.loc[row_id],
            }).round(3)
            print("    " + table.to_string().replace("\n", "\n    "))

        print("\nTruncation / input-length checks:")
        for m in available:
            meta = metas[m]
            if "texts_truncated" in meta:
                print(f"  {SHORT_NAMES[m]:<10} truncated {meta['texts_truncated']} of {meta['n_cached_texts']} texts "
                      f"(max_seq_length {meta['max_seq_length']}, longest text {meta['token_count_max']} tokens)")
            else:
                print(f"  {SHORT_NAMES[m]:<10} input limit {meta['input_token_limit']} tokens; longest text "
                      f"{meta['max_text_utf8_bytes']} UTF-8 bytes (an upper bound on tokens) -> "
                      f"{'may exceed' if meta['any_text_may_exceed_limit'] else 'no text can exceed the limit'}")

    report = buffer.getvalue()
    print(report, end="")
    SUMMARY_OUTPUT.write_text(report, encoding="utf-8")
    print(f"\nWrote {LONG_OUTPUT} ({len(records)} rows) and {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
