"""
reproduce_proposal_synthetic_table.py

Reproduces the proposal's 12-case synthetic illustration table from
data/proposal_synthetic_cases.csv (the source of truth for those cases).

For each case, computes via csd_core.csd_decompose (all-mpnet-base-v2):
  - aligned_normalized
  - residual_normalized
  - z_aligned, z_residual: z-scores of the two columns above across the 12
    cases, using the population SD (ddof=0)
  - cosine_initial_final: cosine similarity between I and F (cos_IF)

Rows are sorted by descending z_aligned. Writes the full-precision table to
results/proposal_synthetic_table.csv and prints it rounded to 3 decimals.

Usage:
    python reproduce_proposal_synthetic_table.py
"""

import csv
from pathlib import Path

import numpy as np

from csd_core import csd_decompose
from embedding import embed

INPUT_PATH = Path("data/proposal_synthetic_cases.csv")
OUTPUT_PATH = Path("results/proposal_synthetic_table.csv")
MODEL_NAME = "all-mpnet-base-v2"

COLUMNS = [
    "case_id",
    "final_decision",
    "aligned_normalized",
    "residual_normalized",
    "z_aligned",
    "z_residual",
    "cosine_initial_final",
]


def load_cases(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std(ddof=0)


def compute_table(cases):
    texts = []
    for c in cases:
        texts.extend([c["initial_decision"], c["critique"], c["final_decision"]])
    vecs = embed(texts, model_name=MODEL_NAME)

    results = [
        csd_decompose(vecs[i], vecs[i + 1], vecs[i + 2])
        for i in range(0, len(vecs), 3)
    ]

    # z-scores are only meaningful if every case has a defined decomposition.
    bad = [c["case_id"] for c, r in zip(cases, results) if r["status"] != "ok"]
    if bad:
        raise ValueError(f"Cases without a defined decomposition: {bad}")

    z_aligned = zscore([r["aligned_normalized"] for r in results])
    z_residual = zscore([r["residual_normalized"] for r in results])

    rows = [
        {
            "case_id": c["case_id"],
            "final_decision": c["final_decision"],
            "aligned_normalized": r["aligned_normalized"],
            "residual_normalized": r["residual_normalized"],
            "z_aligned": float(za),
            "z_residual": float(zr),
            "cosine_initial_final": r["cos_IF"],
        }
        for c, r, za, zr in zip(cases, results, z_aligned, z_residual)
    ]
    rows.sort(key=lambda row: row["z_aligned"], reverse=True)
    return rows


def write_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    header = (
        f"{'case_id':<9} {'aligned_n':>9} {'residual_n':>10} {'z_aligned':>9} "
        f"{'z_residual':>10} {'cos_IF':>7}  final_decision"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case_id']:<9} {row['aligned_normalized']:>9.3f} "
            f"{row['residual_normalized']:>10.3f} {row['z_aligned']:>9.3f} "
            f"{row['z_residual']:>10.3f} {row['cosine_initial_final']:>7.3f}  "
            f"{row['final_decision'][:60]}"
        )


def main():
    cases = load_cases(INPUT_PATH)
    rows = compute_table(cases)
    write_table(rows, OUTPUT_PATH)
    print_table(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
