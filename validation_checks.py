"""
validation_checks.py

Saves three small validation results:

  results/translation_invariance_check.csv
      Adding one random vector a to I, C, and F leaves every csd_decompose
      decomposition output unchanged (100 sampled participants).
  results/origin_diagnostic_summary.csv
      The origin-dependent "-I" diagnostic vs. the translation-invariant
      unrelated-text baseline.
  results/truncation_summary.csv
      Which participants/fields the local Sentence-Transformers models
      truncate, and how much excluding them moves cross-model correlations.

No embeddings are generated: participant and placebo vectors come from the
local MPNet caches (data/embedding_cache/), and truncation is measured with
the models' tokenizers only. The sampling (numpy default_rng(20261024),
100 participants without replacement, then one standard-normal shift per
participant in sample order) matches the interactive check exactly.

Usage:
    python validation_checks.py
"""

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

import critique_permutation_test as perm
import unrelated_placebo_test as placebo
from csd_core import csd_decompose

SEED = 20261024
SAMPLE_N = 100
RESULTS_DIR = Path("results")
DECOMPOSITION_KEYS = [
    "aligned_signed", "aligned_signed_normalized", "aligned_raw", "residual_raw",
    "revision_magnitude", "aligned_normalized", "residual_normalized",
]
COSINE_KEYS = ["cos_IF", "cos_IC", "cos_CF"]
MACHINE_TOLERANCE = 1e-12
TEXT_FIELDS = ["initial_reasoning", "critique", "final_reasoning"]


def abs_cos(a, b):
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def translation_and_origin_checks():
    _, I, C, F, U, _ = placebo.load_inputs()
    I, C, F, U = (x.astype(np.float64) for x in (I, C, F, U))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(I), size=SAMPLE_N, replace=False)

    max_diff = dict.fromkeys(DECOMPOSITION_KEYS + COSINE_KEYS, 0.0)
    statuses_ok = True
    shift_norms, minus_i_before, minus_i_after = [], [], []
    unrelated_before, unrelated_after, unrelated_max_diff = [], [], 0.0

    for i in sample:
        a = rng.standard_normal(I.shape[1])
        shift_norms.append(np.linalg.norm(a))
        r0 = csd_decompose(I[i], C[i], F[i])
        r1 = csd_decompose(I[i] + a, C[i] + a, F[i] + a)
        statuses_ok &= r0["status"] == r1["status"] == "ok"
        for k in max_diff:
            max_diff[k] = max(max_diff[k], abs(r0[k] - r1[k]))

        d = F[i] - I[i]
        minus_i_before.append(abs_cos(d, -I[i]))
        minus_i_after.append(abs_cos(d, -(I[i] + a)))

        u0 = np.array([csd_decompose(I[i], u, F[i])["aligned_normalized"] for u in U])
        u1 = np.array([csd_decompose(I[i] + a, u + a, F[i] + a)["aligned_normalized"] for u in U])
        unrelated_before.append(u0.mean())
        unrelated_after.append(u1.mean())
        unrelated_max_diff = max(unrelated_max_diff, np.max(np.abs(u0 - u1)))

    meta = {"sample_n": SAMPLE_N, "seed": SEED,
            "all_statuses_ok_before_and_after": statuses_ok,
            "mean_shift_norm": float(np.mean(shift_norms)),
            "mean_embedding_norm": float(np.mean(np.linalg.norm(I[sample], axis=1)))}

    rows = [{"output": k, "kind": "csd_decomposition", "max_absolute_difference": max_diff[k],
             "invariant_to_machine_precision": max_diff[k] < MACHINE_TOLERANCE}
            for k in DECOMPOSITION_KEYS]
    rows += [{"output": k, "kind": "raw_embedding_cosine_diagnostic (not expected to be invariant)",
              "max_absolute_difference": max_diff[k], "invariant_to_machine_precision": None}
             for k in COSINE_KEYS]
    rows += [{"output": f"meta:{k}", "kind": "metadata", "max_absolute_difference": None,
              "invariant_to_machine_precision": None, "value": v} for k, v in meta.items()]
    translation = pd.DataFrame(rows)

    # Full-sample (N=451) value of the -I diagnostic, as originally reported.
    d_all = F - I
    minus_i_full = np.mean(np.abs((d_all * -I).sum(axis=1))
                           / (np.linalg.norm(d_all, axis=1) * np.linalg.norm(I, axis=1)))
    origin_rows = [
        ("minus_I_diagnostic_original_full_sample", len(I), minus_i_full,
         "Origin-dependent diagnostic; not an intrinsic property of CSD."),
        ("minus_I_diagnostic_original_translation_sample", SAMPLE_N, np.mean(minus_i_before),
         "Origin-dependent diagnostic; not an intrinsic property of CSD."),
        ("minus_I_diagnostic_after_translation", SAMPLE_N, np.mean(minus_i_after),
         "Origin-dependent diagnostic; not an intrinsic property of CSD."),
        ("unrelated_text_baseline_before_translation", SAMPLE_N, np.mean(unrelated_before),
         "Translation-invariant property of the relative geometry; not an origin artifact."),
        ("unrelated_text_baseline_after_translation", SAMPLE_N, np.mean(unrelated_after),
         "Translation-invariant property of the relative geometry; not an origin artifact."),
        ("unrelated_text_scores_max_abs_difference_before_vs_after", SAMPLE_N, unrelated_max_diff,
         "Translation-invariant property of the relative geometry; not an origin artifact."),
    ]
    origin = pd.DataFrame(origin_rows, columns=["quantity", "n_participants", "value", "interpretation"])
    origin["note"] = ("|cos(F_i - I_i, -I_i)| averaged over participants; unrelated baseline = "
                      "mean aligned_normalized over the 50 unrelated texts. Why the unrelated "
                      "baseline is nonzero is not established.")
    return translation, origin


def truncation_summary():
    from embedding import load_model

    ws = openpyxl.load_workbook(perm.INPUT_PATH, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))[1:]
    counts, affected, fields_hit = {}, set(), set()
    critique_truncations = 0
    for model_id in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
        model = load_model(model_id)
        n = 0
        for r in rows:
            for field, text in zip(TEXT_FIELDS, r[1:]):
                if len(model.tokenizer(text, truncation=False)["input_ids"]) > model.max_seq_length:
                    n += 1
                    affected.add(r[0])
                    fields_hit.add(field)
                    critique_truncations += field == "critique"
        counts[model_id] = n
    sens = pd.read_csv(RESULTS_DIR / "crossmodel_truncation_sensitivity.csv")
    max_change = max(sens["pearson_difference"].abs().max(), sens["spearman_difference"].abs().max())
    return pd.DataFrame([{
        "total_participants": len(rows),
        "affected_participants": len(affected),
        "retained_participants": len(rows) - len(affected),
        "mpnet_truncated_texts": counts["all-mpnet-base-v2"],
        "minilm_truncated_texts": counts["all-MiniLM-L6-v2"],
        "critique_truncated_texts": critique_truncations,
        "affected_fields": "; ".join(f for f in TEXT_FIELDS if f in fields_hit),
        "maximum_change_in_pairwise_correlation_after_exclusion": round(float(max_change), 4),
    }])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    translation, origin = translation_and_origin_checks()
    translation.to_csv(RESULTS_DIR / "translation_invariance_check.csv", index=False)
    origin.to_csv(RESULTS_DIR / "origin_diagnostic_summary.csv", index=False)
    truncation = truncation_summary()
    truncation.to_csv(RESULTS_DIR / "truncation_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
        print(translation.to_string(index=False))
        print()
        print(origin.drop(columns=["note"]).to_string(index=False))
        print()
        print(truncation.T.to_string(header=False))


if __name__ == "__main__":
    main()
