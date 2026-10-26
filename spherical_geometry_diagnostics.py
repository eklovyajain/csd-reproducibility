"""
spherical_geometry_diagnostics.py

Unit-sphere diagnostics for CSD on the deidentified N=451 data. All vectors
come from the shipped caches in data/embedding_cache/; nothing is embedded
(a text missing from a cache raises an error). Chord-based scores are
computed with the canonical csd_core.csd_decompose.

When I, C, F are unit vectors, M = ||F - I|| and K = ||C - I|| are chords
and the signed chord orientation cos(theta) = aligned_signed_normalized
decomposes exactly as

    cos(theta) = M*K/4 + sqrt(1 - M^2/4) * sqrt(1 - K^2/4) * cos(psi)

where M*K/4 is a curvature term and psi is the angle between the tangent
components of F and C at I:

    tF = F - (F . I) I,   tC = C - (C . I) I,
    cos(psi) = (tF . tC) / (||tF|| ||tC||).

|cos(psi)| is the tangent-plane heading counterpart of aligned_normalized.

Sections (all-mpnet-base-v2 unless stated):
  embedding_norms                  L2 norms of the I, C, F vectors, all four models
  unit_sphere_identity             M vs sqrt(2 - 2 cos_IF)
  spherical_decomposition          reconstruction error; curvature, K, scale,
                                   cos(psi), |cos(psi)|; chord vs tangent
  signed_negative_counts           cos(theta) < 0 and cos(psi) < 0
  same_task_specificity            actual minus mean over the other 450
                                   critiques, and its relation to M
  unrelated_placebo                actual / same-task / unrelated means
  unrelated_decomposition          curvature and heading terms for the 50
                                   unrelated texts; analytic random-direction value
  crossmodel_agreement             min / max / mean pairwise correlations
  tangent_vs_chord_within_model    rank correspondence in each model
  openai_normalization_sensitivity raw vs L2-normalized OpenAI vectors

The two OpenAI models are not unit-normalized to float precision, so their
tangent scores are computed after L2-normalizing I, C, F; chord scores use
the vectors as cached, as in analyze_crossmodel_agreement.py.

Outputs:
  results/spherical_geometry_summary.csv
  results/tangent_crossmodel_correlations.csv

Usage:
    python spherical_geometry_diagnostics.py
"""

import csv
import itertools
from math import exp, lgamma, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze_crossmodel_agreement as crossmodel
import unrelated_placebo_test as placebo
from csd_core import csd_decompose

RESULTS_DIR = Path("results")
PRIMARY = "all-mpnet-base-v2"
UNIT_TOLERANCE = 1e-6   # max |norm - 1| for "effectively unit-normalized"
COLUMNS = ["section", "quantity", "model", "measure", "n", "value", "note"]


def load_vectors(model_id, rows):
    """I, C, F (n x dim, float64) from a model's cache; never embeds."""
    cache, _ = crossmodel.load_model(model_id)
    if cache is None:
        raise FileNotFoundError(f"No cached embeddings for {model_id}")
    try:
        return [np.stack([cache[crossmodel.text_hash(r[k])] for r in rows]).astype(np.float64)
                for k in (1, 2, 3)]
    except KeyError:
        raise RuntimeError(f"{model_id}: a participant text is missing from the cache")


def load_unrelated():
    cache = placebo.load_cache(placebo.PLACEBO_CACHE)
    with open(placebo.PLACEBO_PATH, newline="", encoding="utf-8") as f:
        texts = [p["text"] for p in csv.DictReader(f)]
    try:
        return np.stack([cache[placebo.text_hash(t)] for t in texts]).astype(np.float64)
    except KeyError:
        raise RuntimeError("An unrelated placebo text is missing from the cache")


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def chord_matrix(I, refs, F):
    """Signed cos(theta) and aligned_normalized for every (participant, reference)."""
    signed = np.empty((len(I), len(refs)))
    aligned = np.empty((len(I), len(refs)))
    for i in range(len(I)):
        for k in range(len(refs)):
            r = csd_decompose(I[i], refs[k], F[i])
            if r["status"] != "ok":
                raise ValueError(f"Undefined decomposition (i={i}, k={k}): {r['status']}")
            signed[i, k] = r["aligned_signed_normalized"]
            aligned[i, k] = r["aligned_normalized"]
    return signed, aligned


def tangent_matrix(I, refs, F):
    """cos(psi) between tF_i and the tangent component of each reference at I_i."""
    tF = F - (F * I).sum(axis=1, keepdims=True) * I
    RI = I @ refs.T                                    # R_k . I_i
    tFR = tF @ refs.T - RI * (tF * I).sum(axis=1, keepdims=True)
    tR_sq = (refs ** 2).sum(axis=1)[None, :] - 2 * RI ** 2 + RI ** 2 * (I ** 2).sum(axis=1, keepdims=True)
    tF_norm, tR_norm = np.linalg.norm(tF, axis=1), np.sqrt(tR_sq)
    if tF_norm.min() < UNIT_TOLERANCE or tR_norm.min() < UNIT_TOLERANCE:
        raise ValueError("Near-zero tangent component: cos(psi) undefined")
    return tFR / (tF_norm[:, None] * tR_norm)


def tangent_actual(I, C, F):
    """cos(psi) with each participant's own critique (diagonal only)."""
    tF = F - (F * I).sum(axis=1, keepdims=True) * I
    tC = C - (C * I).sum(axis=1, keepdims=True) * I
    return (tF * tC).sum(axis=1) / (np.linalg.norm(tF, axis=1) * np.linalg.norm(tC, axis=1))


def random_direction_abs_cos(dim):
    """Analytic E|cos| between a fixed and a uniformly random direction in R^dim."""
    return exp(lgamma(dim / 2) - lgamma((dim + 1) / 2)) / sqrt(pi)


def describe(x):
    return {"mean": x.mean(), "sd": x.std(ddof=1), "median": np.median(x),
            "min": x.min(), "max": x.max()}


def correlations(x, y, label, ci=False):
    p, s = stats.pearsonr(x, y), stats.spearmanr(x, y)
    out = {f"pearson_r({label})": p.statistic, f"pearson_p({label})": p.pvalue}
    if ci:
        lo, hi = p.confidence_interval(0.95)
        out[f"pearson_r_95ci_low({label})"] = lo
        out[f"pearson_r_95ci_high({label})"] = hi
    out[f"spearman_rho({label})"] = s.statistic
    out[f"spearman_p({label})"] = s.pvalue
    return out


def main():
    rows = crossmodel.load_rows()
    n = len(rows)
    off = ~np.eye(n, dtype=bool)
    out = []

    def add(section, values, model=PRIMARY, measure="", n_obs=n, note=""):
        for q, v in values.items():
            out.append({"section": section, "quantity": q, "model": model, "measure": measure,
                        "n": n_obs, "value": float(v), "note": note})

    # Embedding norms, and the tangent / chord scores used for cross-model agreement
    vectors, tangent_scores, chord_scores = {}, {}, {}
    for model_id in crossmodel.MODELS:
        I, C, F = load_vectors(model_id, rows)
        norms = np.linalg.norm(np.concatenate([I, C, F]), axis=1)
        deviation = np.abs(norms - 1).max()
        is_unit = deviation < UNIT_TOLERANCE
        add("embedding_norms", {"min_norm": norms.min(), "max_norm": norms.max(),
                                "mean_norm": norms.mean(), "sd_norm": norms.std(ddof=1),
                                "max_abs_deviation_from_1": deviation,
                                "effectively_unit_normalized": is_unit},
            model=model_id, n_obs=len(norms),
            note=f"I, C, F vectors; unit if max |norm - 1| < {UNIT_TOLERANCE:g}")
        vectors[model_id] = (I, C, F)
        tangent_scores[model_id] = np.abs(tangent_actual(I, C, F) if is_unit
                                          else tangent_actual(unit(I), unit(C), unit(F)))
        chord_scores[model_id] = np.array([csd_decompose(I[i], C[i], F[i])["aligned_normalized"]
                                           for i in range(n)])
        if not is_unit:
            raw = np.abs(tangent_actual(I, C, F))
            add("openai_normalization_sensitivity",
                {"max_abs_difference_raw_vs_normalized": np.abs(raw - tangent_scores[model_id]).max(),
                 "pearson_r_raw_vs_normalized": stats.pearsonr(raw, tangent_scores[model_id]).statistic,
                 "spearman_rho_raw_vs_normalized": stats.spearmanr(raw, tangent_scores[model_id]).statistic},
                model=model_id, measure="tangent",
                note="|cos(psi)| from raw cached vectors vs after L2-normalizing I, C, F")

    # Primary model: unit-sphere identity and spherical decomposition
    I, C, F = vectors[PRIMARY]
    actual = [csd_decompose(I[i], C[i], F[i]) for i in range(n)]
    cos_theta = np.array([r["aligned_signed_normalized"] for r in actual])
    chord = np.array([r["aligned_normalized"] for r in actual])
    M = np.array([r["revision_magnitude"] for r in actual])
    cos_IF = np.array([r["cos_IF"] for r in actual])
    K = np.linalg.norm(C - I, axis=1)
    add("unit_sphere_identity",
        {"max_abs_difference(M, sqrt(2 - 2 cos_IF))": np.abs(M - np.sqrt(2 - 2 * cos_IF)).max(),
         "max_abs_difference(M^2, 2 - 2 cos_IF)": np.abs(M ** 2 - (2 - 2 * cos_IF)).max(),
         "spearman_rho(M, cos_IF)": stats.spearmanr(M, cos_IF).statistic})

    cos_psi = tangent_actual(I, C, F)
    curvature = M * K / 4
    scale = np.sqrt(1 - M ** 2 / 4) * np.sqrt(1 - K ** 2 / 4)
    error = np.abs(curvature + scale * cos_psi - cos_theta)
    add("spherical_decomposition",
        {"max_abs_reconstruction_error": error.max(), "mean_abs_reconstruction_error": error.mean(),
         "n_cos_psi_outside_unit_interval": int((np.abs(cos_psi) > 1 + 1e-8).sum())},
        note="cos(theta) vs M*K/4 + sqrt(1-M^2/4) sqrt(1-K^2/4) cos(psi), cos(psi) from tangent vectors")
    for name, x in [("curvature", curvature), ("K", K), ("scale", scale),
                    ("cos_psi", cos_psi), ("abs_cos_psi", np.abs(cos_psi))]:
        add("spherical_decomposition", {f"{name}_{k}": v for k, v in describe(x).items()})
    add("spherical_decomposition", correlations(chord, np.abs(cos_psi), "aligned_normalized, abs_cos_psi"))
    add("signed_negative_counts", {"n_cos_theta_negative": int((cos_theta < 0).sum())}, measure="chord")
    add("signed_negative_counts", {"n_cos_psi_negative": int((cos_psi < 0).sum())}, measure="tangent")

    # Same-task critiques (all 451 x 451) and unrelated texts (451 x 50)
    U = load_unrelated()
    _, chord_same = chord_matrix(I, C, F)
    unrel_signed, chord_unrel = chord_matrix(I, U, F)
    tangent_same = np.abs(tangent_matrix(I, C, F))
    unrel_psi = tangent_matrix(I, U, F)
    for measure, A, Z in [("chord", chord_same, chord_unrel),
                          ("tangent", tangent_same, np.abs(unrel_psi))]:
        act = np.diag(A)
        wrong = A[off].reshape(n, n - 1).mean(axis=1)
        unrel = Z.mean(axis=1)
        spec = act - wrong
        add("same_task_specificity",
            {"actual_mean": act.mean(), "same_task_wrong_mean": wrong.mean(),
             **{f"specificity_{k}": v for k, v in describe(spec).items()},
             "specificity_standardized_effect": spec.mean() / spec.std(ddof=1),
             "pct_actual_gt_same_task_wrong_mean": 100 * (spec > 0).mean(),
             **correlations(spec, M, "specificity, M", ci=True),
             **correlations(act, M, "actual, M"),
             **correlations(wrong, M, "same_task_wrong_mean, M")},
            measure=measure, note="specificity = actual minus mean over the other 450 critiques")
        diff = act - unrel
        add("unrelated_placebo",
            {"actual_mean": act.mean(), "same_task_wrong_mean": wrong.mean(), "unrelated_mean": unrel.mean(),
             "actual_minus_unrelated_mean": diff.mean(),
             "actual_minus_unrelated_standardized_effect": diff.mean() / diff.std(ddof=1),
             "pct_actual_gt_unrelated_mean": 100 * (diff > 0).mean()},
            measure=measure, note="unrelated = mean over the 50 unrelated texts")

    K_unrel = np.linalg.norm(U[None, :, :] - I[:, None, :], axis=2)
    curv_unrel = M[:, None] * K_unrel / 4
    scale_unrel = np.sqrt(1 - M[:, None] ** 2 / 4) * np.sqrt(1 - K_unrel ** 2 / 4)
    heading_unrel = scale_unrel * unrel_psi
    add("unrelated_decomposition",
        {"K_mean": K_unrel.mean(), "curvature_mean": curv_unrel.mean(), "scale_mean": scale_unrel.mean(),
         "cos_psi_mean": unrel_psi.mean(), "abs_cos_psi_mean": np.abs(unrel_psi).mean(),
         "scaled_heading_mean": heading_unrel.mean(), "cos_theta_mean": unrel_signed.mean(),
         "curvature_plus_scaled_heading_mean": curv_unrel.mean() + heading_unrel.mean(),
         "curvature_plus_scaled_heading_minus_cos_theta_mean":
             curv_unrel.mean() + heading_unrel.mean() - unrel_signed.mean(),
         "pct_cos_theta_positive": 100 * (unrel_signed > 0).mean(),
         "n_cos_theta_negative": int((unrel_signed < 0).sum()),
         "pct_cos_psi_positive": 100 * (unrel_psi > 0).mean()},
        n_obs=unrel_psi.size, note="all participant x unrelated-text pairs")
    tangent_dim = I.shape[1] - 1
    add("unrelated_decomposition",
        {f"random_direction_expected_abs_cos(dim={tangent_dim})": random_direction_abs_cos(tangent_dim)},
        n_obs=0, note="analytic, not empirical: E|cos| for a uniformly random direction "
                      "in the tangent space at I")

    # Cross-model agreement and within-model correspondence
    short = crossmodel.SHORT_NAMES
    pairs = []
    for measure, scores in [("tangent", tangent_scores), ("chord", chord_scores)]:
        wide = pd.DataFrame({short[m]: scores[m] for m in crossmodel.MODELS})
        for method in ("pearson", "spearman"):
            corr = wide.corr(method=method)
            values = []
            for a, b in itertools.combinations(wide.columns, 2):
                pairs.append({"measure": measure, "method": method, "model_a": a, "model_b": b,
                              "n": n, "value": corr.loc[a, b]})
                values.append(corr.loc[a, b])
            add("crossmodel_agreement", {f"{method}_min_pairwise": min(values),
                                         f"{method}_max_pairwise": max(values),
                                         f"{method}_mean_pairwise": np.mean(values)},
                model="all four", measure=measure)
    for model_id in crossmodel.MODELS:
        add("tangent_vs_chord_within_model",
            {"pearson_r": stats.pearsonr(tangent_scores[model_id], chord_scores[model_id]).statistic,
             "spearman_rho": stats.spearmanr(tangent_scores[model_id], chord_scores[model_id]).statistic},
            model=model_id, measure="abs_cos_psi vs aligned_normalized")

    summary = pd.DataFrame(out, columns=COLUMNS)
    pairs = pd.DataFrame(pairs)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "spherical_geometry_summary.csv", index=False)
    pairs.to_csv(RESULTS_DIR / "tangent_crossmodel_correlations.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 60):
        print(summary.drop(columns=["note"]).to_string(index=False))
        print()
        print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
