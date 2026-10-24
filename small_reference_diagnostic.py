"""
small_reference_diagnostic.py

Boundary-condition diagnostic for very small critique-reference magnitudes
||C - I|| on the deidentified N=451 data (all-mpnet-base-v2, canonical
csd_core.csd_decompose). No threshold is chosen or applied.

Outputs:
  results/reference_magnitude_summary.csv
      Distribution of reference_magnitude = ||C - I|| across all participants.
  results/small_reference_diagnostic.csv
      The 20 smallest-reference participants with CSD outputs, texts, and the
      qualitative judgments recorded in JUDGMENTS below.
  results/reference_magnitude_orientation.csv
      Pearson/Spearman correlations of reference_magnitude with
      aligned_normalized and residual_normalized, and descriptive statistics
      of aligned_normalized for the bottom 5% vs. the remaining 95%.

Embeddings come from the existing MPNet cache (data/embedding_cache/); no
new embeddings are created. The qualitative judgments were made by reading
the initial, critique, and final texts for the 20 cases; they are not
computed.

Usage:
    python small_reference_diagnostic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import critique_permutation_test as perm
from csd_core import csd_decompose
from small_revision_diagnostic import PERCENTILES, describe

RESULTS_DIR = Path("results")
N_SMALLEST = 20
BOTTOM_PERCENT = 5

# row_id -> (critique_distinctiveness, reference_interpretable, interpretation_note)
JUDGMENTS = {
    274: ("clearly_distinct", "yes",
          "Challenges the Amazon-decline premise and adds marketing, service, and returns capacity."),
    26: ("small_but_real", "caution",
         "Critique restates the initial plan and reuses risks the initial already named (data use, "
         "brand dilution); the new element is mainly the commoditization argument."),
    101: ("clearly_distinct", "caution",
          "Initial text summarizes the scenario without making a recommendation, so the critique "
          "addresses a decision that was not stated; its content (support capacity, dilution) is "
          "distinct."),
    199: ("small_but_real", "yes",
          "Shares the initial's control/premium vocabulary but adds discounting pressure and Amazon "
          "data/rules dependence."),
    193: ("clearly_distinct", "yes",
          "Questions the copying-risk evidence and raises website traffic and trust at $950."),
    388: ("clearly_distinct", "yes",
          "Engages the copying premise, then adds marketing skills/budget and fit with a "
          "lower-price, higher-volume product."),
    114: ("clearly_distinct", "yes",
          "Adds profitability, control of pricing/service/returns, and brand dilution risks absent "
          "from the initial."),
    444: ("clearly_distinct", "yes",
          "Argues the $950 model already shifts the brand away from exclusivity and raises "
          "awareness/volume risk."),
    218: ("clearly_distinct", "yes",
          "Challenges demand at $950 and adds service, reviews, data, and dependence risks."),
    248: ("clearly_distinct", "yes",
          "Questions whether a lower-priced bike needs a high-end experience; adds visibility, "
          "selling-cost, and fulfillment capacity points."),
    89: ("clearly_distinct", "yes",
         "Challenges the marketing-drives-traffic assumption and adds high-volume service "
         "readiness."),
    70: ("clearly_distinct", "yes",
         "Challenges brand-trust transfer from the $4,900 model and the Amazon-low-quality claim."),
    21: ("clearly_distinct", "yes",
         "Argues the $950 model already shifts the brand image; adds volume, marketing, and "
         "service capacity."),
    137: ("clearly_distinct", "yes",
          "Challenges guilt-by-association and adds search behavior, awareness, and capacity "
          "points."),
    87: ("clearly_distinct", "yes",
         "Adds reach limits, dilution from the lower price itself, and support/delivery strain."),
    256: ("small_but_real", "yes",
          "Overlaps with the initial's premium-protection framing but adds the $4,900 price "
          "comparison and Amazon control of search, reviews, and data."),
    127: ("clearly_distinct", "yes",
          "Adds trust in an unfamiliar brand, premium-image damage, and data/control loss."),
    311: ("clearly_distinct", "yes",
          "Directly rebuts the price-outlier claim and adds discovery and capability risks."),
    105: ("clearly_distinct", "yes",
          "Challenges substituting social ads for Amazon traffic and adds volume, service, and "
          "trust barriers."),
    116: ("clearly_distinct", "yes",
          "Questions inevitable brand damage and adds the audience shift, awareness, and marketing "
          "capability points."),
}


def main():
    rows = perm.load_rows()
    I, C, F = perm.load_vectors(rows)
    scores = pd.DataFrame([csd_decompose(I[i], C[i], F[i]) for i in range(len(rows))])
    if not (scores["status"] == "ok").all():
        raise ValueError("Unexpected non-ok status")
    texts = pd.DataFrame(rows, columns=["row_id", "initial_reasoning", "critique", "final_reasoning"])
    df = pd.concat([texts, scores], axis=1)
    df["reference_magnitude"] = np.linalg.norm(C.astype(np.float64) - I.astype(np.float64), axis=1)
    R = df["reference_magnitude"].to_numpy()

    summary = {"N": len(R), "min": R.min()}
    summary.update({f"p{p}": np.percentile(R, p) for p in PERCENTILES})
    summary["mean"] = R.mean()
    summary["max"] = R.max()
    summary = pd.DataFrame([summary]).rename(columns={"p50": "median"})
    summary = summary[["N", "min", "p1", "p5", "p10", "p25", "median", "mean",
                       "p75", "p90", "p95", "p99", "max"]]

    small = df.nsmallest(N_SMALLEST, "reference_magnitude").copy()
    if set(small["row_id"]) != set(JUDGMENTS):
        raise ValueError("Smallest-reference participants differ from those judged")
    small["rank_by_reference_magnitude"] = np.arange(1, N_SMALLEST + 1)
    small[["critique_distinctiveness", "reference_interpretable", "interpretation_note"]] = [
        JUDGMENTS[r] for r in small["row_id"]]
    small = small[["rank_by_reference_magnitude", "row_id", "reference_magnitude",
                   "revision_magnitude", "aligned_normalized", "residual_normalized",
                   "aligned_signed_normalized", "cos_IC", "critique_distinctiveness",
                   "reference_interpretable", "interpretation_note",
                   "initial_reasoning", "critique", "final_reasoning"]]

    orient = []
    for k in ["aligned_normalized", "residual_normalized"]:
        y = df[k].to_numpy()
        orient.append({"quantity": f"pearson_r(reference_magnitude, {k})", "n": len(R),
                       "value": stats.pearsonr(R, y)[0]})
        orient.append({"quantity": f"spearman_rho(reference_magnitude, {k})", "n": len(R),
                       "value": stats.spearmanr(R, y)[0]})
    orient.append({"quantity": "pearson_r(reference_magnitude, revision_magnitude)", "n": len(R),
                   "value": stats.pearsonr(R, df["revision_magnitude"])[0]})
    cutoff = np.percentile(R, BOTTOM_PERCENT)
    bottom = R <= cutoff
    a = df["aligned_normalized"].to_numpy()
    for label, mask in [(f"bottom_{BOTTOM_PERCENT}pct_reference", bottom),
                        (f"remaining_{100 - BOTTOM_PERCENT}pct_reference", ~bottom)]:
        for stat, v in describe(a[mask]).items():
            orient.append({"quantity": f"aligned_normalized_{stat}:{label}",
                           "n": int(mask.sum()), "value": v})
    orient.append({"quantity": f"reference_cutoff_p{BOTTOM_PERCENT} (bottom group is <= cutoff)",
                   "n": int(bottom.sum()), "value": cutoff})
    orient = pd.DataFrame(orient)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "reference_magnitude_summary.csv", index=False)
    small.to_csv(RESULTS_DIR / "small_reference_diagnostic.csv", index=False)
    orient.to_csv(RESULTS_DIR / "reference_magnitude_orientation.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).T.to_string(header=False))
        print()
        print(small.iloc[:, :10].round(4).to_string(index=False))
        print()
        print(small["critique_distinctiveness"].value_counts().to_string())
        print(small["reference_interpretable"].value_counts().to_string())
        print()
        print(orient.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
