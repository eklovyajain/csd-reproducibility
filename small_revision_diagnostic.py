"""
small_revision_diagnostic.py

Boundary-condition diagnostic for very small revision magnitudes
M = ||F - I|| on the deidentified N=451 data (all-mpnet-base-v2, canonical
csd_core.csd_decompose). No threshold is chosen or applied.

Outputs:
  results/revision_magnitude_summary.csv
      Distribution of M across all participants.
  results/small_revision_diagnostic.csv
      The 20 smallest-M participants with CSD outputs, texts, and the
      qualitative judgments recorded in JUDGMENTS below.
  results/revision_magnitude_orientation.csv
      Pearson/Spearman correlations of M with aligned_normalized and
      residual_normalized, and descriptive statistics of aligned_normalized
      for the bottom 5% of M vs. the remaining 95%.

The qualitative judgments were made by reading initial, critique, and final
texts for the 20 cases; they are not computed.

Usage:
    python small_revision_diagnostic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import critique_permutation_test as perm
from csd_core import csd_decompose

RESULTS_DIR = Path("results")
N_SMALLEST = 20
BOTTOM_PERCENT = 5
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# row_id -> (substantive_change_flag, orientation_interpretable, interpretation_note)
JUDGMENTS = {
    265: ("minimal", "caution",
          "Near-copy: two small wording edits plus one added phrase ('targeted marketing'). "
          "That phrase echoes the critique's marketing point, so the direction is not arbitrary, "
          "but it rests on a single phrase."),
    199: ("small_but_real", "caution",
          "Same position (Amazon); mostly rewritten in different words, adding an explicit "
          "control trade-off and a customer-insights point. Paraphrase likely dominates d."),
    194: ("clear_revision", "yes",
          "Decision reversal: website-only to Amazon plus website, with new reasoning. Small M "
          "reflects shared topic vocabulary, not lack of change."),
    127: ("clear_revision", "yes",
          "Same decision, but adds substantial new content (pricing, trust, premium-image risk) "
          "that engages the critique's trust point."),
    385: ("small_but_real", "caution",
          "Same position; largely a paraphrase of the initial argument, adding one sentence "
          "weighing drawbacks and a reviews point."),
    234: ("small_but_real", "yes",
          "Same position (website first); adds service-capacity and support-demand reasoning that "
          "the critique raised."),
    370: ("small_but_real", "caution",
          "Same position; shorter restatement of the initial reasoning with little new content."),
    177: ("small_but_real", "yes",
          "Same position; paraphrase plus a new advertising-cost argument."),
    400: ("small_but_real", "yes",
          "Same decision with different reasons (copying/knock-offs and marketing) replacing the "
          "initial ones."),
    129: ("small_but_real", "caution",
          "Same position; close paraphrase, adding Amazon data/competition risk and a future "
          "distribution-agreement condition."),
    255: ("clear_revision", "yes",
          "Same final choice, but the final text is restructured around a new cost-benefit "
          "comparison of advertising vs. Amazon."),
    256: ("clear_revision", "yes",
          "Decision reversal: both channels to website-only, citing critique-related risks."),
    376: ("small_but_real", "yes",
          "Same position; new reasons (price pressure, recommendation algorithm, targeted ads)."),
    343: ("small_but_real", "yes",
          "Same position; reasons reframed and a reversibility argument and influencer-marketing "
          "plan added."),
    82: ("clear_revision", "yes",
         "Decision reversal: Amazon to website-only."),
    26: ("clear_revision", "yes",
         "Decision reversal: both channels to website-only first, explicitly citing the critique."),
    270: ("clear_revision", "yes",
          "Decision reversal: Amazon to website-only, with new fee, brand, and data reasons."),
    410: ("small_but_real", "yes",
          "Same position; adds fee/pricing handling and a rebuttal of the premium-image concern."),
    250: ("small_but_real", "yes",
          "Same position; reasons shift to premium status and ad campaigns."),
    121: ("small_but_real", "yes",
          "Position moderated from Amazon toward a dual-channel plan, acknowledging the risks "
          "raised."),
}


def describe(x):
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "min": x.min(),
            "p25": np.percentile(x, 25), "median": np.median(x),
            "p75": np.percentile(x, 75), "max": x.max()}


def main():
    rows = perm.load_rows()
    I, C, F = perm.load_vectors(rows)
    scores = pd.DataFrame([csd_decompose(I[i], C[i], F[i]) for i in range(len(rows))])
    if not (scores["status"] == "ok").all():
        raise ValueError("Unexpected non-ok status")
    texts = pd.DataFrame(rows, columns=["row_id", "initial_reasoning", "critique", "final_reasoning"])
    df = pd.concat([texts, scores], axis=1)
    M = df["revision_magnitude"].to_numpy()

    summary = {"N": len(M), "min": M.min()}
    summary.update({f"p{p}": np.percentile(M, p) for p in PERCENTILES})
    summary["mean"] = M.mean()
    summary["max"] = M.max()
    summary = pd.DataFrame([summary]).rename(columns={"p50": "median"})
    summary = summary[["N", "min", "p1", "p5", "p10", "p25", "median", "mean",
                       "p75", "p90", "p95", "p99", "max"]]

    small = df.nsmallest(N_SMALLEST, "revision_magnitude").copy()
    if set(small["row_id"]) != set(JUDGMENTS):
        raise ValueError("Smallest-M participants differ from those judged")
    small["rank_by_M"] = np.arange(1, N_SMALLEST + 1)
    small[["substantive_change_flag", "orientation_interpretable", "interpretation_note"]] = [
        JUDGMENTS[r] for r in small["row_id"]]
    small = small[["rank_by_M", "row_id", "revision_magnitude", "aligned_normalized",
                   "residual_normalized", "aligned_signed_normalized", "cos_IF",
                   "substantive_change_flag", "orientation_interpretable", "interpretation_note",
                   "initial_reasoning", "final_reasoning", "critique"]]

    orient = []
    for k in ["aligned_normalized", "residual_normalized"]:
        y = df[k].to_numpy()
        orient.append({"quantity": f"pearson_r(revision_magnitude, {k})", "n": len(M),
                       "value": stats.pearsonr(M, y)[0]})
        orient.append({"quantity": f"spearman_rho(revision_magnitude, {k})", "n": len(M),
                       "value": stats.spearmanr(M, y)[0]})
    cutoff = np.percentile(M, BOTTOM_PERCENT)
    bottom = M <= cutoff
    a = df["aligned_normalized"].to_numpy()
    for label, mask in [(f"bottom_{BOTTOM_PERCENT}pct_M", bottom),
                        (f"remaining_{100 - BOTTOM_PERCENT}pct_M", ~bottom)]:
        for stat, v in describe(a[mask]).items():
            orient.append({"quantity": f"aligned_normalized_{stat}:{label}",
                           "n": int(mask.sum()), "value": v})
    orient.append({"quantity": f"M_cutoff_p{BOTTOM_PERCENT} (bottom group is M <= cutoff)",
                   "n": int(bottom.sum()), "value": cutoff})
    orient = pd.DataFrame(orient)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "revision_magnitude_summary.csv", index=False)
    small.to_csv(RESULTS_DIR / "small_revision_diagnostic.csv", index=False)
    orient.to_csv(RESULTS_DIR / "revision_magnitude_orientation.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).T.to_string(header=False))
        print()
        print(small.iloc[:, :9].round(4).to_string(index=False))
        print()
        print(small["substantive_change_flag"].value_counts().to_string())
        print(small["orientation_interpretable"].value_counts().to_string())
        print()
        print(orient.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
