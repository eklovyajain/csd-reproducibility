# Reproducing the results

All commands are run from the repository root.

## A. Environment

Tested with Python 3.11.5.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## B. Default route: exact reproduction from the shipped caches

This route needs no OpenAI API key and does not re-embed any participant
text. `generate_crossmodel_embeddings.py` is **not** part of this route: the
caches in `data/embedding_cache/` are complete for all 451 participants and
all four models, and every analysis below reads them directly.

The two local Hugging Face models (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`)
are still loaded, at pinned revisions, for tokenizer-based truncation counts
and for the synthetic examples. On a fresh machine they are downloaded once
from Hugging Face (internet access, no account or key).

Run in this order (steps 2, 3 and 6 depend on earlier outputs):

```bash
# 1. Participant-level CSD scores for all four models, cross-model correlations
python analyze_crossmodel_agreement.py

# 2. Descriptive distribution of the primary (MPNet) scores
python plot_descriptive_distributions.py

# 3. Cross-model robustness figures and truncation sensitivity
python plot_crossmodel_robustness.py

# 4. Same-task critique permutation
python critique_permutation_test.py

# 5. Unrelated-text placebo
python unrelated_placebo_test.py

# 6. Translation invariance (and truncation summary)
python validation_checks.py

# 7. Near-degenerate diagnostic: smallest revision magnitudes
python small_revision_diagnostic.py

# 8. Near-degenerate diagnostic: smallest reference magnitudes
python small_reference_diagnostic.py

# Controlled synthetic examples
python reproduce_proposal_synthetic_table.py
```

`reproduce_proposal_synthetic_table.py` embeds its 13 short texts (one
initial decision, one critique, twelve revisions) with the pinned local
`all-mpnet-base-v2` model. It does not require an API key.

## C. Optional route: re-embedding from the source text

```bash
cp .env.example .env   # then set OPENAI_API_KEY in .env
python generate_crossmodel_embeddings.py            # all four models
python generate_crossmodel_embeddings.py all-mpnet-base-v2 all-MiniLM-L6-v2   # local models only
```

- The script embeds only texts missing from a cache, so to re-embed a model
  from scratch, move its `.npz` and `.meta.json` files out of
  `data/embedding_cache/` first.
- The two local models are pinned to exact Hugging Face revisions
  (`embedding.MODEL_REVISIONS`):
  `all-mpnet-base-v2` at `e8c3b32edf5434bc2275fc9bab85f82640a19130` and
  `all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Re-embedding with `text-embedding-3-large` or `text-embedding-3-small`
  requires `OPENAI_API_KEY` (see `.env.example`). The OpenAI models are not
  versioned snapshots, so their outputs are not guaranteed to stay identical
  over time; this is why the shipped caches are the default route.
- Re-running the script on complete caches makes no API calls but updates the
  `last_run_utc` field in each model's `.meta.json`.

Then run steps 1–8 above.

## D. Expected key values

| Analysis | Quantity | Value | File |
|---|---|---|---|
| Primary MPNet scores | N | 451 | `descriptive_distribution_summary.csv` |
| | mean / SD of `aligned_normalized` | 0.4156 / 0.1459 | |
| | median / min / max | 0.4155 / 0.0268 / 0.8762 | |
| Cross-model | Pearson range, mean | 0.627–0.882, 0.710 | `crossmodel_agreement_summary.txt` |
| | Spearman range, mean | 0.599–0.871, 0.686 | |
| Same-task permutation | actual mean | 0.4156 | `critique_permutation_summary.csv` |
| | wrong same-task mean | 0.4040 | |
| | difference | 0.0116 | |
| | permutations ≥ observed | 0 of 1,000 | |
| Unrelated placebo | actual / same-task / unrelated | 0.4156 / 0.4039 / 0.2604 | `unrelated_placebo_summary.csv` |
| | paired standardized difference (actual − unrelated) | 1.175 | |
| | participants with actual > unrelated average | 87.58% | |
| Translation invariance | max absolute difference, CSD outputs | 1.1e-16 to 2.2e-16 | `translation_invariance_check.csv` |

The two same-task means differ slightly because they are computed
differently. The permutation value (0.4040) averages over 1,000 random
critique reassignments. The placebo value (0.4039) averages, for each
participant, over all 450 other participants' critiques.

## E. Output map

| Script | Main outputs |
|---|---|
| `analyze_crossmodel_agreement.py` | `results/crossmodel_csd_long.csv`, `results/crossmodel_agreement_summary.txt` |
| `plot_descriptive_distributions.py` | `results/descriptive_distribution_summary.csv`, `figures/aligned_normalized_distribution.pdf` |
| `plot_crossmodel_robustness.py` | `figures/crossmodel_alignment_heatmap_pearson_spearman.pdf`, `figures/crossmodel_alignment_percentile_3panel.pdf`, `results/crossmodel_truncation_sensitivity.csv` |
| `critique_permutation_test.py` | `results/critique_permutation_summary.csv`, `results/critique_actual_percentile_ranks.csv` |
| `unrelated_placebo_test.py` | `results/unrelated_placebo_summary.csv`, `results/unrelated_placebo_participant_scores.csv`, `results/unrelated_axis_similarity.csv`, `figures/unrelated_placebo_three_conditions.pdf` |
| `validation_checks.py` | `results/translation_invariance_check.csv`, `results/truncation_summary.csv` |
| `small_revision_diagnostic.py` | `results/small_revision_diagnostic.csv`, `results/revision_magnitude_summary.csv`, `results/revision_magnitude_orientation.csv` |
| `small_reference_diagnostic.py` | `results/small_reference_diagnostic.csv`, `results/reference_magnitude_summary.csv`, `results/reference_magnitude_orientation.csv` |
| `reproduce_proposal_synthetic_table.py` | `results/proposal_synthetic_table.csv` |

## F. Notes

- Run scripts from the repository root; paths are relative to it.
- Some scripts also write additional diagnostic outputs and PNG copies of
  figures that are not part of the retained package.
- `results/small_revision_diagnostic_notes.md` and
  `results/small_reference_diagnostic_notes.md` record qualitative
  inspection of the smallest-magnitude cases. They are written by hand, not
  generated by the scripts.
- The embedding caches in `data/embedding_cache/` are part of the
  reproducibility package; each has a `.meta.json` recording the model,
  revision, dimension, and package versions.
