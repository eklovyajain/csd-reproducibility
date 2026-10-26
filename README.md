# Conceptual Shift Decomposition (CSD)

Conceptual Shift Decomposition (CSD) describes how reasoning is revised after
an external input such as a critique. Given sentence embeddings of initial
reasoning `I`, an external input `C`, and revised reasoning `F`, CSD
decomposes the revision direction `F - I` relative to the reference direction
`C - I` into an aligned component (along the reference direction) and a
residual component (perpendicular to it). The primary focal measure is the
normalized aligned orientation, `aligned_normalized`, in [0, 1]. These are
geometric descriptions of the direction of revision; they are not measures of
agreement with, acceptance of, or uptake of the input.

## Repository contents

| Component | Files |
|---|---|
| Core implementation | `csd_core.py` (`csd_decompose`), `embedding.py` (local sentence-transformers embedding, pinned model revisions) |
| Deidentified participant data (N = 451) | `data/ProlificData451_csd_only.xlsx` |
| Controlled synthetic examples | `data/proposal_synthetic_cases.csv`, `reproduce_proposal_synthetic_table.py` |
| Embedding caches (four models) | `data/embedding_cache/` |
| Participant-level CSD scores | `analyze_crossmodel_agreement.py` → `results/crossmodel_csd_long.csv` |
| Descriptive distribution | `plot_descriptive_distributions.py` |
| Cross-model robustness | `analyze_crossmodel_agreement.py`, `plot_crossmodel_robustness.py` |
| Same-task critique permutation | `critique_permutation_test.py` |
| Unrelated-text placebo | `unrelated_placebo_test.py`, `data/unrelated_placebo_texts.csv` |
| Translation-invariance check | `validation_checks.py` |
| Near-degenerate diagnostics | `small_revision_diagnostic.py`, `small_reference_diagnostic.py` |
| Unit-sphere and tangent-plane diagnostics | `spherical_geometry_diagnostics.py` |
| Saved outputs | `results/`, `figures/` |

## Data

`data/ProlificData451_csd_only.xlsx` contains one row per participant
(N = 451) with four columns: `row_id`, `initial_reasoning`, `critique`,
`final_reasoning`. It contains no participant identifiers.

The four embedding models are `all-mpnet-base-v2` (primary),
`all-MiniLM-L6-v2`, `text-embedding-3-large`, and `text-embedding-3-small`.
Their cached vectors, with model and version metadata, are in
`data/embedding_cache/`, along with the cached `all-mpnet-base-v2` vectors
for the 50 unrelated placebo texts.

## Reproducing the results

The shipped embedding caches allow exact reproduction of all reported results
without an OpenAI API key. Re-embedding from the source text is optional; it
requires internet access, and `OPENAI_API_KEY` for the two OpenAI models.

See [REPRODUCE.md](REPRODUCE.md) for the exact commands and expected values.

## Using `csd_decompose`

```python
from embedding import embed
from csd_core import csd_decompose

I_vec, C_vec, F_vec = embed([initial, critique, revision])
result = csd_decompose(I_vec, C_vec, F_vec)
```

With `d = F - I` and `ĉ = (C - I) / ‖C - I‖`:

| Key | Definition |
|---|---|
| `aligned_signed` | `d · ĉ`, signed projection onto the reference direction |
| `aligned_signed_normalized` | `(d · ĉ) / ‖d‖`, in [-1, 1] |
| `aligned_raw` | `\|d · ĉ\|` |
| `residual_raw` | `‖d - (d · ĉ)ĉ‖` |
| `revision_magnitude` | `‖d‖` |
| `aligned_normalized` | `aligned_raw / revision_magnitude`, in [0, 1] (primary measure) |
| `residual_normalized` | `residual_raw / revision_magnitude`, in [0, 1] |
| `cos_IF`, `cos_IC`, `cos_CF` | plain cosine similarities, for reference |
| `status` | `"ok"`, `"no_revision"` (F = I), or `"reference_undefined"` (C = I) |

When `status` is `"ok"`, `aligned_normalized**2 + residual_normalized**2 = 1`,
so the residual orientation is a complementary representation rather than a
separate measure.

## License

MIT; see [LICENSE](LICENSE).
