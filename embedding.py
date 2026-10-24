"""
embedding.py

Text-embedding bridge for Conceptual Shift Decomposition (CSD).

csd_core.py works purely in vector space. This module turns real text
(an initial decision, a critique, a revision, ...) into the vectors that
csd_core.csd_decompose() expects, using sentence-transformers.

The embedding model is a swappable parameter, not a hard-coded choice, so
different sentence-transformers models (e.g. "all-mpnet-base-v2" vs. the
smaller/faster "all-MiniLM-L6-v2") can be compared later without touching
csd_core.py at all.
"""

from sentence_transformers import SentenceTransformer

# Cache of already-loaded models, keyed by model_name, so repeated calls to
# embed() -- even across different texts -- don't reload the model (and its
# weights) from disk/network every time.
_model_cache = {}

# Hugging Face snapshot revisions used for the reported results, pinned so a
# fresh re-embedding loads the same weights rather than the latest version.
MODEL_REVISIONS = {
    "all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


def load_model(model_name):
    """Load a sentence-transformers model once, at its pinned revision if any."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(
            model_name, revision=MODEL_REVISIONS.get(model_name)
        )
    return _model_cache[model_name]


def embed(texts, model_name="all-mpnet-base-v2"):
    """
    Embed a list of strings into sentence embedding vectors.

    Parameters
    ----------
    texts : list of str
        The sentences/passages to embed.
    model_name : str, optional
        Name of the sentence-transformers model to use (default
        "all-mpnet-base-v2"). Any model name accepted by
        `sentence_transformers.SentenceTransformer(...)` works, e.g.
        "all-MiniLM-L6-v2" for a smaller/faster alternative. Kept as a
        parameter (rather than hard-coded) so different embedding models
        can be swapped in and compared.

    Returns
    -------
    numpy.ndarray
        Array of shape (len(texts), embedding_dim), one row per input
        string, in the same order as `texts`.

    Notes
    -----
    Models are loaded once per model_name and cached in-process (in
    `_model_cache`), so calling embed() repeatedly -- e.g. once per piece
    of text in a CSD pipeline -- only pays the model-loading cost the
    first time for each model_name.
    """
    model = load_model(model_name)
    return model.encode(texts, convert_to_numpy=True)
