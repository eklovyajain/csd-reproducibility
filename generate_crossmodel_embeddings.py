"""
generate_crossmodel_embeddings.py

Embeds the deidentified N=451 participant texts with each of the four
cross-model robustness models and caches the vectors locally, so
analyze_crossmodel_agreement.py can compute CSD without further API calls.

Input:  data/ProlificData451_csd_only.xlsx
        (row_id, initial_reasoning, critique, final_reasoning)
Output: data/embedding_cache/<model>.npz        (text SHA-256 -> vector)
        data/embedding_cache/<model>.meta.json  (model / version metadata)

- Every model embeds exactly the same text, read verbatim from the input
  file; nothing is cleaned, truncated, or otherwise altered here. (Local
  Sentence-Transformers models truncate long inputs internally; those
  counts are recorded in the metadata.)
- Texts are keyed by the SHA-256 of their UTF-8 bytes, and reruns embed
  only texts not already in a model's cache. The cache is saved after every
  batch, so an interrupted run resumes where it stopped.
- The cache stores no participant identifiers -- not even row_id; the
  analysis script maps rows to vectors by re-hashing the input text.
- The OpenAI API key is read from the environment or a local .env file and
  is never printed. It is needed only when an OpenAI cache is missing texts;
  with complete caches no API call is made.
- If a model's API call fails or returns something unexpected, that model
  is stopped and reported; no other model is substituted.

Usage:
    python generate_crossmodel_embeddings.py              # all four models
    python generate_crossmodel_embeddings.py MODEL [...]  # a subset
"""

import datetime
import hashlib
import json
import os
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import openpyxl

INPUT_PATH = Path("data/ProlificData451_csd_only.xlsx")
CACHE_DIR = Path("data/embedding_cache")
TEXT_FIELDS = ["initial_reasoning", "critique", "final_reasoning"]

MODELS = {
    "all-mpnet-base-v2": "sentence-transformers",
    "all-MiniLM-L6-v2": "sentence-transformers",
    "text-embedding-3-large": "openai",
    "text-embedding-3-small": "openai",
}

# Published per-input limits for the API models (tokens).
API_INPUT_TOKEN_LIMITS = {"openai": 8191}

OPENAI_BATCH_SIZE = 100
MAX_ATTEMPTS = 5


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_stem(model_id):
    return model_id.replace("/", "__")


def load_dotenv(path=Path(".env")):
    """Load KEY=value lines from .env into os.environ without echoing them."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(key.strip()):
            os.environ[key.strip()] = value


def redact(message):
    """Strip any API key value that might appear in an error message."""
    secret = os.environ.get("OPENAI_API_KEY")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def load_texts(path=INPUT_PATH):
    """Return the unique input texts (verbatim) in first-seen order."""
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    expected = ["row_id"] + TEXT_FIELDS
    if header != expected:
        raise ValueError(f"Unexpected columns {header!r}; expected {expected!r}")
    texts = {}
    for row in rows[1:]:
        for text in row[1:]:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty or non-text cell in input file")
            texts.setdefault(text_hash(text), text)
    return texts, len(rows) - 1


def load_cache(model_id):
    path = CACHE_DIR / f"{cache_stem(model_id)}.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as npz:
        return dict(zip(npz["hashes"].tolist(), npz["vectors"]))


def save_cache(model_id, cache):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_stem(model_id)}.npz"
    tmp = CACHE_DIR / f"{cache_stem(model_id)}.tmp.npz"
    hashes = sorted(cache)
    np.savez(
        tmp,
        hashes=np.array(hashes, dtype="<U64"),
        vectors=np.stack([cache[h] for h in hashes]).astype(np.float32),
    )
    os.replace(tmp, path)


def is_retryable(error):
    """Retry only rate limits and server-side errors, not auth/billing/4xx."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return True  # network errors and the like carry no status code


def with_retries(call, what):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as e:  # provider SDKs raise many error types
            if attempt == MAX_ATTEMPTS or not is_retryable(e):
                raise RuntimeError(f"{what} failed after {attempt} attempts: "
                                   f"{type(e).__name__}: {redact(str(e))[:300]}")
            wait = 2 ** attempt
            print(f"    {what}: {type(e).__name__}, retrying in {wait}s")
            time.sleep(wait)


def embed_sentence_transformers(model_id, texts):
    from embedding import embed

    vectors = embed(texts, model_name=model_id)  # canonical project path
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def sentence_transformers_metadata(model_id, texts, meta):
    """Truncation check plus model/package provenance for a local model."""
    from huggingface_hub import snapshot_download

    from embedding import MODEL_REVISIONS, load_model

    # Count each text's tokens (incl. special tokens) against the model's
    # max_seq_length, without truncating.
    model = load_model(model_id)
    lengths = [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    meta["max_seq_length"] = model.max_seq_length
    meta["token_count_max"] = max(lengths)
    meta["texts_truncated"] = sum(n > model.max_seq_length for n in lengths)

    meta["hf_repo"] = f"sentence-transformers/{model_id}"
    meta["hf_snapshot_revision"] = Path(
        snapshot_download(meta["hf_repo"], revision=MODEL_REVISIONS.get(model_id),
                          local_files_only=True)
    ).name
    meta["package"] = {
        "sentence-transformers": package_version("sentence-transformers"),
        "transformers": package_version("transformers"),
        "torch": package_version("torch"),
    }
    meta["embedding_settings"] = (
        "embedding.embed(): SentenceTransformer.encode defaults "
        "(no normalization, model's default max_seq_length)"
    )


def embed_openai_batch(client, model_id, batch, meta):
    response = with_retries(
        lambda: client.embeddings.create(model=model_id, input=batch),
        f"OpenAI {model_id}",
    )
    items = sorted(response.data, key=lambda d: d.index)
    if len(items) != len(batch):
        raise RuntimeError(f"OpenAI returned {len(items)} embeddings for {len(batch)} inputs")
    meta.setdefault("provider_reported_models", set()).add(response.model)
    meta["usage_prompt_tokens"] = meta.get("usage_prompt_tokens", 0) + response.usage.prompt_tokens
    return [np.asarray(d.embedding, dtype=np.float32) for d in items]


def run_model(model_id, provider, texts):
    cache = load_cache(model_id)
    todo = [h for h in texts if h not in cache]
    meta_path = CACHE_DIR / f"{cache_stem(model_id)}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model_id": model_id, "provider": provider})
    if isinstance(meta.get("provider_reported_models"), list):
        meta["provider_reported_models"] = set(meta["provider_reported_models"])
    print(f"[{model_id}] {len(texts)} unique texts, {len(cache)} cached, {len(todo)} to embed")

    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if provider == "sentence-transformers":
        # Local and cheap: embed any missing texts in one call, and always
        # recompute the truncation check over all texts.
        if todo:
            vectors = embed_sentence_transformers(model_id, [texts[h] for h in todo])
            cache.update(zip(todo, vectors))
            save_cache(model_id, cache)
        sentence_transformers_metadata(model_id, list(texts.values()), meta)
    else:
        # Upper bound on tokens: every token covers at least one UTF-8 byte.
        max_bytes = max(len(t.encode("utf-8")) for t in texts.values())
        limit = API_INPUT_TOKEN_LIMITS[provider]
        meta["input_token_limit"] = limit
        meta["max_text_utf8_bytes"] = max_bytes
        meta["any_text_may_exceed_limit"] = max_bytes > limit

        # Only create a client (and require OPENAI_API_KEY) when texts are missing.
        if todo:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
            meta["package"] = {"openai": package_version("openai")}
            meta["embedding_settings"] = "embeddings.create defaults (full native dimensions)"

        for start in range(0, len(todo), OPENAI_BATCH_SIZE):
            chunk = todo[start:start + OPENAI_BATCH_SIZE]
            vectors = embed_openai_batch(client, model_id, [texts[h] for h in chunk], meta)
            cache.update(zip(chunk, vectors))
            save_cache(model_id, cache)
            print(f"    embedded {min(start + OPENAI_BATCH_SIZE, len(todo))}/{len(todo)}")

    dims = {v.shape[0] for v in cache.values()}
    if len(dims) != 1:
        raise RuntimeError(f"Inconsistent embedding dimensions in cache: {sorted(dims)}")
    norms = np.linalg.norm(np.stack(list(cache.values())), axis=1)
    meta["output_dimension"] = dims.pop()
    meta["vector_norm_min"] = float(norms.min())
    meta["vector_norm_max"] = float(norms.max())
    meta["n_cached_texts"] = len(cache)
    meta.setdefault("first_run_utc", started)
    meta["last_run_utc"] = started
    meta["python"] = sys.version.split()[0]
    meta["numpy"] = np.__version__
    if "provider_reported_models" in meta:
        meta["provider_reported_models"] = sorted(meta["provider_reported_models"])
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[{model_id}] done: {len(cache)} vectors, dim {meta['output_dimension']}")


def main():
    load_dotenv()
    selected = sys.argv[1:] or list(MODELS)
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Choose from {list(MODELS)}")

    texts, n_rows = load_texts()
    print(f"Loaded {n_rows} rows, {len(texts)} unique texts from {INPUT_PATH}\n")

    failures = {}
    for model_id in selected:
        try:
            run_model(model_id, MODELS[model_id], texts)
        except Exception as e:
            failures[model_id] = redact(f"{type(e).__name__}: {e}")[:400]
            print(f"[{model_id}] STOPPED: {failures[model_id]}")
        print()

    if failures:
        print("Models not completed:")
        for model_id, reason in failures.items():
            print(f"  {model_id}: {reason}")
        sys.exit(1)
    print("All selected models completed.")


if __name__ == "__main__":
    main()
