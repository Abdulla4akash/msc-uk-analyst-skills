"""
S2 — frozen sentence-embedding similarity.

* Frozen model: sentence-transformers/all-MiniLM-L6-v2 (see model_config.py / SEMANTIC_MODEL_SELECTION.md)
* No training on project data.
* Category vectors from frozen CATEGORY_LABELS only — no frozen lexicon resources.
* Long-document: deterministic chunking (256 tokens), mean-pool chunk embeddings.
* Batch-invariant: encoding one posting alone equals encoding in batch.
"""

import re
import numpy as np

# Enforce no lexicon import: this file must NOT import those resources.
# (Test checks source text for forbidden strings; comment itself must not contain them.)

from v4.config import CATEGORIES, CATEGORY_LABELS

from v4.semantic.model_config import (
    S2_MODEL_ID,
    S2_MAX_TOKENS,
    S2_CHUNK_TOKENS,
)

# Lazy model cache
_S2_MODEL = None
_S2_TOKENIZER = None
_S2_DEVICE = None


def _load_s2_model():
    global _S2_MODEL, _S2_TOKENIZER, _S2_DEVICE
    if _S2_MODEL is not None:
        return _S2_MODEL, _S2_TOKENIZER, _S2_DEVICE
    from sentence_transformers import SentenceTransformer
    import torch
    # device: mps if available else cpu
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = SentenceTransformer(S2_MODEL_ID, device=device)
    # SentenceTransformer wraps tokenizer; we also need raw tokenizer for chunking
    tokenizer = model.tokenizer
    model.eval()
    _S2_MODEL = model
    _S2_TOKENIZER = tokenizer
    _S2_DEVICE = device
    return model, tokenizer, device


def _chunk_ids(input_ids, chunk_size, overlap):
    if len(input_ids) <= chunk_size:
        return [input_ids]
    chunks = []
    step = chunk_size - overlap if overlap < chunk_size else chunk_size
    for start in range(0, len(input_ids), step):
        chunk = input_ids[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(input_ids):
            break
    return chunks


def _encode_texts_chunked(texts, batch_size=32):
    """
    Encode each text via deterministic chunking + mean-pool across chunks.
    Returns L2-normalised embeddings shape (n, dim).
    """
    model, tokenizer, device = _load_s2_model()
    import torch
    import numpy as np

    # Tokenise without special tokens to get raw ids for chunking
    all_embs = []
    for text in texts:
        # Handle empty
        if not text or not text.strip():
            text = " "
        enc = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
        ids = enc["input_ids"]
        chunks_ids = _chunk_ids(ids, S2_CHUNK_TOKENS, overlap=0)
        chunk_texts = []
        for cids in chunks_ids:
            # Decode chunk back to text for SentenceTransformer encode (or encode via ids)
            # Use tokenizer.decode to preserve text; SentenceTransformer handles internal tokenisation
            chunk_text = tokenizer.decode(cids, skip_special_tokens=True)
            if not chunk_text.strip():
                chunk_text = " "
            chunk_texts.append(chunk_text)
        # Encode chunks
        chunk_embs = model.encode(
            chunk_texts, batch_size=min(len(chunk_texts), batch_size), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        # chunk_embs shape (n_chunks, dim), already L2-normalised
        # Mean-pool across chunks, then renormalise
        mean_emb = chunk_embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        all_embs.append(mean_emb)
    return np.vstack(all_embs)


def get_category_embeddings():
    """
    Frozen category embeddings in CATEGORIES order, single chunk each.
    """
    cat_texts = [CATEGORY_LABELS[c] for c in CATEGORIES]
    return _encode_texts_chunked(cat_texts)


def embedding_scores(texts, cat_embs=None):
    """
    Cosine similarity matrix (n,13) between posting embeddings and category embeddings.
    Batch-invariant: score for posting X does not depend on other postings (tested).
    """
    if cat_embs is None:
        cat_embs = get_category_embeddings()
    post_embs = _encode_texts_chunked(texts)
    # Both are L2-normalised, cosine = dot
    S = post_embs @ cat_embs.T  # (n,13)
    # Also expose raw cosine in [-1,1]; thresholds will be tuned on this scale.
    # Clip for safety
    S = np.clip(S, -1.0, 1.0)
    return S


# For tests: ensure no fit() is exposed
def _assert_frozen():
    # Model is in eval mode, no training; this is documentation for test_embedding_model_frozen
    pass
