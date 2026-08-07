"""
S3 — frozen zero-shot NLI.

* Frozen model: facebook/bart-large-mnli (see model_config.py)
* 13 deterministic hypotheses from CATEGORY_LABELS, template "This job requires {label}."
* Long-document: sentence/chunk split + MAX entailment aggregation.
* No lexicon use; multi-label independence (no cross-category softmax).
"""

import re
import numpy as np

from v4.config import CATEGORIES, CATEGORY_LABELS
from v4.semantic.model_config import S3_MODEL_ID, S3_CHUNK_TOKENS, NLI_HYPOTHESES_LIST

_NLI_MODEL = None
_NLI_TOKENIZER = None
_NLI_DEVICE = None


def _load_nli():
    global _NLI_MODEL, _NLI_TOKENIZER, _NLI_DEVICE
    if _NLI_MODEL is not None:
        return _NLI_MODEL, _NLI_TOKENIZER, _NLI_DEVICE
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    # Force CPU for deterministic, efficient batching on small batches (MPS overhead high for BART-large 400M with many small batches)
    # Original selection allowed MPS/CPU/CUDA auto; CPU is more stable for this workload and still local.
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(S3_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(S3_MODEL_ID)
    model.to(device)
    model.eval()
    _NLI_MODEL = model
    _NLI_TOKENIZER = tokenizer
    _NLI_DEVICE = device
    return model, tokenizer, device


def _split_premise(text):
    """
    Deterministic split: sentences via regex, then token-chunk long sentences to 400 tokens.
    Returns list of premise chunks (strings).
    """
    if not text or not text.strip():
        return [" "]
    # Sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    model, tokenizer, device = _load_nli()
    chunks = []
    for sent in sentences:
        ids = tokenizer(sent, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
        if len(ids) <= S3_CHUNK_TOKENS:
            chunks.append(sent)
        else:
            # token-chunk the long sentence
            for start in range(0, len(ids), S3_CHUNK_TOKENS):
                sub_ids = ids[start : start + S3_CHUNK_TOKENS]
                sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True)
                if sub_text.strip():
                    chunks.append(sub_text)
    return chunks if chunks else [" "]


def nli_scores_for_texts(texts, batch_size=13):
    """
    For each posting and each hypothesis, compute MAX entailment probability across chunks.
    Returns S shape (n,13) with values in [0,1].
    Independent per category (no cross-category softmax).

    Batching: 13 pairs per forward (one chunk × 13 hypotheses) — matches original per-chunk logic but now
    runs on CPU for stability. Batches are small to avoid padding overhead of large cross-chunk batches.
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer, device = _load_nli()
    n = len(texts)
    S = np.zeros((n, len(CATEGORIES)), dtype=float)

    entail_id = None
    try:
        entail_id = model.config.label2id.get("entailment", 2)
    except Exception:
        entail_id = 2

    for i, text in enumerate(texts):
        chunks = _split_premise(text)
        best_per_cat = np.zeros(len(CATEGORIES), dtype=float)
        for chunk in chunks:
            premises = [chunk] * len(CATEGORIES)
            hypotheses = NLI_HYPOTHESES_LIST
            enc = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits  # (13, 3)
                probs = F.softmax(logits, dim=-1)[:, entail_id].detach().cpu().numpy()
            best_per_cat = np.maximum(best_per_cat, probs)
        S[i] = best_per_cat
    return S


def get_hypotheses():
    """Return frozen hypotheses list in CATEGORIES order."""
    return list(NLI_HYPOTHESES_LIST)
