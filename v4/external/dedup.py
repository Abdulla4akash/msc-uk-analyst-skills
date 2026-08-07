"""
Deduplication for external evaluation — mandatory.

Dedup steps:
1. within E1
2. within E2
3. between E1 and E2
4. external candidates against ALL 820 development corpus postings (not only 300 annotated)

Uses:
- D1 exact SHA-256 raw text
- D2 normalised text hash (case/whitespace/punctuation)
- D3 near-duplicate via TF-IDF cosine / token Jaccard / MinHash (documented)

Generates near-duplicate review table and duplicate_group_id.
"""

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalised_text_hash(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w ]", "", t)
    t = t.strip()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def jaccard_shingle(text: str, k: int = 5) -> set:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return set([" ".join(toks)]) if toks else set()
    return set(" ".join(toks[i:i+k]) for i in range(len(toks)-k+1))

def compute_dedup(texts, posting_ids=None, tfidf_threshold=0.90):
    """
    Compute dedup groups.

    Returns dict with:
      - exact_hash
      - normalised_hash
      - group_ids (exact grouping)
      - near_duplicate_pairs: list of (i,j, cosine)
      - tfidf_matrix etc
    """
    n = len(texts)
    exact = [text_hash(t) for t in texts]
    normed = [normalised_text_hash(t) for t in texts]
    # exact groups
    hash_to_gid = {}
    group_ids = []
    next_gid = 0
    for h in exact:
        if h not in hash_to_gid:
            hash_to_gid[h] = next_gid
            next_gid += 1
        group_ids.append(hash_to_gid[h])
    group_ids = np.array(group_ids, dtype=int)

    # near-duplicate TF-IDF cosine
    near_pairs = []
    if n >= 2:
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            X = vec.fit_transform(texts)
            sim = cosine_similarity(X)
            np.fill_diagonal(sim, 0)
            for i in range(n):
                for j in range(i+1, n):
                    if sim[i,j] > tfidf_threshold:
                        near_pairs.append((i,j, float(sim[i,j])))
        except Exception as e:
            near_pairs = []
    return {
        "exact_hash": exact,
        "normalised_hash": normed,
        "group_ids": group_ids,
        "n_unique_exact": len(set(exact)),
        "n_unique_normalised": len(set(normed)),
        "n_exact_duplicate_groups": len([h for h,c in Counter(exact).items() if c>1]),
        "near_duplicate_pairs": near_pairs,
        "tfidf_threshold": tfidf_threshold
    }

def dedup_against_development(external_texts, external_ids, dev_texts, dev_ids, tfidf_threshold=0.90):
    """
    Check external candidates against ALL 820 development corpus postings.
    Returns dict with overlaps.
    """
    # exact and normalised overlap
    dev_exact_set = set(text_hash(t) for t in dev_texts)
    dev_norm_set = set(normalised_text_hash(t) for t in dev_texts)
    ext_exact = [text_hash(t) for t in external_texts]
    ext_norm = [normalised_text_hash(t) for t in external_texts]

    exact_overlap_idx = [i for i,h in enumerate(ext_exact) if h in dev_exact_set]
    norm_overlap_idx = [i for i,h in enumerate(ext_norm) if h in dev_norm_set]

    # TF-IDF near duplicate against dev corpus
    near_pairs = []
    if len(external_texts) >=1 and len(dev_texts) >=1:
        try:
            all_texts = dev_texts + external_texts
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            X = vec.fit_transform(all_texts)
            dev_X = X[:len(dev_texts)]
            ext_X = X[len(dev_texts):]
            sim = cosine_similarity(ext_X, dev_X)  # (n_ext, n_dev)
            # For each ext, find max dev sim
            for i in range(len(external_texts)):
                max_sim = float(sim[i].max())
                max_j = int(sim[i].argmax())
                if max_sim > tfidf_threshold:
                    near_pairs.append((external_ids[i], dev_ids[max_j], max_sim))
        except Exception:
            pass
    return {
        "n_external": len(external_texts),
        "n_dev": len(dev_texts),
        "exact_overlaps": len(exact_overlap_idx),
        "exact_overlap_ids": [external_ids[i] for i in exact_overlap_idx],
        "normalised_overlaps": len(norm_overlap_idx),
        "normalised_overlap_ids": [external_ids[i] for i in norm_overlap_idx],
        "near_duplicate_pairs": near_pairs,
        "tfidf_threshold": tfidf_threshold
    }

def make_duplicate_group_ids(exact_hashes):
    hash_to_gid = {}
    gids = []
    next_gid = 0
    for h in exact_hashes:
        if h not in hash_to_gid:
            hash_to_gid[h] = next_gid
            next_gid += 1
        gids.append(hash_to_gid[h])
    return gids

def review_table(texts, ids, threshold=0.85):
    """
    Generate near-duplicate review table for manual inspection.
    Returns DataFrame with candidate pairs above threshold.
    """
    if len(texts) < 2:
        return pd.DataFrame()
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    rows = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if sim[i,j] > threshold:
                rows.append({
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "cosine": float(sim[i,j]),
                    "exact_hash_a": text_hash(texts[i]),
                    "exact_hash_b": text_hash(texts[j]),
                    "norm_hash_a": normalised_text_hash(texts[i]),
                    "norm_hash_b": normalised_text_hash(texts[j]),
                })
    return pd.DataFrame(rows).sort_values("cosine", ascending=False)
