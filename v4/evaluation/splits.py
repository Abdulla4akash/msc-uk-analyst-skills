"""
Split utilities for v4.

- make_dev_test_split(): reproducible stratified split on role_family,
  used only to define the internal held-out evaluation partition.
  Thresholds and model selection must NOT use the test partition.

- make_cv_splits(): grouped/repeated cross-validation over the DEV set only.
  This is the defensible development-only evaluation strategy.

Design choices
--------------
- Stratification key is role_family (human-confirmed).
- Deduplication: exact job_summary SHA-256 is computed; if duplicates exist,
  they are assigned to the same fold via group labels. Normalised-hash and
  TF-IDF near-duplicate detection are also checked and documented but not
  forced to merge unless evidence supports it.

- Rare categories (e.g. ethics_governance n=13) make per-category stratified
  CV statistically fragile. We therefore stratify splits on role_family only,
  and document that per-category performance for rare labels has high variance.
  Threshold tuning is still per-category but evaluated with CV averaging.

- For v4 the 300 postings are treated as DEVELOPMENT DATA for future work.
  The current dev/test terminology in this module refers to an internal
  evaluation partition of that development corpus, not to an external locked
  test set (which does not exist yet).
"""

import hashlib
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


RANDOM_SEED = 42


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalised_text_hash(text):
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w ]", "", t)
    t = t.strip()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def compute_duplicate_groups(texts, posting_ids=None):
    """
    Compute group ids for deduplication.

    Returns dict with:
      - exact_hash: list of hashes
      - normalised_hash: list
      - group_ids: ndarray of ints where duplicates share an id (exact match grouping)
      - n_unique_exact, n_unique_normalised
      - duplicate_summary: DataFrame listing any exact duplicates
    """
    exact = [text_hash(t) for t in texts]
    normed = [normalised_text_hash(t) for t in texts]
    # map exact hash -> group id
    hash_to_group = {}
    groups = []
    next_gid = 0
    for h in exact:
        if h not in hash_to_group:
            hash_to_group[h] = next_gid
            next_gid += 1
        groups.append(hash_to_group[h])
    groups = np.array(groups, dtype=int)
    # near-duplicate summary (exact only for now)
    counter = Counter(exact)
    dup_hashes = {h for h, c in counter.items() if c > 1}
    summary = []
    if posting_ids is not None:
        for pid, h, t in zip(posting_ids, exact, texts):
            if h in dup_hashes:
                summary.append({"posting_id": pid, "hash": h})
    summary_df = pd.DataFrame(summary)
    return {
        "exact_hash": exact,
        "normalised_hash": normed,
        "group_ids": groups,
        "n_unique_exact": len(set(exact)),
        "n_unique_normalised": len(set(normed)),
        "n_exact_duplicate_groups": len(dup_hashes),
        "duplicate_summary": summary_df,
    }


def _grouped_stratified_indices(gold_df, group_ids, n_splits, seed):
    """
    Approximate stratified GroupKFold.

    sklearn has no StratifiedGroupKFold in older versions, so we implement a
    simple greedy assignment that respects groups and tries to balance role_family.

    For the current corpus (no exact duplicates in gold), groups are unique,
    so this reduces to StratifiedKFold. We keep the grouping logic for
    defensibility if duplicates appear.
    """
    # If all groups unique, delegate to StratifiedKFold
    if len(set(group_ids)) == len(group_ids):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_strat = gold_df["role_family"].values
        return list(skf.split(np.zeros(len(gold_df)), y_strat))
    # Otherwise greedy group assignment
    # This is a best-effort; we document limitations.
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_strat = gold_df["role_family"].values
        return list(sgkf.split(np.zeros(len(gold_df)), y_strat, groups=group_ids))
    except ImportError:
        # Fallback: GroupKFold (loses stratification, document it)
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.zeros(len(gold_df)), groups=group_ids))


def make_dev_test_split(gold_df, dev_frac=1 / 3, seed=RANDOM_SEED):
    """
    Reproducible stratified internal split on role_family.

    The 300-posting gold standard is treated as DEVELOPMENT material.
    This function partitions it into:
      - internal_tuning (∼100, historically "dev")
      - internal_holdout (∼200, historically "test")
    The historical "test" partition is NOT an external locked test set.
    The external_locked_test (future) will be independently collected.

    Mirrors v3/evaluate.py:make_split but returns explicit masks and ids
    with publication-safe terminology.
    """
    rng = np.random.default_rng(seed)
    is_internal_tuning = np.zeros(len(gold_df), dtype=bool)
    for _, idx in gold_df.groupby("role_family").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_dev = max(1, int(round(len(idx) * dev_frac)))
        is_internal_tuning[idx[:n_dev]] = True
    is_internal_holdout = ~is_internal_tuning
    return {
        "is_internal_tuning": is_internal_tuning,
        "is_internal_holdout": is_internal_holdout,
        "internal_tuning_ids": gold_df.loc[is_internal_tuning, "posting_id"].tolist(),
        "internal_holdout_ids": gold_df.loc[is_internal_holdout, "posting_id"].tolist(),
        "seed": seed,
        "dev_frac": dev_frac,
        "stratify_on": "role_family",
        "n_internal_tuning": int(is_internal_tuning.sum()),
        "n_internal_holdout": int(is_internal_holdout.sum()),
    }


def make_internal_holdout_split(*args, **kwargs):
    """Alias for make_dev_test_split with publication-safe name."""
    return make_dev_test_split(*args, **kwargs)


def make_cv_splits(gold_df, texts=None, n_splits=5, n_repeats=1, seed=RANDOM_SEED):
    if n_repeats != 1:
        raise ValueError(
            "Nested prediction aggregation currently supports one outer-CV repetition only "
            "(n_repeats must be 1). Repeated nested CV would require aggregating multiple "
            "predictions per posting; not implemented."
        )
    """
    Build CV folds over the *development* portion (or over all 300 if caller
    treats 300 as dev).  Each split is (train_idx, val_idx) with grouping.

    Parameters
    ----------
    gold_df : DataFrame
        Rows to split (typically the 300 gold rows, or dev subset).
    texts : list[str] | None
        If provided, used to compute duplicate groups.
    n_splits : int
        Number of folds (5 preferred; falls back if per-fold counts fragile).
    n_repeats : int
        Repeats with different seeds if >1.
    seed : int

    Returns
    -------
    splits : list of (train_idx, val_idx)
    meta : dict with grouping info and limitations
    """
    n = len(gold_df)
    # Check feasibility: smallest role_family group determines max n_splits
    role_counts = gold_df["role_family"].value_counts()
    min_role_n = int(role_counts.min())
    # Smallest category support also limits reliability
    # We document rather than block; caller decides.
    limitations = []
    if n_splits > min_role_n:
        limitations.append(
            f"n_splits={n_splits} exceeds smallest role_family count {min_role_n}; "
            f"StratifiedKFold would fail. Reducing to {min_role_n} folds."
        )
        n_splits = min_role_n
    if n_splits < 2:
        n_splits = 2
        limitations.append("Forced n_splits=2 due to tiny group.")

    # Deduplication groups
    if texts is not None:
        dup = compute_duplicate_groups(texts, posting_ids=gold_df["posting_id"].tolist())
        group_ids = dup["group_ids"]
        if dup["n_exact_duplicate_groups"] == 0:
            limitations.append("No exact duplicates detected; grouping has no effect (splits == StratifiedKFold).")
        if dup["n_unique_exact"] != dup["n_unique_normalised"]:
            limitations.append(
                f"Normalised-hash uniques ({dup['n_unique_normalised']}) differ from exact ({dup['n_unique_exact']}); "
                "consider normalised grouping in future if evidence of near-duplicates."
            )
    else:
        group_ids = np.arange(n)
        limitations.append("No texts provided; grouping disabled (assumes no duplicates).")

    # Also check TF-IDF near-duplicate heuristic if texts available
    if texts is not None and len(texts) >= 10:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            X = vec.fit_transform(texts)
            sim = cosine_similarity(X)
            # count pairs > 0.90 excluding diagonal
            np.fill_diagonal(sim, 0)
            n_close = int((sim > 0.90).sum() // 2)
            if n_close > 0:
                limitations.append(f"TF-IDF near-duplicate check: {n_close} pairs with cosine > 0.90 (check manually).")
            else:
                limitations.append("TF-IDF near-duplicate check: no pairs > 0.90 (reassuring).")
        except Exception as e:
            limitations.append(f"TF-IDF near-duplicate check skipped: {e}")

    splits = []
    for rep in range(n_repeats):
        rep_seed = seed + rep * 1000
        fold_list = _grouped_stratified_indices(gold_df, group_ids, n_splits, rep_seed)
        splits.extend(fold_list)

    meta = {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_total_folds": len(splits),
        "seed": seed,
        "group_dedup_applied": texts is not None,
        "limitations": limitations,
        "role_family_counts": role_counts.to_dict(),
        "group_ids": group_ids.tolist() if isinstance(group_ids, np.ndarray) else list(group_ids),
    }
    return splits, meta
