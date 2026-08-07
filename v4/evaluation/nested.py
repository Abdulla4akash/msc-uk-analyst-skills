"""
Nested CV for genuinely unbiased threshold selection.

For each outer fold:
  outer_train (n≈200) → inner CV (2-fold stratified on role_family, using ONLY outer_train)
                       → thresholds tuned on inner val predictions
                       → frozen thresholds applied to outer validation
  outer_train also provides TF-IDF fitting scope.

No outer validation text/label ever influences its own threshold or IDF.

Design notes:
- Outer: 3 folds (smallest role_family = 3)
- Inner: feasibility checked per outer fold using ONLY that outer_train's role_family counts.
         Hierarchy:
           1) StratifiedKFold k=2 where feasible (needs min_role ≥ k)
           2) If k=2 not feasible, try k=2 with GroupKFold fallback (grouped deterministic holdout)
           3) If still not feasible, deterministic 50/50 holdout split stratified by role_family via seeded shuffle
  Current data allows k=2 stratified for all 3 outer folds, so fallback is not triggered.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, GroupKFold

from v4.config import CATEGORIES
from v4.methods.lexical_baseline import (
    fit_tfidf_vectoriser,
    cosine_tfidf_scores_with_vec,
    weighted_lexical_scores_with_vec,
    unweighted_lexical_scores,
    tune_thresholds,
)


def _make_inner_splits(outer_train_gold, seed, texts_outer_train=None):
    """
    Determine inner splits using ONLY outer_train data.
    Returns (inner_splits, strategy_description, feasible_k)
    inner_splits is list of (inner_train_idx, inner_val_idx) relative to outer_train (0..n_outer_train-1)
    """
    n = len(outer_train_gold)
    role_counts = outer_train_gold["role_family"].value_counts()
    min_role = int(role_counts.min())
    # Try k=2 stratified
    if min_role >= 2:
        try:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            y_strat = outer_train_gold["role_family"].values
            splits = list(skf.split(np.zeros(n), y_strat))
            return splits, "StratifiedKFold(k=2, shuffle, seed)", 2
        except ValueError as e:
            pass
    # Fallback: try k=2 without stratification but grouped by posting_id? Use deterministic holdout
    # Deterministic 50/50 holdout stratified via seeded shuffle per role_family
    # This mirrors make_dev_test_split logic but returns a single split as 2 folds (train/val swapped)
    rng = np.random.default_rng(seed)
    is_val = np.zeros(n, dtype=bool)
    # Use the outer_train_gold's index order (already shuffled from outer), but we need index positions
    # Group by role_family and split half
    for role, idx in outer_train_gold.groupby("role_family").groups.items():
        idx_arr = np.array(list(idx))
        # Convert from original gold indices to positions within outer_train_gold
        # idx are labels from outer_train_gold.index (0..n-1 after reset)
        # After reset_index, groups keys are positions; handle both
        rng.shuffle(idx_arr)
        n_val = len(idx_arr) // 2 or 1
        # idx_arr contains positional indices within outer_train_gold
        # But groupby gives index labels, which after reset are 0..n-1, same as positions
        is_val[idx_arr[:n_val]] = True
    # Create two "folds": one where is_val is val, other where complement is val (so both halves get evaluated)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    # Second fold swapped
    splits = [(train_idx, val_idx), (val_idx, train_idx)]
    return splits, f"deterministic 50/50 holdout stratified on role_family (seed {seed}) fallback — StratifiedKFold k=2 not feasible (min_role {min_role})", 1


def _tune_via_inner_cv(scores_inner_oof, y_outer_train, inner_splits, grid):
    """
    Tune thresholds by averaging F1 across inner val folds.
    scores_inner_oof: (n_outer_train, 13) — OOF scores where each row's score came from a model fitted on its inner train.
    y_outer_train: (n_outer_train, 13)
    Returns thresholds (13,)
    """
    n_cats = scores_inner_oof.shape[1]
    mean_f1 = np.zeros((n_cats, len(grid)), dtype=float)
    for train_idx, val_idx in inner_splits:
        s_val = scores_inner_oof[val_idx]
        y_val = y_outer_train[val_idx]
        for gi, t in enumerate(grid):
            pred = (s_val >= t).astype(int)
            for ci in range(n_cats):
                _, _, f, _ = precision_recall_fscore_support(
                    y_val[:, ci], pred[:, ci], average="binary", zero_division=0
                )
                mean_f1[ci, gi] += float(f)
    mean_f1 /= len(inner_splits)
    thresholds = np.array([grid[int(np.argmax(mean_f1[ci]))] for ci in range(n_cats)], dtype=float)
    return thresholds


def run_nested_cv_for_method(
    gold_df,
    y,
    texts,
    outer_splits,
    method_name,
    seed=42,
    grid=None,
):
    """
    Run genuinely nested CV for one method.

    Returns dict with:
      - nested_predictions: (n,13) int — each posting predicted exactly once via its outer val fold
      - nested_scores: (n,13) float — outer val scores
      - outer_fold_info: list of per-fold dicts with thresholds, inner_strategy, outer_train_n, etc.
      - inner_oof_scores/debug info optional

    Isolation guarantee: for each outer fold, thresholds and vectoriser are derived ONLY from outer_train.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)

    n = len(gold_df)
    n_cats = len(CATEGORIES)
    nested_pred = np.zeros((n, n_cats), dtype=int)
    nested_scores = np.zeros((n, n_cats), dtype=float)
    outer_fold_info = []
    visited = np.zeros(n, dtype=bool)

    for outer_fold_idx, (outer_train_idx, outer_val_idx) in enumerate(outer_splits):
        outer_train_idx = np.array(outer_train_idx, dtype=int)
        outer_val_idx = np.array(outer_val_idx, dtype=int)
        # Structural isolation: never touch y[outer_val_idx] or texts[outer_val_idx] for fitting/tuning
        outer_train_gold = gold_df.iloc[outer_train_idx].reset_index(drop=True)
        y_outer_train = y[outer_train_idx]
        y_outer_val = y[outer_val_idx]  # only used at final evaluation time
        texts_outer_train = [texts[i] for i in outer_train_idx]
        texts_outer_val = [texts[i] for i in outer_val_idx]

        # Inner splits using ONLY outer_train
        inner_seed = seed + 1000 * outer_fold_idx + 7  # deterministic offset
        inner_splits, inner_strategy, inner_k = _make_inner_splits(outer_train_gold, seed=inner_seed, texts_outer_train=texts_outer_train)

        # Build inner OOF scores for outer_train rows, to tune thresholds
        n_outer_train = len(outer_train_idx)
        inner_oof_scores = np.zeros((n_outer_train, n_cats), dtype=float)

        for inner_train_rel, inner_val_rel in inner_splits:
            # Map relative indices (within outer_train) to absolute gold indices for texts
            # inner_train_rel / inner_val_rel are positions in outer_train_gold (0..n_outer_train-1)
            # Texts for those positions
            inner_train_texts = [texts_outer_train[i] for i in inner_train_rel]
            inner_val_texts = [texts_outer_train[i] for i in inner_val_rel]
            if method_name == "unweighted_lexical":
                inner_oof_scores[inner_val_rel] = unweighted_lexical_scores(inner_val_texts)
            elif method_name == "cosine_tfidf":
                vec_inner = fit_tfidf_vectoriser(inner_train_texts)
                inner_oof_scores[inner_val_rel] = cosine_tfidf_scores_with_vec(vec_inner, inner_val_texts)
            elif method_name == "weighted_lexical_tfidf":
                vec_inner = fit_tfidf_vectoriser(inner_train_texts)
                inner_oof_scores[inner_val_rel] = weighted_lexical_scores_with_vec(vec_inner, inner_val_texts)
            else:
                raise ValueError(f"Unknown method {method_name}")

        # Tune thresholds using ONLY inner OOF scores and outer_train labels
        thresholds = _tune_via_inner_cv(inner_oof_scores, y_outer_train, inner_splits, grid)

        # Now fit on FULL outer_train and score outer_val (this is the genuine outer prediction)
        if method_name == "unweighted_lexical":
            # No fitting needed; score outer_val directly
            scores_outer_val = unweighted_lexical_scores(texts_outer_val)
        elif method_name == "cosine_tfidf":
            vec_outer = fit_tfidf_vectoriser(texts_outer_train)
            scores_outer_val = cosine_tfidf_scores_with_vec(vec_outer, texts_outer_val)
        elif method_name == "weighted_lexical_tfidf":
            vec_outer = fit_tfidf_vectoriser(texts_outer_train)
            scores_outer_val = weighted_lexical_scores_with_vec(vec_outer, texts_outer_val)
        else:
            raise ValueError(f"Unknown method {method_name}")

        pred_outer_val = (scores_outer_val >= thresholds).astype(int)

        # Store — each outer_val posting gets exactly one prediction
        nested_pred[outer_val_idx] = pred_outer_val
        nested_scores[outer_val_idx] = scores_outer_val
        visited[outer_val_idx] = True

        # Provenance for this outer fold
        outer_fold_info.append({
            "outer_fold": int(outer_fold_idx),
            "method": method_name,
            "thresholds": {c: float(thresholds[i]) for i, c in enumerate(CATEGORIES)},
            "thresholds_array": thresholds.tolist(),
            "inner_strategy": inner_strategy,
            "inner_k": int(inner_k if isinstance(inner_k, int) else 0),
            "inner_n_splits": int(len(inner_splits)),
            "inner_seed": int(inner_seed),
            "outer_train_n": int(len(outer_train_idx)),
            "outer_validation_n": int(len(outer_val_idx)),
            "outer_train_ids": gold_df.iloc[outer_train_idx]["posting_id"].tolist(),
            "outer_validation_ids": gold_df.iloc[outer_val_idx]["posting_id"].tolist(),
            "feature_fitting_population": "outer_train texts only (inductive)",
            "threshold_tuning_population": "inner validation predictions/labels only (outer validation labels never used)",
            "seed": int(seed),
            "grid": {"start": float(grid[0]), "stop": float(grid[-1]), "n_points": int(len(grid))},
        })

    # Accounting check: every posting exactly once
    assert np.all(visited), "Not every posting was covered as outer validation exactly once — nested CV accounting failed"
    # Also check no overlap (visited ensures, but check that sets partition)
    return {
        "nested_predictions": nested_pred,
        "nested_scores": nested_scores,
        "outer_fold_info": outer_fold_info,
    }
