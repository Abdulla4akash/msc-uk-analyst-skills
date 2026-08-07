"""
Selective semantic fallback — asymmetric, with OFF option.

A1 (whole-word any-hit) is the gate: if A1 predicts 1 for (posting, category),
final гибрид = 1 regardless of semantic score.  Semantic fallback only applies
when A1 = 0 and semantic score >= per-category threshold.

OFF (represented as 2.0, since scores in [0,1]) means no fallback for that
category: hybrid = A1 always.

Thresholds are tuned on FULL HYBRID F1 (lexical OR semantic), not semantic-only.
Conservative tie-break: when multiple thresholds tie on F1, the highest
(most conservative, OFF=2.0 beats any numeric) wins.

Batch-invariant: no cross-posting normalisation.

Reuses frozen definitions:
  - A1: v4/ablation/lexical_ablation score_for_variant("A1", texts) -> any_hit
  - S1: v4/semantic/supervised_tfidf vectoriser+LR (C in {0.1,1,10})
  - S3: v4/semantic/zero_shot_nli NLI MAX entailment (typeform/distilbert-base-uncased-mnli)
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from v4.config import CATEGORIES

# Grid: 51 points 0..1 inclusive (step 0.02) + OFF
GRID_51 = np.linspace(0.0, 1.0, 51)
OFF_THRESHOLD = 2.0  # >1 so semantic_scores < OFF always
GRID_WITH_OFF = np.concatenate([GRID_51, np.array([OFF_THRESHOLD])])
# For conservative tie-break we iterate descending (OFF first)
GRID_DESC = np.sort(GRID_WITH_OFF)[::-1]


def lexical_A1_predictions(texts):
    """Frozen A1 gate: whole-word any-hit binary predictions shape (n,13)."""
    from v4.ablation.lexical_ablation import score_for_variant, any_hit_predictions

    S = score_for_variant("A1", texts)
    return any_hit_predictions(S)


def lexical_A1_scores(texts):
    """Continuous A1 scores (for completeness) — same as variant A1 scores."""
    from v4.ablation.lexical_ablation import score_for_variant

    return score_for_variant("A1", texts)


def apply_hybrid_thresholds(lexical_binary, semantic_scores, thresholds):
    """
    Apply asymmetric fallback.

    lexical_binary: (n,13) int 0/1 from A1
    semantic_scores: (n,13) float in [0,1]
    thresholds: (13,) float, OFF=2.0 means no fallback
    Returns hybrid binary (n,13) int
    """
    thresholds = np.asarray(thresholds, dtype=float)
    n_cats = lexical_binary.shape[1]
    if thresholds.size != n_cats or semantic_scores.shape[1] != n_cats:
        raise ValueError(f"shape mismatch: lexical {lexical_binary.shape}, semantic {semantic_scores.shape}, thresholds {thresholds.size}")
    n = lexical_binary.shape[0]
    hybrid = np.zeros((n, n_cats), dtype=int)
    for ci, thr in enumerate(thresholds):
        lex = lexical_binary[:, ci].astype(int)
        if thr >= 1.5:  # OFF (covers 2.0 and inf)
            hybrid[:, ci] = lex
        else:
            sem = (semantic_scores[:, ci] >= thr).astype(int)
            # asymmetric: lexical 1 cannot be vetoed
            hybrid[:, ci] = np.maximum(lex, sem)
            # equivalent to lex OR sem when lex=0, but ensures lex=1 stays 1 even if sem=0
            # Also ensure when lex=1, sem low does not veto: already via maximum
    return hybrid


def tune_hybrid_thresholds(lexical_binary, semantic_scores, y_true, grid=None):
    """
    Per-category threshold search that maximises per-category F1 of the hybrid.
    Since macro-F1 is mean of per-category F1, this maximises macro.

    lexical_binary: (n,13) int
    semantic_scores: (n,13) float
    y_true: (n,13) int
    grid: array of thresholds to try, must include OFF. If None uses GRID_WITH_OFF.
    Returns thresholds (13,) float, conservative tie-break (highest wins on tie).

    Leak-safe when called with only outer_train / inner data.
    """
    if grid is None:
        grid = GRID_WITH_OFF
    grid = np.asarray(grid, dtype=float)
    # Ensure descending for tie-break
    grid_desc = np.sort(grid)[::-1]
    n_cats = lexical_binary.shape[1]
    thresholds = np.zeros(n_cats, dtype=float)
    for ci in range(n_cats):
        best_f = -1.0
        best_t = OFF_THRESHOLD
        y_col = y_true[:, ci]
        lex_col = lexical_binary[:, ci]
        sem_col = semantic_scores[:, ci]
        for thr in grid_desc:
            if thr >= 1.5:
                pred = lex_col
            else:
                pred = np.maximum(lex_col, (sem_col >= thr).astype(int))
            _, _, f, _ = precision_recall_fscore_support(
                y_col, pred, average="binary", zero_division=0
            )
            f = float(f)
            if f > best_f:  # strictly greater -> conservative (first max in descending order wins)
                best_f = f
                best_t = float(thr)
        thresholds[ci] = best_t
    return thresholds


def tune_hybrid_thresholds_inner_cv(lexical_outer_train, semantic_outer_train, y_outer_train, inner_splits, grid=None):
    """
    Genuinely nested tuning via inner 2-fold CV over outer_train.

    For each category and each threshold candidate, compute mean F1 across
    inner validation folds where hybrid = lexical_val OR (semantic_val >= thr).
    Pick threshold with highest mean F1, conservative tie-break (highest).

    This mirrors semantic_nested S2/S3 inner logic but hybrid-aware.

    Returns thresholds (13,) float.
    """
    if grid is None:
        grid = GRID_WITH_OFF
    grid = np.asarray(grid, dtype=float)
    grid_desc = np.sort(grid)[::-1]
    n_cats = lexical_outer_train.shape[1]
    thresholds = np.zeros(n_cats, dtype=float)
    for ci in range(n_cats):
        best_mean = -1.0
        best_t = OFF_THRESHOLD
        for thr in grid_desc:
            f_vals = []
            for train_rel, val_rel in inner_splits:
                y_val = y_outer_train[val_rel, ci]
                lex_val = lexical_outer_train[val_rel, ci]
                sem_val = semantic_outer_train[val_rel, ci]
                if thr >= 1.5:
                    pred = lex_val
                else:
                    pred = np.maximum(lex_val, (sem_val >= thr).astype(int))
                _, _, f, _ = precision_recall_fscore_support(
                    y_val, pred, average="binary", zero_division=0
                )
                f_vals.append(float(f))
            mean_f = float(np.mean(f_vals)) if f_vals else 0.0
            if mean_f > best_mean:
                best_mean = mean_f
                best_t = float(thr)
        thresholds[ci] = best_t
    return thresholds


# Convenience for threshold type handling (dict vs ndarray) in tests
def _thr_value(thr, cat):
    if isinstance(thr, dict):
        return thr[cat]
    else:
        return thr[CATEGORIES.index(cat)]
