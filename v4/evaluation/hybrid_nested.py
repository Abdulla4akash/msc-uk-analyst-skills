"""
Genuinely nested evaluation for hybrid selective fallback H1/H2.

- H1 = A1 (frozen whole-word any-hit) + S1 (supervised TF-IDF LR) with asymmetric fallback
- H2 = A1 + S3 (frozen NLI MAX) with asymmetric fallback

Identical outer splits (seed 42, 3-fold stratified role_family, dedup) to lexical
ablation and semantic baselines.  Inner 2-fold CV per outer_train for threshold
(and for H1 also C) selection, optimising FULL HYBRID F1, with OFF + conservative
tie-break.  Outer validation never influences fitting/thresholds.

S3 NLI scores are frozen and cached with provenance hash (model_id, hypotheses,
chunking, texts hash) to avoid 650s recomputation while keeping provenance.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from v4.config import CATEGORIES
from v4.evaluation.metrics import evaluate
from v4.hybrid.selective_fallback import (
    GRID_WITH_OFF,
    OFF_THRESHOLD,
    apply_hybrid_thresholds,
    lexical_A1_predictions,
    tune_hybrid_thresholds,
    tune_hybrid_thresholds_inner_cv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _inner_splits_for_outer(gold_df, train_idx, outer_idx, seed):
    roles_outer = gold_df.iloc[train_idx]["role_family"].values
    try:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
        splits = list(skf.split(np.zeros(len(train_idx)), roles_outer))
    except ValueError:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
        splits = list(kf.split(np.zeros(len(train_idx))))
    return splits


def _provenance_hash(texts, model_id, hypotheses, chunk_tokens):
    h = hashlib.sha256()
    h.update(model_id.encode())
    for hyp in hypotheses:
        h.update(hyp.encode())
    h.update(str(chunk_tokens).encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _get_nli_scores_cached(texts, cache_dir=None):
    """
    Frozen NLI scores with provenance hash cache.
    Returns (scores (n,13), provenance dict, was_cached bool)
    Cache location: REPO_ROOT/v4/results/semantic/s3_nli_scores_cache.npz
    """
    from v4.semantic.model_config import S3_MODEL_ID, S3_CHUNK_TOKENS, NLI_HYPOTHESES_LIST
    from v4.semantic.zero_shot_nli import nli_scores_for_texts

    if cache_dir is None:
        cache_dir = REPO_ROOT / "v4" / "results" / "semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "s3_nli_scores_cache.npz"
    provenance_path = cache_dir / "s3_nli_scores_provenance.json"

    cur_hash = _provenance_hash(texts, S3_MODEL_ID, NLI_HYPOTHESES_LIST, S3_CHUNK_TOKENS)
    provenance = {
        "model_id": S3_MODEL_ID,
        "chunk_tokens": S3_CHUNK_TOKENS,
        "n_texts": len(texts),
        "texts_hash": cur_hash,
        "hypotheses": NLI_HYPOTHESES_LIST,
    }
    if cache_path.exists() and provenance_path.exists():
        try:
            prov = json.load(open(provenance_path))
            if prov.get("texts_hash") == cur_hash and prov.get("model_id") == S3_MODEL_ID:
                data = np.load(cache_path)
                scores = data["scores"]
                if scores.shape == (len(texts), len(CATEGORIES)):
                    return scores, provenance, True
        except Exception:
            pass
    # Compute fresh
    scores = nli_scores_for_texts(texts)
    np.savez_compressed(cache_path, scores=scores)
    json.dump(provenance, open(provenance_path, "w"), indent=2)
    return scores, provenance, False


def run_nested_hybrid_S1(gold_df, y, texts, outer_splits, seed=42):
    """
    H1: A1 + S1 (TF-IDF LR) asymmetric fallback.
    Tunes C (0.1,1,10) and per-category thresholds via inner CV optimising hybrid F1.
    Returns dict with nested_predictions, nested_scores (semantic), thresholds, etc.
    """
    from v4.semantic.supervised_tfidf import build_vectoriser
    from v4.semantic.model_config import S1_C_GRID, S1_CLASS_WEIGHT
    from sklearn.linear_model import LogisticRegression

    n = len(gold_df)
    outer_pred = np.zeros((n, len(CATEGORIES)), dtype=int)
    outer_scores_sem = np.zeros((n, len(CATEGORIES)), dtype=float)
    outer_info = []

    for outer_idx, (train_idx, val_idx) in enumerate(outer_splits):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_train = y[train_idx]
        # Lexical gate for outer_train/val
        lex_train = lexical_A1_predictions(train_texts)
        lex_val = lexical_A1_predictions(val_texts)

        # Inner splits for this outer fold
        inner_splits = _inner_splits_for_outer(gold_df, train_idx, outer_idx, seed)

        # Select C + thresholds via inner CV hybrid-aware
        best_C = None
        best_thr = None
        best_inner_macro = -1
        best_details = None

        for C in S1_C_GRID:
            # Compute inner OOF semantic scores for outer_train for this C
            # Use helper from supervised_tfidf to get OOF per inner split
            # We replicate _inner_oof_scores logic locally to avoid import cycle
            n_outer_train = len(train_texts)
            oof_sem = np.zeros((n_outer_train, len(CATEGORIES)), dtype=float)
            for inner_train_rel, inner_val_rel in inner_splits:
                vec = build_vectoriser()
                X_tr = vec.fit_transform([train_texts[i] for i in inner_train_rel])
                X_val = vec.transform([train_texts[i] for i in inner_val_rel])
                for ci in range(len(CATEGORIES)):
                    clf = LogisticRegression(
                        C=C, class_weight=S1_CLASS_WEIGHT, solver="lbfgs", max_iter=1000, random_state=42
                    )
                    clf.fit(X_tr, y_train[inner_train_rel, ci])
                    prob = clf.predict_proba(X_val)[:, 1]
                    oof_sem[inner_val_rel, ci] = prob
            # Tune hybrid thresholds on OOF (lex_train + oof_sem) optimising hybrid F1
            # Use per-category tuning on OOF directly (since OOF already inner-CV, tuning on OOF is valid and equivalent to inner CV averaging)
            thr_candidate = tune_hybrid_thresholds(lex_train, oof_sem, y_train, grid=GRID_WITH_OFF)
            hybrid_oof = apply_hybrid_thresholds(lex_train, oof_sem, thr_candidate)
            rep = evaluate(y_train, hybrid_oof)
            macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
            if macro > best_inner_macro:  # strictly greater, conservative: first wins on tie
                best_inner_macro = macro
                best_C = C
                best_thr = thr_candidate
                best_details = {"C": C, "macro": macro}

        # Fit on full outer_train with best_C, score outer_val
        vec = build_vectoriser()
        X_train = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)
        prob_val = np.zeros((len(val_texts), len(CATEGORIES)), dtype=float)
        for ci in range(len(CATEGORIES)):
            clf = LogisticRegression(
                C=best_C, class_weight=S1_CLASS_WEIGHT, solver="lbfgs", max_iter=1000, random_state=42
            )
            clf.fit(X_train, y_train[:, ci])
            prob_val[:, ci] = clf.predict_proba(X_val)[:, 1]

        outer_scores_sem[val_idx] = prob_val
        outer_pred[val_idx] = apply_hybrid_thresholds(lex_val, prob_val, best_thr)
        outer_info.append(
            {
                "outer_fold": outer_idx,
                "best_C": best_C,
                "thresholds": best_thr,
                "best_inner_macro": float(best_inner_macro),
                "n_outer_train": len(train_idx),
                "n_outer_val": len(val_idx),
                "inner_details": best_details,
            }
        )

    return {
        "nested_predictions": outer_pred,
        "nested_scores_semantic": outer_scores_sem,
        "outer_fold_info": outer_info,
    }


def run_nested_hybrid_S3(gold_df, y, texts, outer_splits, seed=42, cache_dir=None):
    """
    H2: A1 + S3 (frozen NLI) asymmetric fallback.
    Only thresholds are tuned (NLI frozen), via inner CV hybrid F1, with OFF.
    Uses cached NLI scores with provenance hash.
    """
    n = len(gold_df)
    outer_pred = np.zeros((n, len(CATEGORIES)), dtype=int)
    outer_scores_sem = np.zeros((n, len(CATEGORIES)), dtype=float)
    outer_info = []

    # Precompute all NLI scores once (with cache)
    all_sem_scores, provenance, was_cached = _get_nli_scores_cached(texts, cache_dir=cache_dir)

    for outer_idx, (train_idx, val_idx) in enumerate(outer_splits):
        # Lexical for outer_train/val
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        lex_train = lexical_A1_predictions(train_texts)
        lex_val = lexical_A1_predictions(val_texts)
        sem_train = all_sem_scores[train_idx]
        sem_val = all_sem_scores[val_idx]

        inner_splits = _inner_splits_for_outer(gold_df, train_idx, outer_idx, seed)
        # Tune thresholds via inner CV hybrid
        thr = tune_hybrid_thresholds_inner_cv(lex_train, sem_train, y[train_idx], inner_splits, grid=GRID_WITH_OFF)
        # Also evaluate inner macro for reporting
        # Compute inner OOF hybrid macro with chosen thr (for best_inner_macro)
        # Average across inner folds
        inner_macros = []
        for tr_rel, va_rel in inner_splits:
            hybrid_va = apply_hybrid_thresholds(
                lex_train[va_rel], sem_train[va_rel], thr
            )
            rep = evaluate(y[train_idx][va_rel], hybrid_va)
            inner_macros.append(float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0]))
        best_inner_macro = float(np.mean(inner_macros)) if inner_macros else 0.0

        outer_scores_sem[val_idx] = sem_val
        outer_pred[val_idx] = apply_hybrid_thresholds(lex_val, sem_val, thr)
        outer_info.append(
            {
                "outer_fold": outer_idx,
                "thresholds": thr,
                "best_inner_macro": best_inner_macro,
                "n_outer_train": len(train_idx),
                "n_outer_val": len(val_idx),
                "nli_provenance": provenance,
                "nli_was_cached": was_cached,
            }
        )

    return {
        "nested_predictions": outer_pred,
        "nested_scores_semantic": outer_scores_sem,
        "outer_fold_info": outer_info,
        "nli_provenance": provenance,
        "nli_was_cached": was_cached,
    }
