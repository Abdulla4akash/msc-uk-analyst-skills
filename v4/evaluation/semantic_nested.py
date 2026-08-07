"""
Generic nested evaluation helpers for semantic baselines — reuse split/metrics/bootstrap
where possible, but keep lexical nested.py untouched (existing tests rely on it).

Each helper enforces: outer validation never influences fitting/thresholds/model selection.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

from v4.config import CATEGORIES
from v4.methods.lexical_baseline import tune_thresholds, apply_thresholds
from v4.evaluation.metrics import evaluate

from v4.semantic.supervised_tfidf import (
    select_hyperparameters_inner_cv,
    get_outer_scores_and_thresholds,
)
from v4.semantic.embedding_similarity import embedding_scores, get_category_embeddings
from v4.semantic.zero_shot_nli import nli_scores_for_texts

# Cache helpers for NLI (reuse proven cache from hybrid, environment/model-cache problem fix)
REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()

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

def _get_nli_scores_cached(texts):
    from v4.semantic.model_config import S3_MODEL_ID, S3_CHUNK_TOKENS, NLI_HYPOTHESES_LIST
    from v4.semantic.zero_shot_nli import nli_scores_for_texts
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
                if scores.shape == (len(texts), len(__import__("v4.config", fromlist=["CATEGORIES"]).CATEGORIES)):
                    return scores, provenance, True
        except Exception:
            pass
    scores = nli_scores_for_texts(texts)
    np.savez_compressed(cache_path, scores=scores)
    json.dump(provenance, open(provenance_path, "w"), indent=2)
    return scores, provenance, False




def _make_inner_splits(y_outer_train, outer_train_idx, seed):
    """
    Two inner folds over outer_train indices (stratified on role_family via caller?).
    For simplicity here we stratify on first label column or use plain KFold if rare?
    Use outer_train role_family via the same logic as make_cv_splits: we need role_family.
    Caller will pass gold_df; this helper will be called with gold_df for proper stratification.
    """
    raise NotImplementedError


def run_nested_supervised_tfidf(gold_df, y, texts, outer_splits, seed=42):
    """
    Returns dict with nested_predictions, nested_scores, per_outer info.
    Mirrors lexical nested but with supervised hyperparameter selection.
    """
    n = len(gold_df)
    outer_pred = np.zeros((n, len(CATEGORIES)), dtype=int)
    outer_scores = np.zeros((n, len(CATEGORIES)), dtype=float)
    outer_info = []
    # For prediction_count integrity: ensure each posting predicted exactly once (outer val)
    for outer_idx, (train_idx, val_idx) in enumerate(outer_splits):
        outer_train_texts = [texts[i] for i in train_idx]
        y_outer_train = y[train_idx]
        outer_val_texts = [texts[i] for i in val_idx]
        # Build inner splits over outer_train (2-fold stratified on role_family)
        roles_outer = gold_df.iloc[train_idx]["role_family"].values
        # Deduplicate groups not needed here (inner dedup would require groups); use simple StratifiedKFold on roles
        # If too few per class, fall back to KFold
        try:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
            inner_splits_idx = list(skf.split(np.zeros(len(train_idx)), roles_outer))
        except ValueError:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
            inner_splits_idx = list(kf.split(np.zeros(len(train_idx))))
        # Convert inner indices from 0..len(train_idx)-1 to positions in the outer_train arrays (for texts/y_outer_train)
        # Our helper expects texts/y as outer_train-sized; so keep as is
        # But select_hyperparameters_inner_cv expects full arrays and splits over them
        # We'll call with outer_train arrays directly
        # Need inner_splits as list of (inner_train_idx, inner_val_idx) over 0..len(outer_train)-1
        inner_splits = inner_splits_idx
        best_C, best_thr, best_macro, details = select_hyperparameters_inner_cv(
            outer_train_texts, y_outer_train, inner_splits
        )
        # Fit on full outer_train with best_C, score outer_val
        prob_val, vec, clfs = get_outer_scores_and_thresholds(
            outer_train_texts, y_outer_train, outer_val_texts, best_C, best_thr
        )
        outer_scores[val_idx] = prob_val
        outer_pred[val_idx] = apply_thresholds(prob_val, best_thr)
        outer_info.append(
            {
                "outer_fold": outer_idx,
                "best_C": best_C,
                "best_inner_macro": best_macro,
                "thresholds": best_thr,
                "inner_details": details,
                "vocab_size": len(vec.vocabulary_),
                "n_outer_train": len(train_idx),
                "n_outer_val": len(val_idx),
            }
        )
    return {"nested_predictions": outer_pred, "nested_scores": outer_scores, "outer_fold_info": outer_info}


def run_nested_embedding(gold_df, y, texts, outer_splits, seed=42):
    """
    S2: frozen embeddings, only thresholds are nested.
    Embedding model itself never sees labels.
    """
    n = len(gold_df)
    outer_pred = np.zeros((n, len(CATEGORIES)), dtype=int)
    outer_scores = np.zeros((n, len(CATEGORIES)), dtype=float)
    outer_info = []
    cat_embs = get_category_embeddings()
    # Precompute all posting scores once (frozen, no leakage concern for scores themselves; thresholds still nested)
    all_scores = embedding_scores(texts, cat_embs=cat_embs)
    # But we must ensure thresholds for each outer fold use only outer_train scores/labels.
    # So we slice all_scores for threshold tuning per fold.
    for outer_idx, (train_idx, val_idx) in enumerate(outer_splits):
        # Inner threshold tuning: 2-fold CV over outer_train scores
        roles_outer = gold_df.iloc[train_idx]["role_family"].values
        try:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
            inner_splits = list(skf.split(np.zeros(len(train_idx)), roles_outer))
        except ValueError:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
            inner_splits = list(kf.split(np.zeros(len(train_idx))))
        # Inner OOF thresholds: for each inner split, we have outer_train scores
        # We want to choose thresholds that maximise inner macro-F1, evaluated via inner OOF predictions.
        # Simplest: tune thresholds on the concatenation of inner validation predictions? Instead we do same as lexical nested:
        # For each inner fold, tune thresholds on inner_train, evaluate on inner_val, then average? Lexical does tuning per inner?
        # To keep consistent, we will pool inner OOF: tune on full outer_train via 2-fold OOF evaluation.
        # Implementation: for each inner split, we could tune thresholds on inner_train scores then apply to inner_val, but S2 scores are precomputed, so we just tune on inner_train.
        # Instead we will do: tune thresholds on outer_train scores using inner CV for selection (same as supervised: try thresholds on inner_train, score inner_val).
        # We replicate lexical nested threshold selection: for each inner split, tune on inner_train, score inner_val, aggregate macro.
        # But thresholds are not hyperparameter of S2 beyond themselves; we just pick thresholds that maximise inner macro directly on outer_train.
        # For simplicity and leakage safety, we tune thresholds on outer_train via inner CV: search thresholds using inner_train only, evaluate on inner_val, select thresholds with best inner macro.
        # That is: try tuning on each inner_train fold's scores, evaluate on inner_val, pick thresholds with best avg inner macro.
        best_thr = None
        best_inner_macro = -1
        for inner_train_rel, inner_val_rel in inner_splits:
            # map relative outer_train indices to global
            # inner_*_rel are indices into outer_train arrays (0..len(train_idx)-1)
            thr_candidate = tune_thresholds(all_scores[train_idx][inner_train_rel], y[train_idx][inner_train_rel])
            pred_val = apply_thresholds(all_scores[train_idx][inner_val_rel], thr_candidate)
            rep = evaluate(y[train_idx][inner_val_rel], pred_val)
            macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
            if macro > best_inner_macro:
                best_inner_macro = macro
                best_thr = thr_candidate
        # Fallback: if no inner split worked, tune on full outer_train
        if best_thr is None:
            best_thr = tune_thresholds(all_scores[train_idx], y[train_idx])
            best_inner_macro = 0.0
        outer_scores[val_idx] = all_scores[val_idx]
        outer_pred[val_idx] = apply_thresholds(all_scores[val_idx], best_thr)
        outer_info.append(
            {
                "outer_fold": outer_idx,
                "thresholds": best_thr,
                "best_inner_macro": float(best_inner_macro),
                "n_outer_train": len(train_idx),
                "n_outer_val": len(val_idx),
            }
        )
    return {"nested_predictions": outer_pred, "nested_scores": outer_scores, "outer_fold_info": outer_info, "cat_embs": cat_embs}


def run_nested_nli(gold_df, y, texts, outer_splits, seed=42):
    """
    S3: frozen NLI, only thresholds nested, MAX aggregation already done per posting.
    """
    n = len(gold_df)
    outer_pred = np.zeros((n, len(CATEGORIES)), dtype=int)
    outer_scores = np.zeros((n, len(CATEGORIES)), dtype=float)
    outer_info = []
    # Precompute all NLI scores (frozen, no label influence) — use cache to avoid 650s recomputation
    all_scores, _, _ = _get_nli_scores_cached(texts)
    for outer_idx, (train_idx, val_idx) in enumerate(outer_splits):
        roles_outer = gold_df.iloc[train_idx]["role_family"].values
        try:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
            inner_splits = list(skf.split(np.zeros(len(train_idx)), roles_outer))
        except ValueError:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=2, shuffle=True, random_state=seed + 1000 * outer_idx + 7)
            inner_splits = list(kf.split(np.zeros(len(train_idx))))
        best_thr = None
        best_inner_macro = -1
        for inner_train_rel, inner_val_rel in inner_splits:
            thr_candidate = tune_thresholds(all_scores[train_idx][inner_train_rel], y[train_idx][inner_train_rel])
            pred_val = apply_thresholds(all_scores[train_idx][inner_val_rel], thr_candidate)
            rep = evaluate(y[train_idx][inner_val_rel], pred_val)
            macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
            if macro > best_inner_macro:
                best_inner_macro = macro
                best_thr = thr_candidate
        if best_thr is None:
            best_thr = tune_thresholds(all_scores[train_idx], y[train_idx])
            best_inner_macro = 0.0
        outer_scores[val_idx] = all_scores[val_idx]
        outer_pred[val_idx] = apply_thresholds(all_scores[val_idx], best_thr)
        outer_info.append(
            {
                "outer_fold": outer_idx,
                "thresholds": best_thr,
                "best_inner_macro": float(best_inner_macro),
                "n_outer_train": len(train_idx),
                "n_outer_val": len(val_idx),
            }
        )
    return {"nested_predictions": outer_pred, "nested_scores": outer_scores, "outer_fold_info": outer_info}
