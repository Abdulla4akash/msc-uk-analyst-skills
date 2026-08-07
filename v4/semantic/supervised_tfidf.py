"""
S1 — supervised TF-IDF logistic regression (multi-label one-vs-rest).

Leakage-safe: vectoriser + classifier fitted only on inner_train / outer_train.
No outer text/label influence on vocabulary/IDF/C/threshold.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from v4.config import CATEGORIES
from v4.methods.lexical_baseline import tune_thresholds, apply_thresholds

from v4.semantic.model_config import S1_VECTORISER_CONFIG, S1_C_GRID, S1_CLASS_WEIGHT


def build_vectoriser():
    return TfidfVectorizer(
        lowercase=S1_VECTORISER_CONFIG["lowercase"],
        stop_words=S1_VECTORISER_CONFIG["stop_words"],
        ngram_range=S1_VECTORISER_CONFIG["ngram_range"],
        min_df=S1_VECTORISER_CONFIG["min_df"],
        max_df=S1_VECTORISER_CONFIG["max_df"],
        sublinear_tf=S1_VECTORISER_CONFIG["sublinear_tf"],
    )


def _inner_oof_scores(texts, y, C, class_weight, inner_splits):
    """
    For a given C, produce inner out-of-fold probability matrix (n,13) via inner CV.
    Each inner model fitted only on inner_train.
    """
    n = len(texts)
    oof = np.zeros((n, len(CATEGORIES)), dtype=float)
    for train_idx, val_idx in inner_splits:
        vec = build_vectoriser()
        X_train = vec.fit_transform([texts[i] for i in train_idx])
        X_val = vec.transform([texts[i] for i in val_idx])
        for j in range(len(CATEGORIES)):
            clf = LogisticRegression(
                C=C,
                class_weight=class_weight,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            clf.fit(X_train, y[train_idx, j])
            prob = clf.predict_proba(X_val)[:, 1]
            oof[val_idx, j] = prob
    return oof


def select_hyperparameters_inner_cv(texts, y, inner_splits, verbose=False):
    """
    Nested model selection: try each C, compute inner OOF probabilities, tune thresholds on OOF, score macro-F1.
    Returns best_C, best_thresholds (dict), best_inner_macro, details.
    Thresholds tuned only on inner data (OOF).
    """
    from v4.evaluation.metrics import evaluate
    best_macro = -1
    best_C = None
    best_thr = None
    details = []
    for C in S1_C_GRID:
        oof = _inner_oof_scores(texts, y, C, S1_CLASS_WEIGHT, inner_splits)
        thr = tune_thresholds(oof, y)
        pred = apply_thresholds(oof, thr)
        rep = evaluate(y, pred)
        macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
        details.append({"C": C, "macro": macro, "thresholds": thr})
        # deterministic tie-break: first max wins (grid order)
        if macro > best_macro:
            best_macro = macro
            best_C = C
            best_thr = thr
    return best_C, best_thr, best_macro, details


def fit_outer_and_score(outer_train_texts, y_outer_train, outer_val_texts):
    """
    Fit on full outer_train with already-selected C/threshold? This is helper after selection.
    Returns vec, classifiers dict, proba matrix for outer_val, thresholds used.
    """
    raise NotImplementedError("use run_nested path")


def get_outer_scores_and_thresholds(outer_train_texts, y_outer_train, outer_val_texts, best_C, best_thr):
    vec = build_vectoriser()
    X_train = vec.fit_transform(outer_train_texts)
    X_val = vec.transform(outer_val_texts)
    prob_val = np.zeros((len(outer_val_texts), len(CATEGORIES)), dtype=float)
    clfs = []
    for j in range(len(CATEGORIES)):
        clf = LogisticRegression(
            C=best_C, class_weight=S1_CLASS_WEIGHT, solver="lbfgs", max_iter=1000, random_state=42
        )
        clf.fit(X_train, y_outer_train[:, j])
        prob_val[:, j] = clf.predict_proba(X_val)[:, 1]
        clfs.append(clf)
    return prob_val, vec, clfs
