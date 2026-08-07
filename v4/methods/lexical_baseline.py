"""
v4 classical baselines with corrected evaluation protocol.

All methods produce a score matrix S of shape (n, 13) in CATEGORIES order.
Scores are BATCH-INVARIANT: evaluating one posting alone gives the same score
as evaluating it inside a larger batch.

Baselines
---------
Baseline 0 — unweighted lexical match
  For each category, detect valid lexicon terms (whole-word, case-insensitive)
  after masking negative patterns. Score is:
    (# distinct matched terms) / (lexicon size for that category)
  This is deterministic, in [0,1], and batch-invariant. Binary prediction can
  be derived via thresholds.

Baseline 1 — cosine TF-IDF (inductive)
  Fit TfidfVectorizer on TRAINING texts only. Category pseudo-documents are
  built from LEXICONS and transformed with the same vectoriser. Cosine similarity
  is returned RAW (no batch-max division). This is the natural similarity scale
  in [0,1] (cosine is already normalised by vector norms).

Baseline 2 — weighted lexical TF-IDF (inductive, stable scoring)
  Fit TfidfVectorizer on TRAINING texts only. For each category, sum IDF weights
  of lexicon terms present in the posting, divided by sum of IDF weights for the
  whole lexicon (so score in [0,1]). No batch-max normalisation.

Invariants
----------
- vectoriser fitted only on supplied train_texts (never on val/test texts)
- no score depends on other documents in the batch (no max-normalisation)
- thresholds are NOT learned here; tuning happens externally on training/validation splits
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from v4.config import CATEGORIES, LEXICONS, NEGATIVE_PATTERNS

# Shared vectoriser config — identical to v3 for comparability, except fitting scope.
VECTORISER_CONFIG = dict(
    sublinear_tf=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#/.\-]+\b",
)


def build_category_documents():
    return [" ".join(LEXICONS[c]) for c in CATEGORIES]


def mask_negative_patterns(text, category):
    for pat in NEGATIVE_PATTERNS.get(category, []):
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


# ---------- Baseline 0: unweighted lexical ----------

def unweighted_lexical_scores(target_texts):
    """
    Stable unweighted lexical scores.

    For each (posting, category): fraction of distinct lexicon terms present
    after negative-pattern masking. No IDF, no vectoriser, batch-invariant.

    Returns ndarray (n, 13) in [0,1].
    """
    S = np.zeros((len(target_texts), len(CATEGORIES)), dtype=float)
    for j, cat in enumerate(CATEGORIES):
        terms = [t.lower() for t in LEXICONS[cat]]
        # compile whole-word patterns once per category
        patterns = [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), t) for t in terms]
        n_terms = len(terms) if terms else 1
        for i, text in enumerate(target_texts):
            cleaned = mask_negative_patterns(text, cat)
            hits = sum(1 for pat, _ in patterns if pat.search(cleaned))
            S[i, j] = hits / n_terms
    return S


# ---------- Baseline 1: cosine TF-IDF (inductive) ----------

def fit_tfidf_vectoriser(train_texts, config=None):
    """
    Fit TfidfVectorizer on train_texts ONLY.

    Returns fitted vectoriser.
    Caller must NOT pass test texts here.
    """
    if config is None:
        config = VECTORISER_CONFIG
    vec = TfidfVectorizer(**config)
    vec.fit(train_texts)
    return vec


def cosine_tfidf_scores(train_texts, target_texts, vectoriser_config=None):
    """
    Inductive cosine scoring.

    Fit on train_texts, transform target_texts + category pseudo-docs,
    return raw cosine similarities (no batch normalisation).

    Returns (S, vec) where S shape (n_target, 13) in [0,1].
    """
    vec = fit_tfidf_vectoriser(train_texts, config=vectoriser_config)
    S = cosine_tfidf_scores_with_vec(vec, target_texts)
    return S, vec


def cosine_tfidf_scores_with_vec(vec, target_texts):
    """
    Score target_texts using an already-fitted vectoriser.
    This is the transform-only path for validation/test.
    Batch-invariant: no normalisation across target_texts.
    """
    X = vec.transform(target_texts)
    C = vec.transform(build_category_documents())
    S = cosine_similarity(X, C)
    # Clip to [0,1] for safety (cosine is already in that range for non-negative TF-IDF)
    S = np.clip(S, 0.0, 1.0)
    return S


# ---------- Baseline 2: weighted lexical TF-IDF (inductive) ----------

def weighted_lexical_scores(train_texts, target_texts, vectoriser_config=None):
    """
    Inductive weighted lexical scoring.

    Fit vectoriser on train_texts, then for each category compute
    sum(IDF weights of matched terms) / sum(IDF weights of all terms).

    No batch-max normalisation.

    Returns (S, vec).
    """
    vec = fit_tfidf_vectoriser(train_texts, config=vectoriser_config)
    S = weighted_lexical_scores_with_vec(vec, target_texts)
    return S, vec


def weighted_lexical_scores_with_vec(vec, target_texts):
    """
    Score target_texts using already-fitted vectoriser.
    Batch-invariant.
    """
    vocab = vec.vocabulary_
    idf = vec.idf_
    # Precompute per-category pattern + normalised IDF weights
    cat_patterns = []
    for cat in CATEGORIES:
        terms = [t.lower() for t in LEXICONS[cat]]
        # IDF lookup: if term not in vocab (e.g. rare ngram filtered by min_df),
        # treat weight as 0 contribution to denominator?  We include it with weight 1.0
        # in denominator so missing vocab doesn't silently boost scores, but we
        # also give it 0 achievable numerator if unseen.  Documented choice.
        # Simpler: use 1.0 for unseen in both numerator and denominator so score
        # definition is stable.  We do that for comparability with v3 behaviour
        # which also used 1.0 fallback.
        weights = np.array([idf[vocab[t]] if t in vocab else 1.0 for t in terms], dtype=float)
        denom = weights.sum() if weights.sum() != 0 else 1.0
        # store (pattern, weight, denom) — pattern compiled per term
        patterns = [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), w) for t, w in zip(terms, weights)]
        cat_patterns.append((patterns, denom))

    S = np.zeros((len(target_texts), len(CATEGORIES)), dtype=float)
    for j, cat in enumerate(CATEGORIES):
        patterns, denom = cat_patterns[j]
        for i, text in enumerate(target_texts):
            cleaned = mask_negative_patterns(text, cat)
            s = 0.0
            for pat, w in patterns:
                if pat.search(cleaned):
                    s += float(w)
            S[i, j] = s / denom
    # Already in [0,1] by construction
    S = np.clip(S, 0.0, 1.0)
    return S


# ---------- Threshold helpers ----------

def tune_thresholds(scores, y_true, grid=None):
    """
    Pick per-category threshold maximising F1 on given data.

    scores, y_true: ndarrays (n,13).  Grid over [0,1].
    Returns thresholds ndarray (13,).
    """
    from sklearn.metrics import precision_recall_fscore_support
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    thresholds = np.zeros(scores.shape[1], dtype=float)
    for i in range(scores.shape[1]):
        best_f, best_t = -1.0, 0.5
        for t in grid:
            pred = (scores[:, i] >= t).astype(int)
            _, _, f, _ = precision_recall_fscore_support(
                y_true[:, i], pred, average="binary", zero_division=0
            )
            if f > best_f:
                best_f, best_t = float(f), float(t)
        thresholds[i] = best_t
    return thresholds


def tune_thresholds_cv(scores, y, cv_splits, grid=None):
    """
    Tune thresholds via cross-validation over development data.

    For each fold, thresholds are tuned on TRAIN portion; candidate thresholds
    are evaluated on VAL. We average F1 across folds for each threshold value
    per category and pick the best.

    If grid is fixed, this searches that grid. For efficiency we reuse
    the same grid for all folds.

    Returns thresholds ndarray (13,).
    """
    from sklearn.metrics import precision_recall_fscore_support
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)  # coarser for CV to keep cost reasonable

    n_cats = scores.shape[1]
    n_grid = len(grid)
    # per category, per threshold: mean F1 across folds
    mean_f1 = np.zeros((n_cats, n_grid), dtype=float)

    for train_idx, val_idx in cv_splits:
        s_val = scores[val_idx]
        y_val = y[val_idx]
        # We DO NOT re-tune thresholds inside; we evaluate every grid point
        # on val and accumulate.
        for gi, t in enumerate(grid):
            pred = (s_val >= t).astype(int)
            for ci in range(n_cats):
                _, _, f, _ = precision_recall_fscore_support(
                    y_val[:, ci], pred[:, ci], average="binary", zero_division=0
                )
                mean_f1[ci, gi] += float(f)
    mean_f1 /= len(cv_splits)
    thresholds = np.array([grid[int(np.argmax(mean_f1[ci]))] for ci in range(n_cats)], dtype=float)
    return thresholds


def apply_thresholds(scores, thresholds):
    """Apply per-category thresholds to score matrix -> binary predictions."""
    return (scores >= thresholds).astype(int)
