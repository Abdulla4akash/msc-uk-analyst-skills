"""
Cumulative lexical ablation definitions (CASE B — seed lexicon not recoverable).

A0: naive substring, final frozen lexicon, no negatives, any hit (>0) → positive
A1: A0 + whole-word / phrase-safe matching
A2: A1 + frozen NEGATIVE_PATTERNS suppression
A3: OMITTED — lexicon expansion not identifiable from provenance (recorded as NOT IDENTIFIABLE)
A4: A2 + continuous unweighted score (distinct matched / lexicon size) + genuinely nested per-category thresholds
A5: A4 + inductive IDF weighting (TF-IDF fitted on inner/outer train only) + nested thresholds

All scoring uses final frozen LEXICONS from v4/config.py (do NOT edit that file).
Negative patterns use frozen NEGATIVE_PATTERNS.

A0–A3 use fixed any-hit rule: positive if score > 0 (NOT >=0).
A4–A5 use nested CV via evaluation.nested (thresholds tuned on inner val only, outer val invisible).
"""

import re
import numpy as np

from v4.config import CATEGORIES, LEXICONS, NEGATIVE_PATTERNS
from v4.methods.lexical_baseline import (
    unweighted_lexical_scores,  # for A4 continuous score
    fit_tfidf_vectoriser,
    weighted_lexical_scores_with_vec,
    cosine_tfidf_scores_with_vec,  # not used in ablation but kept for consistency
)


def mask_negative_patterns(text, category):
    for pat in NEGATIVE_PATTERNS.get(category, []):
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


# ---------- Scoring primitives ----------

def _substring_scores(texts):
    """A0: naive case-insensitive substring matching (no word boundaries)."""
    S = np.zeros((len(texts), len(CATEGORIES)), dtype=float)
    lex_lower = {c: [t.lower() for t in LEXICONS[c]] for c in CATEGORIES}
    for j, cat in enumerate(CATEGORIES):
        terms = lex_lower[cat]
        n_terms = len(terms) if terms else 1
        for i, text in enumerate(texts):
            low = text.lower()
            hits = sum(1 for t in terms if t in low)
            S[i, j] = hits / n_terms
    return S


def _wholeword_scores(texts, use_negative=False):
    """A1/A2: whole-word / phrase-safe matching, optionally with negative suppression."""
    S = np.zeros((len(texts), len(CATEGORIES)), dtype=float)
    for j, cat in enumerate(CATEGORIES):
        terms = [t.lower() for t in LEXICONS[cat]]
        patterns = [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), t) for t in terms]
        n_terms = len(terms) if terms else 1
        for i, text in enumerate(texts):
            cleaned = mask_negative_patterns(text, cat) if use_negative else text
            hits = sum(1 for pat, _ in patterns if pat.search(cleaned))
            S[i, j] = hits / n_terms
    return S


def any_hit_predictions(scores):
    """Fixed rule for A0–A3: positive if score > 0."""
    return (scores > 0).astype(int)


# ---------- Ablation variant interface ----------

ABLATION_VARIANTS = ["A0", "A1", "A2", "A4", "A5"]  # A3 omitted
ABLATION_DEFINITIONS = {
    "A0": {
        "lexicon": "final frozen (v4/config.py)",
        "substring": True,
        "whole_word": False,
        "negatives": False,
        "thresholds": "fixed any-hit (>0)",
        "idf": False,
        "description": "naive case-insensitive substring, final lexicon, no negatives, any hit → positive",
    },
    "A1": {
        "lexicon": "final frozen",
        "substring": False,
        "whole_word": True,
        "negatives": False,
        "thresholds": "fixed any-hit (>0)",
        "idf": False,
        "description": "whole-word/phrase-safe matching, final lexicon, no negatives",
    },
    "A2": {
        "lexicon": "final frozen",
        "substring": False,
        "whole_word": True,
        "negatives": True,
        "thresholds": "fixed any-hit (>0)",
        "idf": False,
        "description": "whole-word + frozen NEGATIVE_PATTERNS, final lexicon",
    },
    "A3": {
        "lexicon": "NOT IDENTIFIABLE",
        "substring": False,
        "whole_word": False,
        "negatives": False,
        "thresholds": "NOT IDENTIFIABLE",
        "idf": False,
        "description": "lexicon expansion — NOT IDENTIFIABLE FROM AVAILABLE PROVENANCE (omitted)",
    },
    "A4": {
        "lexicon": "final frozen",
        "substring": False,
        "whole_word": True,
        "negatives": True,
        "thresholds": "genuinely nested per-category (inner CV on outer_train only)",
        "idf": False,
        "description": "A2 + continuous unweighted score + nested thresholds — reproduces unweighted_lexical",
    },
    "A5": {
        "lexicon": "final frozen",
        "substring": False,
        "whole_word": True,
        "negatives": True,
        "thresholds": "genuinely nested per-category (inner CV on outer_train only)",
        "idf": True,
        "description": "A4 + inductive IDF weighting — reproduces weighted_lexical_tfidf",
    },
}


def score_for_variant(variant, texts):
    """
    Score matrix for fixed-rule variants A0–A2 (continuous score in [0,1] but predictions are any-hit).
    Returns S (n,13).
    """
    if variant == "A0":
        return _substring_scores(texts)
    elif variant == "A1":
        return _wholeword_scores(texts, use_negative=False)
    elif variant == "A2":
        return _wholeword_scores(texts, use_negative=True)
    else:
        raise ValueError(f"score_for_variant only for A0–A2, got {variant}")


def predict_fixed_variant(variant, texts):
    """Direct predictions for A0–A2 (any-hit rule)."""
    S = score_for_variant(variant, texts)
    return any_hit_predictions(S), S


# For A4/A5, scoring + prediction are handled via nested CV in the runner;
# we expose score functions for audit purposes.

def unweighted_scores(texts):
    """Continuous unweighted score (used for A4 and audits)."""
    return unweighted_lexical_scores(texts)


def weighted_scores_with_vec(vec, texts):
    """IDF-weighted score with already-fitted vec (used for A5 and audits)."""
    return weighted_lexical_scores_with_vec(vec, texts)
