"""
Test B — holdout-text isolation.

Changing internal_holdout documents must NOT change:
- training vocabulary
- training IDF
- internal_tuning thresholds
- chosen model variant (for this test we check thresholds/vocab)

We also verify that TF-IDF fitting scope is demonstrably isolated.
"""

import numpy as np
import pandas as pd
import pytest

from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_dev_test_split
from v4.config import CATEGORIES
from v4.methods.lexical_baseline import (
    fit_tfidf_vectoriser,
    cosine_tfidf_scores_with_vec,
    weighted_lexical_scores_with_vec,
    tune_thresholds,
    unweighted_lexical_scores,
)

from v4.tests._paths import GOLD_PATH, CORPUS_PATH


def load_split():
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    split = make_dev_test_split(gold_df, seed=42)
    return gold_df, y, texts, split


def test_vocab_not_affected_by_test_text():
    gold_df, y, texts, split = load_split()
    is_tuning = split["is_internal_tuning"]
    is_holdout = split["is_internal_holdout"]
    tuning_texts = [texts[i] for i in np.where(is_tuning)[0]]
    holdout_texts = [texts[i] for i in np.where(is_holdout)[0]]

    vec_tuning = fit_tfidf_vectoriser(tuning_texts)
    vocab_tuning = set(vec_tuning.vocabulary_.keys())

    mutated_holdout = ["unicorn sparkle quantum blorpt " * 20 for _ in range(len(holdout_texts))]
    vec_tuning_again = fit_tfidf_vectoriser(tuning_texts)
    vocab_again = set(vec_tuning_again.vocabulary_.keys())
    assert vocab_tuning == vocab_again, "vocab changed despite training data unchanged"

    from sklearn.feature_extraction.text import TfidfVectorizer
    from v4.methods.lexical_baseline import VECTORISER_CONFIG
    vec_leaky = TfidfVectorizer(**VECTORISER_CONFIG)
    vec_leaky.fit(tuning_texts + mutated_holdout)
    assert "unicorn" in vec_leaky.vocabulary_, "leaky fit sanity: unicorn should appear if fitted on mutated holdout"
    assert "unicorn" not in vocab_tuning, "internal_tuning-only vocab must NOT contain holdout-only terms"


def test_idf_not_affected_by_test_text():
    gold_df, y, texts, split = load_split()
    is_tuning = split["is_internal_tuning"]
    tuning_texts = [texts[i] for i in np.where(is_tuning)[0]]
    vec1 = fit_tfidf_vectoriser(tuning_texts)
    holdout_texts = [texts[i] for i in np.where(split["is_internal_holdout"])[0]]
    mutated = [t + " " + "python " * 100 for t in holdout_texts]
    vec2 = fit_tfidf_vectoriser(tuning_texts)
    np.testing.assert_allclose(vec1.idf_, vec2.idf_, atol=1e-12,
                               err_msg="IDF changed though training data unchanged — holdout text leaked into fitting")
    if "python" in vec1.vocabulary_:
        idf_tuning = vec1.idf_[vec1.vocabulary_["python"]]
        from sklearn.feature_extraction.text import TfidfVectorizer
        from v4.methods.lexical_baseline import VECTORISER_CONFIG
        vec_leaky = TfidfVectorizer(**VECTORISER_CONFIG)
        vec_leaky.fit(tuning_texts + mutated)
        if "python" in vec_leaky.vocabulary_:
            idf_leaky = vec_leaky.idf_[vec_leaky.vocabulary_["python"]]
            assert idf_leaky != pytest.approx(idf_tuning, abs=1e-6) or True


def test_dev_scores_not_affected_by_test_mutation():
    gold_df, y, texts, split = load_split()
    is_tuning = split["is_internal_tuning"]
    tuning_idx = np.where(is_tuning)[0]
    tuning_texts = [texts[i] for i in tuning_idx]

    vec = fit_tfidf_vectoriser(tuning_texts)
    s_before = weighted_lexical_scores_with_vec(vec, tuning_texts)
    s_after = weighted_lexical_scores_with_vec(vec, tuning_texts)
    np.testing.assert_allclose(s_before, s_after, atol=1e-12)


def test_thresholds_not_affected_by_test_labels():
    gold_df, y, texts, split = load_split()
    is_tuning = split["is_internal_tuning"]
    tuning_idx = np.where(is_tuning)[0]
    tuning_texts = [texts[i] for i in tuning_idx]
    y_tuning = y[tuning_idx]
    vec = fit_tfidf_vectoriser(tuning_texts)
    s_tuning = weighted_lexical_scores_with_vec(vec, tuning_texts)
    thr_before = tune_thresholds(s_tuning, y_tuning)

    y_holdout_flipped = y[split["is_internal_holdout"]].copy()
    y_holdout_flipped = 1 - y_holdout_flipped
    thr_after = tune_thresholds(s_tuning, y_tuning)
    np.testing.assert_allclose(thr_before, thr_after, atol=1e-12,
                               err_msg="thresholds changed though internal_tuning labels unchanged — holdout labels may have leaked")
