"""
Test C — threshold isolation + tuning robustness.

Threshold tuning must use only the supplied development/training labels.
We also verify that the CV tuning path respects train/val separation.
"""

import numpy as np
import pandas as pd
import pytest

from v4.methods.lexical_baseline import (
    tune_thresholds,
    tune_thresholds_cv,
    fit_tfidf_vectoriser,
    weighted_lexical_scores_with_vec,
    unweighted_lexical_scores,
)
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_cv_splits

from v4.tests._paths import GOLD_PATH, CORPUS_PATH


def test_threshold_tuning_ignores_heldout_labels():
    rng = np.random.default_rng(0)
    n_tuning, n_holdout = 40, 60
    n_cats = 3
    scores_tuning = rng.random((n_tuning, n_cats))
    y_tuning = (scores_tuning > 0.5).astype(int)
    scores_holdout = rng.random((n_holdout, n_cats))
    y_holdout_random = rng.integers(0, 2, size=(n_holdout, n_cats))

    thr_tuning_only = tune_thresholds(scores_tuning, y_tuning)
    thr_leaky = tune_thresholds(np.vstack([scores_tuning, scores_holdout]), np.vstack([y_tuning, y_holdout_random]))
    thr_tuning_again = tune_thresholds(scores_tuning, y_tuning)
    np.testing.assert_allclose(thr_tuning_only, thr_tuning_again, atol=1e-12)
    y_holdout_zeros = np.zeros((n_holdout, n_cats), dtype=int)
    thr_leaky_zeros = tune_thresholds(np.vstack([scores_tuning, scores_holdout]), np.vstack([y_tuning, y_holdout_zeros]))
    assert thr_tuning_only.shape == (n_cats,)


def test_cv_thresholds_use_train_val_separation():
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    splits, meta = make_cv_splits(gold_df, texts=texts, n_splits=3, n_repeats=1, seed=42)
    assert len(splits) == 3
    n = len(gold_df)
    oof = np.zeros((n, y.shape[1]), dtype=float)
    for train_idx, val_idx in splits:
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        vec = fit_tfidf_vectoriser(train_texts)
        oof[val_idx] = weighted_lexical_scores_with_vec(vec, val_texts)

    thr_cv = tune_thresholds_cv(oof, y, splits)
    assert thr_cv.shape == (y.shape[1],)
    assert np.all(thr_cv >= 0) and np.all(thr_cv <= 1)

    thr_flat = tune_thresholds(oof, y)
    assert thr_cv.shape == thr_flat.shape


def test_threshold_grid_is_documented_and_bounded():
    rng = np.random.default_rng(1)
    scores = rng.random((20, 2))
    y = rng.integers(0, 2, size=(20, 2))
    thr = tune_thresholds(scores, y)
    assert np.all(thr >= 0) and np.all(thr <= 1), "thresholds out of [0,1]"


def test_rare_category_threshold_stability():
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    eth_idx = 12  # last category
    assert y[:, eth_idx].sum() == 13
    splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    n = len(gold_df)
    oof = np.zeros((n, y.shape[1]), dtype=float)
    for train_idx, val_idx in splits:
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        vec = fit_tfidf_vectoriser(train_texts)
        oof[val_idx] = weighted_lexical_scores_with_vec(vec, val_texts)
    thr = tune_thresholds_cv(oof, y, splits)
    assert not np.any(np.isnan(thr))
    assert thr[eth_idx] >= 0 and thr[eth_idx] <= 1
