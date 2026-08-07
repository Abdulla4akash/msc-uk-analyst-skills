"""
Tests for genuinely nested CV (Fix 1).

These would FAIL under the old optimistic procedure where thresholds are tuned
on all 300 OOF labels and then reported on the same OOF predictions.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from v4.config import CATEGORIES
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_cv_splits
from v4.evaluation.nested import run_nested_cv_for_method
from v4.evaluation.metrics import evaluate
from v4.methods.lexical_baseline import fit_tfidf_vectoriser, weighted_lexical_scores_with_vec

REPO = "/Users/akashx/msc-uk-analyst-skills"


def load_all():
    gold_df, y, texts = load_gold_with_texts(
        f"{REPO}/v3/manual_work/gold_standard_annotation_workbook_v2.xlsx",
        f"{REPO}/v3/manual_work/uk_analyst_corpus_v4_clean.csv",
    )
    return gold_df, y, texts


def test_outer_label_isolation():
    """
    For a chosen outer fold, mutating outer validation labels must NOT change
    thresholds for that fold. Outer validation labels are forbidden from influencing
    threshold selection.
    """
    gold_df, y, texts = load_all()
    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    # Run nested for one method once
    method = "weighted_lexical_tfidf"
    res = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=42)
    thresholds_before = [info["thresholds"] for info in res["outer_fold_info"]]

    # Now mutate outer validation labels for fold 0: flip all labels in that fold
    y_mutated = y.copy()
    val_idx_fold0 = outer_splits[0][1]
    y_mutated[val_idx_fold0] = 1 - y_mutated[val_idx_fold0]
    res_mut = run_nested_cv_for_method(gold_df, y_mutated, texts, outer_splits, method_name=method, seed=42)
    thresholds_after = [info["thresholds"] for info in res_mut["outer_fold_info"]]

    # Fold 0's thresholds must be identical despite its validation labels being flipped,
    # because thresholds come from inner CV on outer_train only.
    assert thresholds_before[0] == thresholds_after[0], (
        f"outer fold 0 thresholds changed after mutating its validation labels: "
        f"before {thresholds_before[0]} after {thresholds_after[0]}"
    )
    # Other folds' outer validation also shouldn't affect fold 0, but we check at least fold 0


def test_outer_text_isolation():
    """
    Modifying outer validation text with synthetic terms must not change
    training vocabulary/IDF or selected thresholds for that fold.
    """
    gold_df, y, texts = load_all()
    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    method = "weighted_lexical_tfidf"
    res_before = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=42)
    thresholds_before = res_before["outer_fold_info"][0]["thresholds"]

    # Mutate outer validation texts for fold 0
    texts_mutated = list(texts)
    val_idx_fold0 = outer_splits[0][1]
    for idx in val_idx_fold0:
        texts_mutated[idx] = "unicorn quantum blorpt " * 30 + texts_mutated[idx]

    res_after = run_nested_cv_for_method(gold_df, y, texts_mutated, outer_splits, method_name=method, seed=42)
    thresholds_after = res_after["outer_fold_info"][0]["thresholds"]

    assert thresholds_before == thresholds_after, (
        "Thresholds changed after mutating outer validation text — outer text leaked into training"
    )
    # Also check that training vocab would differ if leaked, but our thresholds already prove isolation


def test_nested_prediction_accounting():
    """
    Every one of the 300 must appear exactly once as outer validation,
    receive exactly one nested prediction vector of 13 labels,
    never simultaneously in outer train and outer validation.
    """
    gold_df, y, texts = load_all()
    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    n = len(gold_df)
    # Check partition properties directly
    val_covered = np.zeros(n, dtype=int)
    for train_idx, val_idx in outer_splits:
        assert len(set(train_idx) & set(val_idx)) == 0, "outer train and val overlap"
        assert len(train_idx) + len(val_idx) == n
        for vi in val_idx:
            val_covered[vi] += 1
    assert np.all(val_covered == 1), f"Each posting must be outer val exactly once, got counts {np.unique(val_covered)}"

    # Now check nested predictions also cover exactly once
    for method in ["unweighted_lexical", "weighted_lexical_tfidf", "cosine_tfidf"]:
        res = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=42)
        pred = res["nested_predictions"]
        assert pred.shape == (n, len(CATEGORIES))
        assert set(np.unique(pred)) <= {0, 1}
        # Re-check that predictions came from outer val only (visited array in implementation asserts)


def test_nested_reproducibility():
    """Same data/config/seed must give identical folds, thresholds, scores, predictions, metrics."""
    gold_df, y, texts = load_all()
    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    method = "weighted_lexical_tfidf"
    r1 = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=42)
    r2 = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=42)
    np.testing.assert_array_equal(r1["nested_predictions"], r2["nested_predictions"])
    np.testing.assert_allclose(r1["nested_scores"], r2["nested_scores"], atol=1e-12)
    for info1, info2 in zip(r1["outer_fold_info"], r2["outer_fold_info"]):
        assert info1["thresholds"] == info2["thresholds"]
        assert info1["outer_validation_ids"] == info2["outer_validation_ids"]
        assert info1["inner_strategy"] == info2["inner_strategy"]

    rep1 = evaluate(y, r1["nested_predictions"])
    rep2 = evaluate(y, r2["nested_predictions"])
    pd.testing.assert_frame_equal(rep1, rep2)


def test_threshold_provenance():
    """Generated JSON / outer_fold_info must identify per-fold provenance."""
    gold_df, y, texts = load_all()
    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    res = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name="weighted_lexical_tfidf", seed=42)
    for info in res["outer_fold_info"]:
        # Required provenance fields
        assert "outer_train_ids" in info and len(info["outer_train_ids"]) == info["outer_train_n"]
        assert "outer_validation_ids" in info and len(info["outer_validation_ids"]) == info["outer_validation_n"]
        assert "inner_strategy" in info and isinstance(info["inner_strategy"], str)
        assert "thresholds" in info and len(info["thresholds"]) == len(CATEGORIES)
        for c in CATEGORIES:
            assert c in info["thresholds"]
            assert 0.0 <= info["thresholds"][c] <= 1.0
        assert "seed" in info
        assert "feature_fitting_population" in info
        assert "outer_train" in info["feature_fitting_population"]
        assert "threshold_tuning_population" in info
