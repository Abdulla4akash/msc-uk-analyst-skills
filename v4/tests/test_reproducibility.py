"""
Test D — reproducibility.
Test E — label-order integrity.
Test F — basic accounting.

Also covers taxonomy version pinning.
"""

import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path

from v4.config import CATEGORIES, TAXONOMY_VERSION
from v4.evaluation.splits import make_dev_test_split
from v4.evaluation.data import load_gold_with_texts
from v4.methods.lexical_baseline import (
    unweighted_lexical_scores,
    weighted_lexical_scores,
    fit_tfidf_vectoriser,
    weighted_lexical_scores_with_vec,
)

from v4.tests._paths import GOLD_PATH, CORPUS_PATH, REPO_ROOT


def load_all():
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    return gold_df, y, texts


# ---- D: reproducibility ----

def test_split_reproducibility_same_seed():
    gold_df, y, texts = load_all()
    s1 = make_dev_test_split(gold_df, seed=42)
    s2 = make_dev_test_split(gold_df, seed=42)
    np.testing.assert_array_equal(s1["is_internal_tuning"], s2["is_internal_tuning"])
    assert s1["internal_tuning_ids"] == s2["internal_tuning_ids"]
    assert s1["internal_holdout_ids"] == s2["internal_holdout_ids"]


def test_split_varies_with_different_seed():
    gold_df, y, texts = load_all()
    s1 = make_dev_test_split(gold_df, seed=42)
    s2 = make_dev_test_split(gold_df, seed=99)
    assert not np.array_equal(s1["is_internal_tuning"], s2["is_internal_tuning"])


def test_predictions_reproducible_same_seed_and_config():
    gold_df, y, texts = load_all()
    split = make_dev_test_split(gold_df, seed=42)
    is_tuning = split["is_internal_tuning"]
    tuning_texts = [texts[i] for i in np.where(is_tuning)[0]]
    vec1 = fit_tfidf_vectoriser(tuning_texts)
    s1 = weighted_lexical_scores_with_vec(vec1, texts)
    vec2 = fit_tfidf_vectoriser(tuning_texts)
    s2 = weighted_lexical_scores_with_vec(vec2, texts)
    np.testing.assert_allclose(s1, s2, atol=1e-12)
    u1 = unweighted_lexical_scores(texts)
    u2 = unweighted_lexical_scores(texts)
    np.testing.assert_allclose(u1, u2, atol=1e-12)


# ---- E: label-order integrity ----

def test_categories_order_and_count():
    expected = [
        "programming",
        "sql",
        "visualisation_bi",
        "reporting",
        "excel",
        "statistics",
        "machine_learning",
        "data_cleaning",
        "etl",
        "data_modelling",
        "cloud",
        "stakeholder_comm",
        "ethics_governance",
    ]
    assert CATEGORIES == expected, "CATEGORIES order must match taxonomy definition"
    assert len(CATEGORIES) == 13
    assert TAXONOMY_VERSION == "v3-13cat-frozen"


def test_prediction_shape_and_column_order():
    gold_df, y, texts = load_all()
    preds = unweighted_lexical_scores(texts)  # (n,13)
    assert preds.shape == (len(gold_df), 13)
    df = pd.DataFrame(preds, columns=CATEGORIES)
    assert df.columns.tolist() == CATEGORIES
    assert gold_df[CATEGORIES].columns.tolist() == CATEGORIES


def test_v4_results_columns_follow_taxonomy():
    outdir = REPO_ROOT / "v4" / "results"
    if not outdir.exists():
        pytest.skip("no v4 results yet")
    for f in outdir.glob("v4_*_predictions_gold*.csv"):
        df = pd.read_csv(f, nrows=2)
        assert "posting_id" in df.columns
        cat_cols = [c for c in df.columns if c in CATEGORIES]
        assert cat_cols == CATEGORIES, f"{f.name} category columns out of order: {cat_cols}"


# ---- F: basic accounting ----

def test_no_duplicate_ids_across_splits():
    gold_df, y, texts = load_all()
    split = make_dev_test_split(gold_df, seed=42)
    assert "is_internal_tuning" in split and "is_internal_holdout" in split
    tuning_ids = set(split["internal_tuning_ids"])
    holdout_ids = set(split["internal_holdout_ids"])
    assert len(tuning_ids & holdout_ids) == 0, "internal_tuning and internal_holdout share posting_ids"
    assert len(tuning_ids) + len(holdout_ids) == len(gold_df)
    combined = tuning_ids | holdout_ids
    assert combined == set(gold_df["posting_id"].astype(str).tolist())


def test_external_locked_test_is_reserved():
    summary_path = REPO_ROOT / "v4" / "results" / "v4_lexical_summary.json"
    if not summary_path.exists():
        pytest.skip("summary not yet generated")
    summary = json.loads(summary_path.read_text())
    assert "external_locked_test" in str(summary.get("data_flow", {}))
    assert not list((REPO_ROOT / "v4" / "results").glob("*external*"))
    readme = (REPO_ROOT / "v4" / "README.md").read_text()
    assert "external_locked_test" in readme
    assert "does not exist" in readme


def test_prediction_and_gold_matrix_shapes_match():
    gold_df, y, texts = load_all()
    preds = unweighted_lexical_scores(texts)
    assert preds.shape == y.shape
    assert preds.shape[1] == len(CATEGORIES)


def test_every_gold_posting_in_accounting():
    from v4.evaluation.metrics import accounting_report
    gold_df, y, texts = load_all()
    preds = (unweighted_lexical_scores(texts) >= 0.02).astype(int)
    acc = accounting_report(y, preds)
    assert acc["n_postings"] == len(gold_df)
    assert acc["total_cells"] == len(gold_df) * len(CATEGORIES)
    assert acc["total_TP"] + acc["total_FP"] + acc["total_FN"] + acc["total_TN"] == acc["total_cells"]


def test_label_prevalence_sums():
    gold_df, y, texts = load_all()
    prevalence = {c: float(y[:, i].mean()) for i, c in enumerate(CATEGORIES)}
    assert prevalence["stakeholder_comm"] > 0.7
    assert prevalence["ethics_governance"] < 0.1
    assert sum(prevalence.values()) == pytest.approx(y.sum() / y.size * len(CATEGORIES), rel=1e-6)


def test_taxonomy_version_recorded_in_summary():
    summary_path = REPO_ROOT / "v4" / "results" / "v4_lexical_summary.json"
    if not summary_path.exists():
        pytest.skip("summary not yet generated")
    summary = json.loads(summary_path.read_text())
    assert summary["taxonomy_version"] == TAXONOMY_VERSION
    assert "vectoriser_config" in summary
    assert "random_seed" in summary
    assert "package_versions" in summary
