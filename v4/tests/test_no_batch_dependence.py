"""
Test A — batch invariance.

Scores for a posting must be identical whether evaluated alone or inside a
batch with unrelated postings. Catches the old max-normalisation bug.
"""

import numpy as np
import pytest
from v4.methods.lexical_baseline import (
    unweighted_lexical_scores,
    cosine_tfidf_scores,
    weighted_lexical_scores,
    cosine_tfidf_scores_with_vec,
    weighted_lexical_scores_with_vec,
    fit_tfidf_vectoriser,
)


SAMPLE_TEXTS = [
    "We need strong Python and SQL experience, plus Tableau and Power BI for dashboards.",
    "Excellent communication skills, stakeholder management, presenting findings to non-technical audiences.",
    "Budget reports will be produced monthly — candidate reports directly to the Finance Director.",
    "Experience with AWS cloud, snowflake and databricks for ETL pipelines, plus GDPR compliance.",
    "Clinical coding and wellbeing programme experience required; also excellent in team collaboration.",
]

# Extra unrelated texts to enlarge the batch
DISTRACTORS = [
    "This posting is about gardening and horticulture, unrelated to data work.",
    "The role involves cooking and culinary arts in a restaurant.",
]


def _batch_invariance_check(scorer_with_train, target_isolated, train_texts):
    """
    Helper: score target_isolated alone vs inside batch [target + distractors],
    compare the target's row.
    scorer_with_train is a callable(train_texts, target_texts) -> (S, vec) or similar.
    """
    # Isolated
    S_iso, vec = scorer_with_train(train_texts, target_isolated)
    # Batch
    batch = target_isolated + DISTRACTORS
    S_batch = scorer_with_train(train_texts, batch)[0] if isinstance(scorer_with_train(train_texts, batch), tuple) else scorer_with_train(train_texts, batch)
    # For weighted/cosine we need to use same vec for isolated vs batch? scorer_with_train fits fresh,
    # so fit is identical. Compare first len(isolated) rows.
    # But to test batch-invariance properly we fit once and transform both ways.
    return S_iso, S_batch


def test_unweighted_batch_invariance():
    target = SAMPLE_TEXTS[:1]
    s_iso = unweighted_lexical_scores(target)
    s_batch = unweighted_lexical_scores(target + DISTRACTORS)
    np.testing.assert_allclose(s_iso[0], s_batch[0], atol=1e-12,
                               err_msg="unweighted lexical score changed with batch size")


def test_cosine_batch_invariance():
    # Need enough training docs for min_df=2 to keep some terms; 3 is too few.
    # Use the real corpus dev texts for a realistic fit, but we also test batch invariance logic alone.
    from v4.evaluation.data import load_gold_with_texts
    from v4.evaluation.splits import make_dev_test_split
    gold_df, y, texts = load_gold_with_texts(
        "/Users/akashx/msc-uk-analyst-skills/v3/manual_work/gold_standard_annotation_workbook_v2.xlsx",
        "/Users/akashx/msc-uk-analyst-skills/v3/manual_work/uk_analyst_corpus_v4_clean.csv",
    )
    split = make_dev_test_split(gold_df, seed=42)
    dev_texts = [texts[i] for i in np.where(split["is_dev"])[0]]
    vec = fit_tfidf_vectoriser(dev_texts)
    target = [SAMPLE_TEXTS[3]]
    s_iso = cosine_tfidf_scores_with_vec(vec, target)
    s_batch = cosine_tfidf_scores_with_vec(vec, target + DISTRACTORS)
    np.testing.assert_allclose(s_iso[0], s_batch[0], atol=1e-12,
                               err_msg="cosine TF-IDF score changed with batch size — likely batch-max normalisation present")


def test_weighted_lexical_batch_invariance():
    from v4.evaluation.data import load_gold_with_texts
    from v4.evaluation.splits import make_dev_test_split
    gold_df, y, texts = load_gold_with_texts(
        "/Users/akashx/msc-uk-analyst-skills/v3/manual_work/gold_standard_annotation_workbook_v2.xlsx",
        "/Users/akashx/msc-uk-analyst-skills/v3/manual_work/uk_analyst_corpus_v4_clean.csv",
    )
    split = make_dev_test_split(gold_df, seed=42)
    dev_texts = [texts[i] for i in np.where(split["is_dev"])[0]]
    vec = fit_tfidf_vectoriser(dev_texts)
    target = [SAMPLE_TEXTS[3]]
    s_iso = weighted_lexical_scores_with_vec(vec, target)
    s_batch = weighted_lexical_scores_with_vec(vec, target + DISTRACTORS)
    np.testing.assert_allclose(s_iso[0], s_batch[0], atol=1e-12,
                               err_msg="weighted lexical score changed with batch size — likely batch-max normalisation present")


def test_cosine_raw_scale_no_batch_max():
    """
    Ensure cosine scores are not artificially scaled to max=1 within batch.

    We fit on train, score two texts where one clearly contains more lexicon
    terms. The max-normalised version would make the weaker text's score
    depend on the stronger one; raw cosine must keep them independent.
    """
    train = [
        "Python SQL Tableau are data tools",
        "Stakeholder communication and presenting",
        "Cloud AWS Azure and data pipelines",
        "Reporting and excel spreadsheets statistical analysis",
        "Machine learning data cleaning etl data modelling ethics governance",
    ]
    vec = fit_tfidf_vectoriser(train)
    weak = ["We use Excel for reports"]
    strong = ["Python Python Python SQL SQL SQL Tableau Power BI machine learning tensorflow"]
    s_weak_alone = cosine_tfidf_scores_with_vec(vec, weak)
    s_both = cosine_tfidf_scores_with_vec(vec, weak + strong)
    # weak score should be unchanged when strong is added
    np.testing.assert_allclose(s_weak_alone[0], s_both[0], atol=1e-12,
                               err_msg="weak posting cosine score altered by presence of strong posting in batch")
    # Additionally, raw cosine values should be <=1 and not all 1.0 for weak
    assert s_weak_alone.max() <= 1.0 + 1e-9


def test_weighted_batch_invariance_multiple():
    """Batch invariance across multiple targets simultaneously."""
    from v4.evaluation.data import load_gold_with_texts
    from v4.evaluation.splits import make_dev_test_split
    gold_df, y, texts = load_gold_with_texts(
        "/Users/akashx/msc-uk-analyst-skills/v3/manual_work/gold_standard_annotation_workbook_v2.xlsx",
        "/Users/akashx/msc-uk-analyst-skills/v3/manual_work/uk_analyst_corpus_v4_clean.csv",
    )
    split = make_dev_test_split(gold_df, seed=42)
    dev_texts = [texts[i] for i in np.where(split["is_dev"])[0]]
    vec = fit_tfidf_vectoriser(dev_texts)
    targets = SAMPLE_TEXTS[2:4]
    s_all = weighted_lexical_scores_with_vec(vec, targets)
    for i, t in enumerate(targets):
        s_one = weighted_lexical_scores_with_vec(vec, [t])
        np.testing.assert_allclose(s_all[i], s_one[0], atol=1e-12,
                                   err_msg=f"weighted lexical batch dependence at index {i}")
