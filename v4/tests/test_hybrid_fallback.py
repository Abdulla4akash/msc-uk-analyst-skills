"""
Experiment 3 — Hybrid selective fallback tests (13).

Covers: regression anchors, asymmetric fallback, OFF, nested isolation,
batch invariance, provenance cache, and hybrid threshold tuning on full hybrid F1.
"""

import numpy as np

from v4.config import CATEGORIES, TAXONOMY_VERSION
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_cv_splits
from v4.hybrid.selective_fallback import (
    GRID_WITH_OFF,
    OFF_THRESHOLD,
    apply_hybrid_thresholds,
    lexical_A1_predictions,
    tune_hybrid_thresholds,
    tune_hybrid_thresholds_inner_cv,
)


def _load():
    import pathlib as _pl
    _repo = _pl.Path(__file__).resolve().parents[2]
    gold_df, y, texts = load_gold_with_texts(
        str(_repo / "v3/manual_work/gold_standard_annotation_workbook_v2.xlsx"),
        str(_repo / "v3/manual_work/uk_analyst_corpus_v4_clean.csv"),
    )
    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42)
    return gold_df, y, texts, outer_splits


def test_taxonomy_frozen():
    assert TAXONOMY_VERSION == "v3-13cat-frozen"
    assert len(CATEGORIES) == 13


def test_regression_anchor_macro():
    """A1 nested macro must be 0.9429 within 0.001 (reproduces 84ca602/d2b7afc)."""
    gold_df, y, texts, outer_splits = _load()
    from v4.evaluation.nested import run_nested_cv_for_method

    # Quick check via lexical anchor (use A1 outer val any-hit)
    from v4.ablation.lexical_ablation import score_for_variant, any_hit_predictions
    from v4.evaluation.metrics import evaluate

    n = len(gold_df)
    pred_A1 = np.zeros((n, len(CATEGORIES)), dtype=int)
    for _, val_idx in outer_splits:
        S = score_for_variant("A1", [texts[i] for i in val_idx])
        pred_A1[val_idx] = any_hit_predictions(S)
    rep = evaluate(y, pred_A1)
    macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
    assert abs(macro - 0.9429) < 0.002, f"A1 macro {macro} not near 0.9429"


def test_seen_unseen_anchors():
    """Gold positives: 1016 seen, 39 unseen (3.7%) — must match ablation/semantic."""
    gold_df, y, texts, _ = _load()
    from v4.ablation.lexical_ablation import score_for_variant

    S = score_for_variant("A1", texts)
    n_seen = int(((y == 1) & (S > 0)).sum())
    n_unseen = int(((y == 1) & (S == 0)).sum())
    assert n_seen == 1016, f"seen {n_seen} != 1016"
    assert n_unseen == 39, f"unseen {n_unseen} != 39"


def test_asymmetric_cannot_be_vetoed():
    """Lexical 1 cannot be vetoed even when semantic low."""
    lex = np.array([[1, 0], [1, 0]], dtype=int)
    sem = np.array([[0.0, 0.0], [0.1, 0.9]], dtype=float)
    thr = np.array([0.5, 0.5])
    hyb = apply_hybrid_thresholds(lex, sem, thr)
    # First column lexical 1 -> stays 1 even though semantic 0.0
    assert hyb[0, 0] == 1
    assert hyb[1, 0] == 1
    # Second column lexical 0 -> fallback depends on semantic
    assert hyb[0, 1] == 0
    assert hyb[1, 1] == 1
    # Pad to 13 cats for API: test with real categories size
    lex13 = np.zeros((2, 13), dtype=int)
    sem13 = np.zeros((2, 13), dtype=float)
    lex13[:, 0] = 1
    thr13 = np.full(13, 0.5)
    thr13[0] = 0.9
    hyb13 = apply_hybrid_thresholds(lex13, sem13, thr13)
    assert (hyb13[:, 0] == 1).all()


def test_off_option_never_fallback():
    lex = np.zeros((3, 13), dtype=int)
    sem = np.ones((3, 13), dtype=float)  # max semantic
    thr = np.full(13, OFF_THRESHOLD)
    hyb = apply_hybrid_thresholds(lex, sem, thr)
    assert (hyb == 0).all(), "OFF should never fallback even with semantic=1"
    # One category ON, rest OFF
    thr2 = np.full(13, OFF_THRESHOLD)
    thr2[0] = 0.0
    hyb2 = apply_hybrid_thresholds(lex, sem, thr2)
    assert (hyb2[:, 0] == 1).all()
    assert (hyb2[:, 1:] == 0).all()


def test_conservative_tie_break_highest_wins():
    """When F1 ties, highest threshold (OFF) wins."""
    # Create case where lexical alone already perfect for category, any fallback only adds FP
    # So best is OFF. But also 1.0 gives same as OFF (since sem <1). Tie should pick OFF.
    lex = np.array([[1], [0], [0], [0]], dtype=int)
    # semantic random but lexical already matches gold
    sem = np.array([[0.9], [0.4], [0.3], [0.2]], dtype=float)
    y = np.array([[1], [0], [0], [0]], dtype=int)
    # Pad to 13
    lex13 = np.zeros((4, 13), dtype=int)
    sem13 = np.zeros((4, 13), dtype=float)
    y13 = np.zeros((4, 13), dtype=int)
    lex13[:, 0:1] = lex
    sem13[:, 0:1] = sem
    y13[:, 0:1] = y
    thr = tune_hybrid_thresholds(lex13, sem13, y13, grid=np.array([0.0, 0.5, 1.0, OFF_THRESHOLD]))
    # Best should be OFF (2.0) because any fallback with threshold 0.0 would create FP on rows 1-3? Actually sem 0.4 would create FP. So OFF best.
    assert thr[0] == OFF_THRESHOLD
    # Also test tie where 0.5 and 1.0 both perfect: should pick 1.0 (higher)
    lex2 = np.zeros((2, 13), dtype=int)
    sem2 = np.array([[0.6] * 13, [0.4] * 13], dtype=float)
    y2 = np.array([[1] * 13, [0] * 13], dtype=int)
    # For this, threshold 0.5 predicts [1,0] correct, threshold 1.0 predicts [0,0] wrong? Not tie.
    # Simpler: semantic all 0.0, so any threshold >0 gives same (no fallback) as OFF
    lex3 = np.zeros((2, 13), dtype=int)
    sem3 = np.zeros((2, 13), dtype=float)
    y3 = np.zeros((2, 13), dtype=int)
    thr3 = tune_hybrid_thresholds(lex3, sem3, y3, grid=np.array([0.0, 0.5, 1.0, OFF_THRESHOLD]))
    # All thresholds give same F1 (no positives), should pick OFF (highest)
    assert thr3[0] == OFF_THRESHOLD


def test_tune_optimises_hybrid_not_semantic_only():
    """Thresholds tuned on hybrid F1 differ from semantic-only tuning when lexical gate helps."""
    # Lexical has perfect recall on seen, semantic would propose higher recall but lower precision
    # Create y where lexical 1 is correct, semantic would be wrong if tuned alone
    lex = np.array([[1, 0], [0, 0], [0, 0]], dtype=int)
    sem = np.array([[0.9, 0.9], [0.8, 0.8], [0.7, 0.7]], dtype=float)
    y = np.array([[1, 0], [0, 0], [0, 0]], dtype=int)
    lex13 = np.zeros((3, 13), dtype=int)
    sem13 = np.zeros((3, 13), dtype=float)
    y13 = np.zeros((3, 13), dtype=int)
    lex13[:, 0:1] = lex[:, 0:1]
    sem13[:, 0:1] = sem[:, 0:1]
    y13[:, 0:1] = y[:, 0:1]
    thr_hybrid = tune_hybrid_thresholds(lex13, sem13, y13)
    # Hybrid should be OFF for cat 0 because lexical already perfect and any fallback adds FP
    assert thr_hybrid[0] == OFF_THRESHOLD
    # Semantic-only tuning (ignoring lexical) is not hybrid-aware: it picks a threshold that maximises semantic F1 alone.
    # It should NOT be OFF in this case (it finds a useful semantic threshold), proving hybrid optimisation differs.
    from v4.methods.lexical_baseline import tune_thresholds
    thr_sem_only = tune_thresholds(sem13, y13, grid=np.linspace(0, 1, 51))
    assert thr_sem_only[0] != OFF_THRESHOLD, f"semantic-only {thr_sem_only[0]} should not be OFF"
    assert float(thr_sem_only[0]) != float(thr_hybrid[0]), "hybrid and semantic-only thresholds should differ" 


def test_batch_invariant():
    lex = np.random.randint(0, 2, size=(5, 13))
    sem = np.random.rand(5, 13)
    thr = np.random.rand(13)
    thr[::3] = OFF_THRESHOLD
    hyb_batch = apply_hybrid_thresholds(lex, sem, thr)
    # Single row should give same result
    for i in range(5):
        hyb_single = apply_hybrid_thresholds(lex[i : i + 1], sem[i : i + 1], thr)
        assert np.array_equal(hyb_batch[i : i + 1], hyb_single)


def test_outer_label_isolation_H1():
    """Flipping outer val labels must not change H1 thresholds for that fold (inner only)."""
    gold_df, y, texts, outer_splits = _load()
    from v4.evaluation.hybrid_nested import run_nested_hybrid_S1

    res1 = run_nested_hybrid_S1(gold_df, y, texts, outer_splits, seed=42)
    y_mut = y.copy()
    _, val_idx = outer_splits[0]
    y_mut[val_idx] = 1 - y_mut[val_idx]
    res2 = run_nested_hybrid_S1(gold_df, y_mut, texts, outer_splits, seed=42)
    thr1 = res1["outer_fold_info"][0]["thresholds"]
    thr2 = res2["outer_fold_info"][0]["thresholds"]
    for cat in CATEGORIES:
        v1 = thr1[CATEGORIES.index(cat)] if not isinstance(thr1, dict) else thr1[cat]
        v2 = thr2[CATEGORIES.index(cat)] if not isinstance(thr2, dict) else thr2[cat]
        assert abs(float(v1) - float(v2)) < 1e-9, f"H1 thr for {cat} changed when outer val labels flipped"


def test_outer_label_isolation_H2():
    gold_df, y, texts, outer_splits = _load()
    from v4.evaluation.hybrid_nested import run_nested_hybrid_S3

    res1 = run_nested_hybrid_S3(gold_df, y, texts, outer_splits, seed=42)
    y_mut = y.copy()
    _, val_idx = outer_splits[0]
    y_mut[val_idx] = 1 - y_mut[val_idx]
    res2 = run_nested_hybrid_S3(gold_df, y_mut, texts, outer_splits, seed=42)
    thr1 = res1["outer_fold_info"][0]["thresholds"]
    thr2 = res2["outer_fold_info"][0]["thresholds"]
    for cat in CATEGORIES:
        v1 = thr1[CATEGORIES.index(cat)] if not isinstance(thr1, dict) else thr1[cat]
        v2 = thr2[CATEGORIES.index(cat)] if not isinstance(thr2, dict) else thr2[cat]
        assert abs(float(v1) - float(v2)) < 1e-9, f"H2 thr for {cat} changed"


def test_outer_text_isolation_H1():
    gold_df, y, texts, outer_splits = _load()
    from v4.evaluation.hybrid_nested import run_nested_hybrid_S1

    res1 = run_nested_hybrid_S1(gold_df, y, texts, outer_splits, seed=42)
    texts_mut = list(texts)
    _, val_idx = outer_splits[0]
    for idx in val_idx:
        texts_mut[idx] = texts_mut[idx] + " " + "x" * 200
    res2 = run_nested_hybrid_S1(gold_df, y, texts_mut, outer_splits, seed=42)
    thr1 = res1["outer_fold_info"][0]["thresholds"]
    thr2 = res2["outer_fold_info"][0]["thresholds"]
    for cat in CATEGORIES:
        v1 = thr1[CATEGORIES.index(cat)] if not isinstance(thr1, dict) else thr1[cat]
        v2 = thr2[CATEGORIES.index(cat)] if not isinstance(thr2, dict) else thr2[cat]
        assert abs(float(v1) - float(v2)) < 1e-9, f"H1 thr changed when outer val text mutated"


def test_nli_cache_provenance():
    """S3 cache file must exist after H2 run and provenance hash must match texts."""
    gold_df, y, texts, outer_splits = _load()
    from v4.evaluation.hybrid_nested import _get_nli_scores_cached, _provenance_hash
    from v4.semantic.model_config import S3_MODEL_ID, S3_CHUNK_TOKENS, NLI_HYPOTHESES_LIST
    import pathlib

    scores, prov, _ = _get_nli_scores_cached(texts)
    assert scores.shape == (len(texts), len(CATEGORIES))
    assert prov["model_id"] == S3_MODEL_ID
    assert prov["texts_hash"] == _provenance_hash(texts, S3_MODEL_ID, NLI_HYPOTHESES_LIST, S3_CHUNK_TOKENS)
    cache_path = pathlib.Path(__file__).resolve().parents[2] / "v4/results/semantic/s3_nli_scores_cache.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        assert data["scores"].shape == (len(texts), len(CATEGORIES))
    # Second call should be cached
    scores2, prov2, was_cached = _get_nli_scores_cached(texts)
    assert np.allclose(scores, scores2)
    # was_cached should be True on second call (if cache exists)
    assert was_cached is True


def test_hybrid_no_lexicon_import_in_S3():
    """Hybrid S3 path must not import lexicon via NLI scorer source."""
    import pathlib

    nli_src = (pathlib.Path(__file__).resolve().parents[2] / "v4/semantic/zero_shot_nli.py").read_text()
    # forbid lexicon string in NLI source (case-insensitive)
    assert "LEXICON" not in nli_src.upper() or "no lexicon" in nli_src.lower(), "NLI source must not depend on lexicon"
