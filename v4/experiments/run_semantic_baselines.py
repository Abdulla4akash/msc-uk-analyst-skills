"""
Experiment 2 — Semantic baselines under same internal evaluation protocol.

Methods:
  A1 lexical (whole-word, no negatives) — anchor
  A2/A4 lexical (whole-word + negatives) — anchor (identical)
  A5 lexical IDF — anchor
  S1 supervised TF-IDF LR (nested C + thresholds)
  S2 frozen embedding similarity (nested thresholds, batch-invariant)
  S3 frozen zero-shot NLI (nested thresholds, MAX aggregation)

Identical outer folds seed 42, 3 folds, stratified role_family.
Primary = nested CV (300). Secondary = internal_tuning 100 / internal_holdout 200.
"""

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v4.config import CATEGORIES, TAXONOMY_VERSION, CATEGORY_LABELS
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_dev_test_split, make_cv_splits, RANDOM_SEED
from v4.evaluation.metrics import evaluate, accounting_report
from v4.evaluation.bootstrap import bootstrap_all
from v4.evaluation.nested import run_nested_cv_for_method as run_lexical_nested
from v4.evaluation.semantic_nested import (
    run_nested_supervised_tfidf,
    run_nested_embedding,
    run_nested_nli,
)
from v4.ablation.lexical_ablation import score_for_variant, any_hit_predictions
from v4.semantic.supervised_tfidf import get_outer_scores_and_thresholds
from v4.semantic.embedding_similarity import get_category_embeddings, embedding_scores
from v4.semantic.zero_shot_nli import nli_scores_for_texts
from v4.semantic.model_config import (
    S1_VECTORISER_CONFIG, S1_C_GRID, S1_CLASS_WEIGHT,
    S2_MODEL_ID, S2_REVISION, S2_MAX_TOKENS, S2_PARAMS, S2_LICENCE,
    S3_MODEL_ID, S3_REVISION, S3_MAX_TOKENS, S3_PARAMS, S3_LICENCE,
    NLI_HYPOTHESES,
)
from v4.methods.lexical_baseline import VECTORISER_CONFIG


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        return None


def get_package_versions():
    import sklearn, pandas as _pd, numpy as _np
    versions = {}
    for mod, name in [(sklearn, "scikit-learn"), (_pd, "pandas"), (_np, "numpy")]:
        try:
            versions[name] = mod.__version__
        except Exception:
            versions[name] = "unknown"
    for extra in ["torch", "transformers", "sentence_transformers", "openpyxl"]:
        try:
            m = __import__(extra)
            versions[extra] = m.__version__
        except Exception:
            versions[extra] = "not installed"
    versions["python"] = platform.python_version()
    try:
        import torch
        if torch.cuda.is_available():
            versions["device"] = "cuda"
        elif torch.backends.mps.is_available():
            versions["device"] = "mps"
        else:
            versions["device"] = "cpu"
    except Exception:
        versions["device"] = "unknown"
    return versions


def _paired_bootstrap_delta(y_true, pred_before, pred_after, n_bootstrap=10000, seed=42):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f_before = f1_score(y_true[idx], pred_before[idx], average="macro", zero_division=0)
        f_after = f1_score(y_true[idx], pred_after[idx], average="macro", zero_division=0)
        deltas[b] = float(f_after - f_before)
    point = float(f1_score(y_true, pred_after, average="macro", zero_division=0) - f1_score(y_true, pred_before, average="macro", zero_division=0))
    return {"delta_macro_f1": point, "ci_lower": float(np.percentile(deltas, 2.5)), "ci_upper": float(np.percentile(deltas, 97.5)), "n_bootstrap": n_bootstrap, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO_ROOT / "v3/manual_work/uk_analyst_corpus_v4_clean.csv"))
    ap.add_argument("--gold", default=str(REPO_ROOT / "v3/manual_work/gold_standard_annotation_workbook_v2.xlsx"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "v4/results/semantic"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--n_outer_splits", type=int, default=3)
    ap.add_argument("--skip_nli", action="store_true", help="skip S3 (slower)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    gold_df, y, texts = load_gold_with_texts(args.gold, args.corpus)
    n = len(gold_df)
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pkg_versions = get_package_versions()

    # Outer splits once
    outer_splits, outer_meta = make_cv_splits(gold_df, texts=texts, n_splits=args.n_outer_splits, n_repeats=1, seed=args.seed)
    holdout_split = make_dev_test_split(gold_df, dev_frac=1/3, seed=args.seed)
    is_internal_tuning = holdout_split["is_internal_tuning"]
    is_internal_holdout = holdout_split["is_internal_holdout"]
    y_holdout = y[is_internal_holdout]

    # Verify identical to ablation if ablation summary exists
    ablation_summary_path = REPO_ROOT / "v4/results/ablation/lexical_ablation_summary.json"
    outer_ids_match = None
    if ablation_summary_path.exists():
        try:
            import json as _j
            abl = _j.load(open(ablation_summary_path))
            abl_ids = [set(entry["validation_ids"]) for entry in abl["outer_splits"]["outer_fold_ids"]]
            new_ids = [set(gold_df.iloc[val_idx]["posting_id"]) for _, val_idx in outer_splits]
            outer_ids_match = all(a == b for a, b in zip(abl_ids, new_ids))
            assert outer_ids_match, f"Outer folds differ from ablation! {abl_ids} vs {new_ids}"
            print(f"Outer fold IDs IDENTICAL to ablation: {outer_ids_match}")
        except AssertionError as e:
            raise
        except Exception as e:
            print(f"Warning: ablation outer check failed: {e}")

    # --- Lexical anchors (nested, same outer_splits) ---
    print("Running lexical anchors A1/A2/A5...")
    lexical_results = {}
    # A1 fixed any-hit
    pred_A1 = np.zeros((n, len(CATEGORIES)), dtype=int)
    scores_A1 = np.zeros((n, len(CATEGORIES)), dtype=float)
    for _, val_idx in outer_splits:
        S = score_for_variant("A1", [texts[i] for i in val_idx])
        pred_A1[val_idx] = any_hit_predictions(S)
        scores_A1[val_idx] = S
    # A2 fixed any-hit
    pred_A2 = np.zeros((n, len(CATEGORIES)), dtype=int)
    scores_A2 = np.zeros((n, len(CATEGORIES)), dtype=float)
    for _, val_idx in outer_splits:
        S = score_for_variant("A2", [texts[i] for i in val_idx])
        pred_A2[val_idx] = any_hit_predictions(S)
        scores_A2[val_idx] = S
    # A5 via lexical nested (weighted)
    res_A5 = run_lexical_nested(gold_df, y, texts, outer_splits, method_name="weighted_lexical_tfidf", seed=args.seed)
    pred_A5 = res_A5["nested_predictions"]
    scores_A5 = res_A5["nested_scores"]
    # A4 is identical to A2 for this corpus (thresholds 0.02 -> same as any-hit), but compute anyway
    res_A4 = run_lexical_nested(gold_df, y, texts, outer_splits, method_name="unweighted_lexical", seed=args.seed)
    pred_A4 = res_A4["nested_predictions"]

    lexical_results["A1"] = (pred_A1, scores_A1)
    lexical_results["A2"] = (pred_A2, scores_A2)
    lexical_results["A5"] = (pred_A5, scores_A5)
    lexical_results["A4"] = (pred_A4, res_A4["nested_scores"])

    # --- S1 supervised TF-IDF LR ---
    print("Running S1 supervised TF-IDF LR (nested)...")
    t0 = time.time()
    res_S1 = run_nested_supervised_tfidf(gold_df, y, texts, outer_splits, seed=args.seed)
    t_S1 = time.time() - t0
    pred_S1 = res_S1["nested_predictions"]
    scores_S1 = res_S1["nested_scores"]

    # --- S2 embedding ---
    print("Running S2 embedding similarity (nested)...")
    t0 = time.time()
    res_S2 = run_nested_embedding(gold_df, y, texts, outer_splits, seed=args.seed)
    t_S2 = time.time() - t0
    pred_S2 = res_S2["nested_predictions"]
    scores_S2 = res_S2["nested_scores"]

    # --- S3 NLI ---
    if args.skip_nli:
        print("Skipping S3 per --skip_nli")
        pred_S3 = None
        scores_S3 = None
        res_S3 = None
        t_S3 = 0
    else:
        print("Running S3 zero-shot NLI (nested)... this may take several minutes")
        t0 = time.time()
        res_S3 = run_nested_nli(gold_df, y, texts, outer_splits, seed=args.seed)
        t_S3 = time.time() - t0
        pred_S3 = res_S3["nested_predictions"]
        scores_S3 = res_S3["nested_scores"]

    # Collect method maps
    method_preds = {
        "A1": pred_A1,
        "A2": pred_A2,
        "A4": pred_A4,
        "A5": pred_A5,
        "S1_TFIDF_LR": pred_S1,
        "S2_embedding": pred_S2,
    }
    method_scores = {
        "A1": scores_A1,
        "A2": scores_A2,
        "A4": res_A4["nested_scores"],
        "A5": scores_A5,
        "S1_TFIDF_LR": scores_S1,
        "S2_embedding": scores_S2,
    }
    runtimes = {
        "A1": 0.0, "A2": 0.0, "A4": 0.0, "A5": 0.0,
        "S1_TFIDF_LR": float(t_S1),
        "S2_embedding": float(t_S2),
    }
    if res_S3 is not None:
        method_preds["S3_NLI"] = pred_S3
        method_scores["S3_NLI"] = scores_S3
        runtimes["S3_NLI"] = float(t_S3)

    # --- Metrics per method (nested) ---
    from v4.evaluation.metrics import evaluate as eval_fn
    nested_reports = {}
    nested_accounting = {}
    for method, pred in method_preds.items():
        rep = eval_fn(y, pred)
        acc = accounting_report(y, pred)
        nested_reports[method] = rep
        nested_accounting[method] = acc

    # Bootstrap per method (supplementary)
    nested_bootstrap = {}
    for method, pred in method_preds.items():
        boot = bootstrap_all(y, pred, n_bootstrap=args.n_bootstrap, seed=args.seed)
        nested_bootstrap[method] = boot

    # --- Paired deltas vs lexical ---
    pairs = [
        ("A1", "S1_TFIDF_LR", "S1 - A1"),
        ("A2", "S1_TFIDF_LR", "S1 - A2"),
        ("A1", "S2_embedding", "S2 - A1"),
        ("A2", "S2_embedding", "S2 - A2"),
        ("S1_TFIDF_LR", "S2_embedding", "S2 - S1"),
    ]
    if "S3_NLI" in method_preds:
        pairs.extend([
            ("A1", "S3_NLI", "S3 - A1"),
            ("A2", "S3_NLI", "S3 - A2"),
            ("S1_TFIDF_LR", "S3_NLI", "S3 - S1"),
            ("S2_embedding", "S3_NLI", "S3 - S2"),
        ])
    paired = {}
    for before, after, label in pairs:
        pb = _paired_bootstrap_delta(y, method_preds[before], method_preds[after], n_bootstrap=args.n_bootstrap, seed=args.seed)
        # FP/FN deltas
        acc_before = nested_accounting[before]
        acc_after = nested_accounting[after]
        d_fp = int(acc_after["total_FP"] - acc_before["total_FP"])
        d_fn = int(acc_after["total_FN"] - acc_before["total_FN"])
        # wrong->correct etc.
        correct_before = (method_preds[before] == y)
        correct_after = (method_preds[after] == y)
        w2c = int(((~correct_before) & correct_after).sum())
        c2w = int((correct_before & (~correct_after)).sum())
        paired[label] = {
            "before": before, "after": after,
            "delta_macro_f1": pb["delta_macro_f1"], "ci_lower": pb["ci_lower"], "ci_upper": pb["ci_upper"],
            "delta_FP": d_fp, "delta_FN": d_fn,
            "wrong_to_correct": w2c, "correct_to_wrong": c2w, "net_corrected": w2c - c2w,
            "n_bootstrap": pb["n_bootstrap"], "seed": pb["seed"],
        }

    # --- Per-category comparison ---
    per_cat_rows = []
    for method in method_preds:
        rep = nested_reports[method]
        for cat in CATEGORIES:
            row = rep.loc[rep.category == cat].iloc[0]
            per_cat_rows.append({"method": method, "category": cat, "precision": float(row.precision), "recall": float(row.recall), "f1": float(row.f1), "support": int(row.support), "predicted": int(row.predicted)})

    # --- Seen vs lexically unseen positives ---
    # Use frozen final lexicon A1 whole-word matcher as analysis probe (no lexicon use in S2/S3 predictions)
    from v4.ablation.lexical_ablation import _wholeword_scores
    probe_scores = _wholeword_scores(texts, use_negative=False)
    # probe >0 means seen lexical expression for that posting×category
    # For each gold-positive cell (y==1), classify seen vs unseen
    seen_mask = (probe_scores > 0) & (y == 1)
    unseen_mask = (probe_scores == 0) & (y == 1)
    n_seen_pos = int(seen_mask.sum())
    n_unseen_pos = int(unseen_mask.sum())
    n_total_pos = int((y == 1).sum())
    seen_unseen_rows = []
    for method in method_preds:
        pred = method_preds[method]
        for cat_idx, cat in enumerate(CATEGORIES):
            gold_pos = (y[:, cat_idx] == 1)
            seen_pos_idx = np.where(gold_pos & (probe_scores[:, cat_idx] > 0))[0]
            unseen_pos_idx = np.where(gold_pos & (probe_scores[:, cat_idx] == 0))[0]
            # recall on seen/unseen
            if len(seen_pos_idx) > 0:
                recall_seen = float((pred[seen_pos_idx, cat_idx] == 1).mean())
                # F1 on seen subset? But subset is only positives, so precision not defined without negatives. Use recall.
            else:
                recall_seen = None
            if len(unseen_pos_idx) > 0:
                recall_unseen = float((pred[unseen_pos_idx, cat_idx] == 1).mean())
            else:
                recall_unseen = None
            seen_unseen_rows.append({
                "method": method, "category": cat,
                "n_gold_pos": int(gold_pos.sum()),
                "n_seen_pos": int(len(seen_pos_idx)),
                "n_unseen_pos": int(len(unseen_pos_idx)),
                "recall_seen": recall_seen,
                "recall_unseen": recall_unseen,
            })
    # Overall recall seen/unseen
    overall_seen_recall = {}
    for method in method_preds:
        pred = method_preds[method]
        overall_seen_recall[method] = {
            "recall_seen": float((pred[seen_mask] == 1).mean()) if n_seen_pos > 0 else None,
            "recall_unseen": float((pred[unseen_mask] == 1).mean()) if n_unseen_pos > 0 else None,
            "n_seen": n_seen_pos, "n_unseen": n_unseen_pos,
        }

    # --- Lexical failure subset (A1 wrong) ---
    a1_wrong_mask = (pred_A1 != y)  # (300,13) bool
    n_a1_wrong_cells = int(a1_wrong_mask.sum())
    # Split A1 wrong into FP vs FN
    a1_fp_mask = (pred_A1 == 1) & (y == 0) & a1_wrong_mask
    a1_fn_mask = (pred_A1 == 0) & (y == 1) & a1_wrong_mask
    failure_rows = []
    for method in ["S1_TFIDF_LR", "S2_embedding"] + (["S3_NLI"] if "S3_NLI" in method_preds else []):
        pred = method_preds[method]
        # Among A1 wrong cells, how many does method correct?
        correct_given_a1_wrong = int(((pred == y) & a1_wrong_mask).sum())
        wrong_given_a1_wrong = int(((pred != y) & a1_wrong_mask).sum())
        # Also new errors where A1 was correct but method wrong
        a1_correct_mask = ~a1_wrong_mask
        new_errors = int(((pred != y) & a1_correct_mask).sum())
        # Breakdown FP/FN recovery
        fp_recovered = int(((pred == y) & a1_fp_mask).sum())
        fn_recovered = int(((pred == y) & a1_fn_mask).sum())
        failure_rows.append({
            "method": method,
            "a1_wrong_cells": n_a1_wrong_cells,
            "a1_fp_cells": int(a1_fp_mask.sum()),
            "a1_fn_cells": int(a1_fn_mask.sum()),
            "recovered": correct_given_a1_wrong,
            "still_wrong": wrong_given_a1_wrong,
            "new_errors_where_a1_correct": new_errors,
            "fp_recovered": fp_recovered,
            "fn_recovered": fn_recovered,
            "net_corrected_vs_a1": int(((pred == y).sum() - (pred_A1 == y).sum())),
        })

    # --- Disagreements CSV (no full text) ---
    disag_rows = []
    for i in range(n):
        pid = str(gold_df.iloc[i]["posting_id"])
        for j, cat in enumerate(CATEGORIES):
            gold = int(y[i, j])
            # seen flag
            seen_flag = int(probe_scores[i, j] > 0) if gold == 1 else -1  # -1 for gold negatives (not applicable)
            # fold
            fold = -1
            for f, (_, val_idx) in enumerate(outer_splits):
                if i in val_idx:
                    fold = f
                    break
            disag_rows.append({
                "posting_id": pid,
                "category": cat,
                "gold": gold,
                "fold": fold,
                "seen_lexical": seen_flag,
                "A1_pred": int(pred_A1[i, j]),
                "A1_score": float(scores_A1[i, j]),
                "A2_pred": int(pred_A2[i, j]),
                "A2_score": float(scores_A2[i, j]),
                "A5_pred": int(pred_A5[i, j]),
                "A5_score": float(scores_A5[i, j]),
                "S1_pred": int(pred_S1[i, j]) if pred_S1 is not None else -1,
                "S1_score": float(scores_S1[i, j]) if scores_S1 is not None else -1,
                "S2_pred": int(pred_S2[i, j]),
                "S2_score": float(scores_S2[i, j]),
                "S3_pred": int(pred_S3[i, j]) if pred_S3 is not None else -1,
                "S3_score": float(scores_S3[i, j]) if scores_S3 is not None else -1,
            })

    # --- Internal holdout secondary (for S1/S2/S3) ---
    # For S1: fit vectoriser+LR on internal_tuning only, score internal_holdout (thresholds from tuning)
    # For S2/S3: thresholds tuned on internal_tuning scores, applied to holdout
    from v4.methods.lexical_baseline import tune_thresholds as tune_thr, apply_thresholds as apply_thr
    from v4.semantic.supervised_tfidf import build_vectoriser
    from sklearn.linear_model import LogisticRegression

    holdout_metrics = {}
    # S1 holdout
    tuning_texts = [texts[i] for i in np.where(is_internal_tuning)[0]]
    holdout_texts = [texts[i] for i in np.where(is_internal_holdout)[0]]
    y_tuning = y[is_internal_tuning]
    y_holdout_local = y[is_internal_holdout]
    # S1 holdout: need to select C via inner CV on tuning? But holdout is internal_holdout evaluation historically uses tuning set for fitting; for simplicity do same as lexical: fit vectoriser on tuning, LR C=1.0 balanced, tune thresholds on tuning
    # For nested consistency we will do inner CV on tuning to pick C, then fit on tuning, score holdout. But tuning set is only 100, inner CV may be small; instead pick C that was best in outer nested's most common? Simpler: try each C via 2-fold CV on tuning, pick best, fit on tuning.
    try:
        from sklearn.model_selection import StratifiedKFold
        roles_tuning = gold_df[is_internal_tuning]["role_family"].values
        skf_t = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        inner_holdout_splits = list(skf_t.split(np.zeros(len(tuning_texts)), roles_tuning))
    except Exception:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        inner_holdout_splits = list(kf.split(np.zeros(len(tuning_texts))))
    # This requires mapping; inner_holdout_splits indices are 0..100
    # Use supervised_tfidf helper
    from v4.semantic.supervised_tfidf import select_hyperparameters_inner_cv as sel_inner
    best_C_hold, best_thr_hold, best_macro_hold, _ = sel_inner(tuning_texts, y_tuning, inner_holdout_splits)
    prob_hold_S1, vec_hold, _ = get_outer_scores_and_thresholds(tuning_texts, y_tuning, holdout_texts, best_C_hold, best_thr_hold)
    pred_hold_S1 = apply_thr(prob_hold_S1, best_thr_hold)
    holdout_metrics["S1_TFIDF_LR"] = (eval_fn(y_holdout_local, pred_hold_S1), accounting_report(y_holdout_local, pred_hold_S1))
    # S2 holdout
    cat_embs_hold = get_category_embeddings() if 'cat_embs' not in locals() else cat_embs
    # Compute holdout scores: need S2 scores for tuning and holdout
    # Reuse embedding_scores but ensure no leakage: scores are frozen, thresholds from tuning only
    all_scores_S2_hold = embedding_scores(texts, cat_embs=cat_embs_hold)
    scores_tuning_S2 = all_scores_S2_hold[is_internal_tuning]
    scores_hold_S2 = all_scores_S2_hold[is_internal_holdout]
    thr_S2_hold = tune_thr(scores_tuning_S2, y_tuning)
    pred_hold_S2 = apply_thr(scores_hold_S2, thr_S2_hold)
    holdout_metrics["S2_embedding"] = (eval_fn(y_holdout_local, pred_hold_S2), accounting_report(y_holdout_local, pred_hold_S2))
    # S3 holdout
    if res_S3 is not None:
        all_scores_S3_hold = nli_scores_for_texts(texts)
        thr_S3_hold = tune_thr(all_scores_S3_hold[is_internal_tuning], y_tuning)
        pred_hold_S3 = apply_thr(all_scores_S3_hold[is_internal_holdout], thr_S3_hold)
        holdout_metrics["S3_NLI"] = (eval_fn(y_holdout_local, pred_hold_S3), accounting_report(y_holdout_local, pred_hold_S3))
    # Lexical holdout anchors (reuse A1/A2/A5 holdout from earlier ablation logic for consistency)
    S_A1_all = score_for_variant("A1", texts)
    pred_A1_all = any_hit_predictions(S_A1_all)
    holdout_metrics["A1"] = (eval_fn(y_holdout_local, pred_A1_all[is_internal_holdout]), accounting_report(y_holdout_local, pred_A1_all[is_internal_holdout]))
    S_A2_all = score_for_variant("A2", texts)
    pred_A2_all = any_hit_predictions(S_A2_all)
    holdout_metrics["A2"] = (eval_fn(y_holdout_local, pred_A2_all[is_internal_holdout]), accounting_report(y_holdout_local, pred_A2_all[is_internal_holdout]))
    # A5 holdout: need weighted scores
    from v4.methods.lexical_baseline import fit_tfidf_vectoriser, weighted_lexical_scores_with_vec
    vec_a5_hold = fit_tfidf_vectoriser(tuning_texts)
    S_a5_all = weighted_lexical_scores_with_vec(vec_a5_hold, texts)
    thr_a5 = tune_thr(S_a5_all[is_internal_tuning], y_tuning)
    pred_a5_all = apply_thr(S_a5_all, thr_a5)
    holdout_metrics["A5"] = (eval_fn(y_holdout_local, pred_a5_all[is_internal_holdout]), accounting_report(y_holdout_local, pred_a5_all[is_internal_holdout]))

    total_runtime = time.time() - start_time

    # --- Summary JSON ---
    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        "timestamp_utc": timestamp,
        "random_seed": args.seed,
        "n_examples": int(n),
        "vectoriser_config_lexical": VECTORISER_CONFIG,
        "vectoriser_config_S1": S1_VECTORISER_CONFIG,
        "S1_grid": {"C": S1_C_GRID, "class_weight": S1_CLASS_WEIGHT},
        "package_versions": pkg_versions,
        "model_selection": {
            "S2": {"model_id": S2_MODEL_ID, "revision": S2_REVISION, "licence": S2_LICENCE, "params": S2_PARAMS, "max_tokens": S2_MAX_TOKENS, "chunk_tokens": 256, "pooling": "mean", "category_source": "CATEGORY_LABELS"},
            "S3": {"model_id": S3_MODEL_ID, "revision": S3_REVISION, "licence": S3_LICENCE, "params": S3_PARAMS, "max_tokens": S3_MAX_TOKENS, "chunk_tokens": 400, "aggregation": "max", "hypotheses": NLI_HYPOTHESES},
        },
        "outer_splits": {
            "n_outer_splits": int(len(outer_splits)),
            "seed": int(args.seed),
            "outer_fold_ids": [
                {"fold": int(i), "validation_ids": gold_df.iloc[val_idx]["posting_id"].tolist(), "train_ids": gold_df.iloc[train_idx]["posting_id"].tolist()}
                for i, (train_idx, val_idx) in enumerate(outer_splits)
            ],
            "outer_ids_match_ablation": outer_ids_match,
        },
        "internal_holdout_split": {
            "n_internal_tuning": int(holdout_split["n_internal_tuning"]),
            "n_internal_holdout": int(holdout_split["n_internal_holdout"]),
            "internal_tuning_ids": holdout_split["internal_tuning_ids"],
            "internal_holdout_ids": holdout_split["internal_holdout_ids"],
        },
        "evaluation": "primary nested 3 outer ×2 inner, secondary internal_holdout 100/200; no external_locked_test",
        "data_flow": {
            "fit_scope": "S1: TF-IDF+LR fitted on inner_train/outer_train only; S2/S3 frozen models, thresholds fitted on inner/outer_train only",
            "threshold_scope": "per-category thresholds tuned on inner CV using outer_train only; outer validation labels never used",
            "external_locked_test": "RESERVED — does not exist",
            "no_lexicon_in_S2_S3": True,
        },
        "runtimes_seconds": runtimes,
        "total_runtime_seconds": float(total_runtime),
        "nested": {
            method: {
                "report": nested_reports[method].to_dict(orient="records"),
                "accounting": nested_accounting[method],
                "bootstrap": {
                    "macro_f1": {"point": float(nested_reports[method].loc[nested_reports[method].category=="MACRO AVG","f1"].iloc[0]), "lower": float(nested_bootstrap[method]["macro_f1"]["lower"]), "upper": float(nested_bootstrap[method]["macro_f1"]["upper"])},
                    "micro_f1": {"point": float(nested_reports[method].loc[nested_reports[method].category=="MICRO AVG","f1"].iloc[0]), "lower": float(nested_bootstrap[method]["micro_f1"]["lower"]), "upper": float(nested_bootstrap[method]["micro_f1"]["upper"])},
                },
                "per_fold_macro": [
                    {"fold": int(f), "macro_f1": float(eval_fn(y[val_idx], method_preds[method][val_idx]).loc[eval_fn(y[val_idx], method_preds[method][val_idx]).category=="MACRO AVG","f1"].iloc[0]), "n_validation": int(len(val_idx))}
                    for f, (_, val_idx) in enumerate(outer_splits)
                ],
                "outer_fold_info": (res_S1["outer_fold_info"] if method=="S1_TFIDF_LR" else res_S2["outer_fold_info"] if method=="S2_embedding" else res_S3["outer_fold_info"] if method=="S3_NLI" else res_A5["outer_fold_info"] if method=="A5" else None),
            } for method in method_preds
        },
        "internal_holdout": {
            method: {
                "report": holdout_metrics[method][0].to_dict(orient="records"),
                "accounting": holdout_metrics[method][1],
            } for method in holdout_metrics
        },
        "paired_deltas": paired,
        "per_category": per_cat_rows,
        "seen_unseen": {
            "n_total_positive_cells": int(n_total_pos),
            "n_seen_positive_cells": int(n_seen_pos),
            "n_unseen_positive_cells": int(n_unseen_pos),
            "per_method_recall": overall_seen_recall,
            "per_category_detail": seen_unseen_rows,
        },
        "lexical_failure_recovery": failure_rows,
        "regression_anchors": {
            "A1_expected": 0.9429, "A1_actual": float(nested_reports["A1"].loc[nested_reports["A1"].category=="MACRO AVG","f1"].iloc[0]),
            "A2_expected": 0.9418, "A2_actual": float(nested_reports["A2"].loc[nested_reports["A2"].category=="MACRO AVG","f1"].iloc[0]),
            "A5_expected": 0.9342, "A5_actual": float(nested_reports["A5"].loc[nested_reports["A5"].category=="MACRO AVG","f1"].iloc[0]),
        },
        "notes": {
            "bootstrap": "posting-level 10k seed 42 percentile 95 supplementary; fold variation primary",
            "lexical_failure": f"A1 wrong cells {n_a1_wrong_cells} (FP {int(a1_fp_mask.sum())}, FN {int(a1_fn_mask.sum())})",
        },
    }

    # Write outputs
    (outdir / "semantic_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # CSVs
    # nested results
    nested_rows = []
    for method in method_preds:
        rep = nested_reports[method]
        acc = nested_accounting[method]
        boot = nested_bootstrap[method]
        nested_rows.append({
            "method": method,
            "type": {"A1":"fixed lexical","A2":"contextual lexical","A4":"contextual lexical","A5":"IDF lexical","S1_TFIDF_LR":"supervised linear","S2_embedding":"frozen semantic","S3_NLI":"frozen semantic"}[method],
            "macro_f1": float(rep.loc[rep.category=="MACRO AVG","f1"].iloc[0]),
            "macro_ci_lower": float(boot["macro_f1"]["lower"]),
            "macro_ci_upper": float(boot["macro_f1"]["upper"]),
            "micro_f1": float(rep.loc[rep.category=="MICRO AVG","f1"].iloc[0]),
            "subset_accuracy": float(rep.loc[rep.category=="SUBSET ACCURACY","f1"].iloc[0]),
            "hamming_accuracy": float(rep.loc[rep.category=="HAMMING ACCURACY","f1"].iloc[0]),
            "TP": int(acc["total_TP"]), "FP": int(acc["total_FP"]), "FN": int(acc["total_FN"]), "TN": int(acc["total_TN"]),
            "n_ads_with_error": int(acc["n_ads_with_at_least_one_error"]),
            "avg_labels_pred": float(acc["avg_labels_per_ad_pred"]),
            "avg_labels_true": float(acc["avg_labels_per_ad_true"]),
            "runtime_seconds": float(runtimes.get(method, 0)),
        })
    pd.DataFrame(nested_rows).to_csv(outdir / "semantic_nested_results.csv", index=False)

    holdout_rows = []
    for method in holdout_metrics:
        rep, acc = holdout_metrics[method]
        holdout_rows.append({
            "method": method,
            "macro_f1": float(rep.loc[rep.category=="MACRO AVG","f1"].iloc[0]),
            "micro_f1": float(rep.loc[rep.category=="MICRO AVG","f1"].iloc[0]),
            "subset_accuracy": float(rep.loc[rep.category=="SUBSET ACCURACY","f1"].iloc[0]),
            "hamming_accuracy": float(rep.loc[rep.category=="HAMMING ACCURACY","f1"].iloc[0]),
        })
    pd.DataFrame(holdout_rows).to_csv(outdir / "semantic_internal_holdout_results.csv", index=False)

    pd.DataFrame(per_cat_rows).to_csv(outdir / "semantic_per_category.csv", index=False)

    # per fold
    per_fold_rows = []
    for method in method_preds:
        for f, (_, val_idx) in enumerate(outer_splits):
            rep = eval_fn(y[val_idx], method_preds[method][val_idx])
            per_fold_rows.append({"method": method, "fold": int(f), "n_validation": int(len(val_idx)), "macro_f1": float(rep.loc[rep.category=="MACRO AVG","f1"].iloc[0])})
    pd.DataFrame(per_fold_rows).to_csv(outdir / "semantic_per_fold.csv", index=False)

    # paired deltas
    pd.DataFrame([{"comparison": k, **v} for k, v in paired.items()]).to_csv(outdir / "semantic_paired_deltas.csv", index=False)
    # bootstrap already in summary

    pd.DataFrame(seen_unseen_rows).to_csv(outdir / "semantic_seen_unseen.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(outdir / "semantic_lexical_failure_recovery.csv", index=False)
    pd.DataFrame(disag_rows).to_csv(outdir / "semantic_disagreements.csv", index=False)

    # runtime
    runtime_rows = [{"method": m, "runtime_seconds": float(runtimes[m]), "avg_ms_per_posting": float(runtimes[m]/n*1000) if runtimes[m]>0 else 0} for m in runtimes]
    runtime_rows.append({"method": "TOTAL", "runtime_seconds": float(total_runtime), "avg_ms_per_posting": float(total_runtime/n*1000)})
    pd.DataFrame(runtime_rows).to_csv(outdir / "semantic_runtime.csv", index=False)

    # per-method per-category reports (nested)
    for method in method_preds:
        nested_reports[method].to_csv(outdir / f"semantic_{method}_nested_per_category.csv", index=False)
        holdout_metrics[method][0].to_csv(outdir / f"semantic_{method}_holdout_per_category.csv", index=False) if method in holdout_metrics else None

    print(f"Wrote semantic outputs to {outdir}")
    print(f"Regression anchors: A1 {nested_reports['A1'].loc[nested_reports['A1'].category=='MACRO AVG','f1'].iloc[0]:.4f} (exp 0.9429), A2 {nested_reports['A2'].loc[nested_reports['A2'].category=='MACRO AVG','f1'].iloc[0]:.4f} (exp 0.9418), A5 {nested_reports['A5'].loc[nested_reports['A5'].category=='MACRO AVG','f1'].iloc[0]:.4f} (exp 0.9342)")
    for row in nested_rows:
        print(f"{row['method']:15} macro {row['macro_f1']:.4f} [{row['macro_ci_lower']:.3f},{row['macro_ci_upper']:.3f}] micro {row['micro_f1']:.4f}")

    # Print paired deltas
    for k, v in paired.items():
        print(f"{k}: Δ {v['delta_macro_f1']:.4f} CI [{v['ci_lower']:.4f},{v['ci_upper']:.4f}] w2c {v['wrong_to_correct']} c2w {v['correct_to_wrong']} net {v['net_corrected']}")

    print(f"Seen positives {n_seen_pos}/{n_total_pos} ({n_seen_pos/n_total_pos:.1%}), unseen {n_unseen_pos} ({n_unseen_pos/n_total_pos:.1%})")
    for m, rec in overall_seen_recall.items():
        print(f"{m} recall seen {rec['recall_seen']:.3f} unseen {rec['recall_unseen']:.3f}" if rec['recall_seen'] else m)


if __name__ == "__main__":
    main()
