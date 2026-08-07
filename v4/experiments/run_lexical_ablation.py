"""
Controlled lexical ablation runner — CASE B (seed lexicon not recoverable).

A0: naive substring, final lexicon, any-hit (>0)
A1: whole-word, final lexicon, any-hit
A2: whole-word + frozen NEGATIVE_PATTERNS, final lexicon, any-hit  (A3 omitted)
A4: A2 + nested per-category thresholds (unweighted)  → reproduces unweighted_lexical
A5: A4 + inductive IDF weighting + nested thresholds  → reproduces weighted_lexical_tfidf

Identical outer splits (seed 42, stratified on role_family) for every variant → paired comparisons.
Primary = nested CV (3 outer folds, inner 2-fold per outer_train, genuinely nested).
Secondary = internal_tuning (100) / internal_holdout (200).

Audits, deltas, paired bootstrap, error transitions are produced alongside metrics.
"""

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v4.config import CATEGORIES, TAXONOMY_VERSION, LEXICONS, NEGATIVE_PATTERNS
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_dev_test_split, make_cv_splits, RANDOM_SEED
from v4.evaluation.metrics import evaluate, accounting_report, format_report
from v4.evaluation.bootstrap import bootstrap_all, bootstrap_ci
from v4.evaluation.nested import run_nested_cv_for_method
from v4.ablation.lexical_ablation import (
    ABLATION_VARIANTS,
    ABLATION_DEFINITIONS,
    score_for_variant,
    any_hit_predictions,
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
    try:
        import openpyxl
        versions["openpyxl"] = openpyxl.__version__
    except Exception:
        versions["openpyxl"] = "unknown"
    versions["python"] = platform.python_version()
    return versions


def _evaluate_variant(y_true, y_pred):
    return evaluate(y_true, y_pred), accounting_report(y_true, y_pred)


def _paired_bootstrap_delta(y_true, pred_before, pred_after, n_bootstrap=10000, seed=42):
    """Paired bootstrap of Δ macro-F1 (after - before) using same resampled indices."""
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
    lower = float(np.percentile(deltas, 2.5))
    upper = float(np.percentile(deltas, 97.5))
    return {"delta_macro_f1": point, "ci_lower": lower, "ci_upper": upper, "n_bootstrap": n_bootstrap, "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO_ROOT / "v3/manual_work/uk_analyst_corpus_v4_clean.csv"))
    ap.add_argument("--gold", default=str(REPO_ROOT / "v3/manual_work/gold_standard_annotation_workbook_v2.xlsx"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "v4/results/ablation"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--n_outer_splits", type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gold_df, y, texts = load_gold_with_texts(args.gold, args.corpus)
    n = len(gold_df)
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Outer splits generated ONCE and reused for every variant (paired)
    outer_splits, outer_meta = make_cv_splits(gold_df, texts=texts, n_splits=args.n_outer_splits, n_repeats=1, seed=args.seed)
    # Internal holdout split (secondary, same seed)
    holdout_split = make_dev_test_split(gold_df, dev_frac=1/3, seed=args.seed)
    is_internal_tuning = holdout_split["is_internal_tuning"]
    is_internal_holdout = holdout_split["is_internal_holdout"]

    # ---------- Nested predictions per variant ----------
    # A0–A2: fixed rule, no fitting — just score each outer validation directly
    nested_preds = {}
    nested_scores = {}
    for variant in ["A0", "A1", "A2"]:
        pred = np.zeros((n, len(CATEGORIES)), dtype=int)
        scores = np.zeros((n, len(CATEGORIES)), dtype=float)
        for _, val_idx in outer_splits:
            val_texts = [texts[i] for i in val_idx]
            S = score_for_variant(variant, val_texts)
            pred[val_idx] = any_hit_predictions(S)
            scores[val_idx] = S
        nested_preds[variant] = pred
        nested_scores[variant] = scores

    # A3 omitted
    # A4, A5 via genuinely nested CV (thresholds from inner CV only)
    for variant, method in [("A4", "unweighted_lexical"), ("A5", "weighted_lexical_tfidf")]:
        res = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=args.seed)
        nested_preds[variant] = res["nested_predictions"]
        nested_scores[variant] = res["nested_scores"]
        # Store outer_fold_info for audit
        # Attach to a holder for later
        if variant == "A4":
            a4_outer_info = res["outer_fold_info"]
        else:
            a5_outer_info = res["outer_fold_info"]

    # ---------- Internal holdout predictions per variant ----------
    holdout_preds = {}
    for variant in ["A0", "A1", "A2"]:
        S_all = score_for_variant(variant, texts)
        holdout_preds[variant] = any_hit_predictions(S_all)
    # A4/A5 holdout: use same logic as run_lexical_baseline's internal_holdout (fit on internal_tuning)
    from v4.methods.lexical_baseline import tune_thresholds, apply_thresholds, unweighted_lexical_scores, fit_tfidf_vectoriser, weighted_lexical_scores_with_vec
    # A4 holdout
    S_a4_all = unweighted_lexical_scores(texts)
    thr_a4 = tune_thresholds(S_a4_all[is_internal_tuning], y[is_internal_tuning])
    holdout_preds["A4"] = apply_thresholds(S_a4_all, thr_a4)
    # A5 holdout
    tuning_texts = [texts[i] for i in np.where(is_internal_tuning)[0]]
    vec_a5 = fit_tfidf_vectoriser(tuning_texts)
    S_a5_all = weighted_lexical_scores_with_vec(vec_a5, texts)
    thr_a5 = tune_thresholds(S_a5_all[is_internal_tuning], y[is_internal_tuning])
    holdout_preds["A5"] = apply_thresholds(S_a5_all, thr_a5)

    # ---------- Metrics ----------
    # Nested (primary)
    nested_reports = {}
    nested_accounting = {}
    for variant in ABLATION_VARIANTS:
        rep, acc = _evaluate_variant(y, nested_preds[variant])
        nested_reports[variant] = rep
        nested_accounting[variant] = acc

    # Holdout (secondary)
    holdout_reports = {}
    holdout_accounting = {}
    y_holdout = y[is_internal_holdout]
    for variant in ABLATION_VARIANTS:
        pred_holdout = holdout_preds[variant][is_internal_holdout]
        rep, acc = _evaluate_variant(y_holdout, pred_holdout)
        holdout_reports[variant] = rep
        holdout_accounting[variant] = acc

    # ---------- Paired deltas (nested, primary) ----------
    transitions = [("A0", "A1"), ("A1", "A2"), ("A2", "A4"), ("A4", "A5")]
    # Also keep A2->A3 as NOT IDENTIFIABLE placeholder
    incremental_rows = []
    paired_bootstrap = {}
    for before, after in transitions:
        rep_before = nested_reports[before]
        rep_after = nested_reports[after]
        macro_before = float(rep_before.loc[rep_before.category == "MACRO AVG", "f1"].iloc[0])
        macro_after = float(rep_after.loc[rep_after.category == "MACRO AVG", "f1"].iloc[0])
        delta = macro_after - macro_before
        acc_before = nested_accounting[before]
        acc_after = nested_accounting[after]
        d_fp = int(acc_after["total_FP"] - acc_before["total_FP"])
        d_fn = int(acc_after["total_FN"] - acc_before["total_FN"])
        # Component name
        comp_map = {("A0","A1"): "whole-word matching", ("A1","A2"): "negative_pattern_suppression",
                    ("A2","A4"): "threshold_tuning (unweighted)", ("A4","A5"): "IDF_weighting"}
        comp = comp_map.get((before, after), f"{before}->{after}")
        # Paired bootstrap CI for delta
        pb = _paired_bootstrap_delta(y, nested_preds[before], nested_preds[after], n_bootstrap=args.n_bootstrap, seed=args.seed)
        incremental_rows.append({
            "transition": f"{before}->{after}",
            "component": comp,
            "macro_before": macro_before,
            "macro_after": macro_after,
            "delta_macro": delta,
            "delta_macro_ci_lower": pb["ci_lower"],
            "delta_macro_ci_upper": pb["ci_upper"],
            "delta_FP": d_fp,
            "delta_FN": d_fn,
        })
        paired_bootstrap[f"{before}_to_{after}"] = {
            "component": comp,
            "delta_macro_f1": pb["delta_macro_f1"],
            "ci_lower": pb["ci_lower"],
            "ci_upper": pb["ci_upper"],
            "n_bootstrap": pb["n_bootstrap"],
            "seed": pb["seed"],
        }
    # Insert A2->A3 as NOT IDENTIFIABLE
    incremental_rows.insert(2, {
        "transition": "A2->A3",
        "component": "lexicon_expansion",
        "macro_before": None,
        "macro_after": None,
        "delta_macro": None,
        "delta_macro_ci_lower": None,
        "delta_macro_ci_upper": None,
        "delta_FP": None,
        "delta_FN": None,
        "note": "NOT IDENTIFIABLE FROM AVAILABLE PROVENANCE",
    })

    # ---------- Per-category deltas (nested) ----------
    per_cat_rows = []
    for before, after in [("A0","A1"),("A1","A2"),("A2","A4"),("A4","A5")]:
        for i, cat in enumerate(CATEGORIES):
            rb = nested_reports[before]
            ra = nested_reports[after]
            f_before = float(rb.loc[rb.category == cat, "f1"].iloc[0])
            f_after = float(ra.loc[ra.category == cat, "f1"].iloc[0])
            per_cat_rows.append({"transition": f"{before}->{after}", "category": cat, "f1_before": f_before, "f1_after": f_after, "delta_f1": f_after - f_before})

    # ---------- Audits ----------
    # Boundary audit A0->A1
    boundary_rows = []
    # For each category, compare A0 vs A1 predictions where they differ
    # Also count suppressed substring that was not whole-word
    for i, cat in enumerate(CATEGORIES):
        a0 = nested_preds["A0"][:, i]
        a1 = nested_preds["A1"][:, i]
        # FP removed: A0=1,A1=0 and y=0
        fp_removed = int(((a0 == 1) & (a1 == 0) & (y[:, i] == 0)).sum())
        tp_lost = int(((a0 == 1) & (a1 == 0) & (y[:, i] == 1)).sum())
        fp_added = int(((a0 == 0) & (a1 == 1) & (y[:, i] == 0)).sum())
        tp_gained = int(((a0 == 0) & (a1 == 1) & (y[:, i] == 1)).sum())
        # Representative match: count of postings where substring matched but whole-word did not (i.e. A0=1,A1=0)
        n_changed = int(((a0 != a1)).sum())
        boundary_rows.append({"category": cat, "n_changed": n_changed, "fp_removed": fp_removed, "tp_lost": tp_lost, "fp_added": fp_added, "tp_gained": tp_gained})

    # Negative pattern audit A1->A2
    # For each category with patterns, count suppressed matches
    neg_audit_rows = []
    neg_detail_rows = []  # posting_id, category, suppressed, gold label
    for i, cat in enumerate(CATEGORIES):
        patterns = NEGATIVE_PATTERNS.get(cat, [])
        if not patterns:
            neg_audit_rows.append({"category": cat, "suppressed_matches": 0, "correct_fp_removals": 0, "incorrect_tp_removals": 0, "net_errors_reduced": 0})
            continue
        a1 = nested_preds["A1"][:, i]
        a2 = nested_preds["A2"][:, i]
        suppressed_mask = (a1 == 1) & (a2 == 0)
        suppressed_n = int(suppressed_mask.sum())
        correct_fp = int(((suppressed_mask) & (y[:, i] == 0)).sum())
        incorrect_tp = int(((suppressed_mask) & (y[:, i] == 1)).sum())
        # Also check new FPs introduced by negatives? Should be 0 since negatives only suppress
        net = correct_fp - incorrect_tp  # positive means net error reduction
        neg_audit_rows.append({"category": cat, "suppressed_matches": suppressed_n, "correct_fp_removals": correct_fp, "incorrect_tp_removals": incorrect_tp, "net_errors_reduced": net})
        # Detail per posting
        for idx in np.where(suppressed_mask)[0]:
            neg_detail_rows.append({"posting_id": str(gold_df.iloc[idx]["posting_id"]), "category": cat, "gold": int(y[idx, i]), "suppressed": 1})

    # Threshold audit: A2 (any-hit) -> A4 (nested thresholds) on nested predictions
    # For A4 we have per-outer-fold thresholds; report them and net FP/FN change
    threshold_rows = []
    # Use nested A2->A4 overall, plus per-category thresholds from outer folds
    for i, cat in enumerate(CATEGORIES):
        a2 = nested_preds["A2"][:, i]
        a4 = nested_preds["A4"][:, i]
        fp_removed = int(((a2 == 1) & (a4 == 0) & (y[:, i] == 0)).sum())
        fn_added = int(((a2 == 1) & (a4 == 0) & (y[:, i] == 1)).sum())
        fp_added = int(((a2 == 0) & (a4 == 1) & (y[:, i] == 0)).sum())
        tp_recovered = int(((a2 == 0) & (a4 == 1) & (y[:, i] == 1)).sum())
        # thresholds per outer fold for this category (from A4 outer info)
        thresh_per_fold = [info["thresholds"][cat] for info in a4_outer_info]
        threshold_rows.append({
            "category": cat, "threshold_outer0": thresh_per_fold[0], "threshold_outer1": thresh_per_fold[1], "threshold_outer2": thresh_per_fold[2],
            "fp_removed": fp_removed, "fn_added": fn_added, "fp_added": fp_added, "tp_recovered": tp_recovered,
        })

    # IDF audit: A4->A5
    idf_rows = []
    for i, cat in enumerate(CATEGORIES):
        a4 = nested_preds["A4"][:, i]
        a5 = nested_preds["A5"][:, i]
        rb = nested_reports["A4"]
        ra = nested_reports["A5"]
        f_before = float(rb.loc[rb.category == cat, "f1"].iloc[0])
        f_after = float(ra.loc[ra.category == cat, "f1"].iloc[0])
        # Count predictions changed
        changed = int((a4 != a5).sum())
        # FP/FN delta per category
        fp_before = int(((a4 == 1) & (y[:, i] == 0)).sum())
        fp_after = int(((a5 == 1) & (y[:, i] == 0)).sum())
        fn_before = int(((a4 == 0) & (y[:, i] == 1)).sum())
        fn_after = int(((a5 == 0) & (y[:, i] == 1)).sum())
        idf_rows.append({
            "category": cat, "f1_before": f_before, "f1_after": f_after, "delta_f1": f_after - f_before,
            "predictions_changed": changed, "fp_before": fp_before, "fp_after": fp_after, "fn_before": fn_before, "fn_after": fn_after,
            "delta_fp": fp_after - fp_before, "delta_fn": fn_after - fn_before,
        })

    # Error transitions: for each adjacent pair, classify posting×category cells
    transition_error_rows = []
    for before, after in [("A0","A1"),("A1","A2"),("A2","A4"),("A4","A5")]:
        pb = nested_preds[before]
        pa = nested_preds[after]
        # per cell
        correct_before = (pb == y)
        correct_after = (pa == y)
        wrong_to_correct = int(((~correct_before) & correct_after).sum())
        correct_to_wrong = int((correct_before & (~correct_after)).sum())
        unchanged_correct = int((correct_before & correct_after).sum())
        unchanged_wrong = int(((~correct_before) & (~correct_after)).sum())
        net = wrong_to_correct - correct_to_wrong
        transition_error_rows.append({
            "transition": f"{before}->{after}",
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "net_corrected": net,
        })

    # ---------- Summary JSON ----------
    # Per-fold macro for each variant (nested)
    per_fold_macro = []
    for variant in ABLATION_VARIANTS:
        for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
            rep = evaluate(y[val_idx], nested_preds[variant][val_idx])
            macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
            per_fold_macro.append({"variant": variant, "fold": int(fold_idx), "n_validation": int(len(val_idx)), "macro_f1": macro})

    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "lexicon_provenance": "v4/LEXICON_PROVENANCE.md — authentic seed NOT RECOVERABLE; A3 omitted (CASE B)",
        "git_commit": git_commit,
        "timestamp_utc": timestamp,
        "random_seed": args.seed,
        "n_examples": int(n),
        "vectoriser_config": VECTORISER_CONFIG,
        "package_versions": get_package_versions(),
        "ablation_definitions": ABLATION_DEFINITIONS,
        "outer_splits": {
            "n_outer_splits": int(len(outer_splits)),
            "seed": int(args.seed),
            "outer_fold_ids": [
                {"fold": int(i), "validation_ids": gold_df.iloc[val_idx]["posting_id"].tolist(), "train_ids": gold_df.iloc[train_idx]["posting_id"].tolist()}
                for i, (train_idx, val_idx) in enumerate(outer_splits)
            ],
            "outer_meta": {k: v for k, v in outer_meta.items() if k != "group_ids"},
        },
        "internal_holdout_split": {
            "method": "stratified internal_tuning/internal_holdout on role_family",
            "seed": int(args.seed),
            "n_internal_tuning": int(holdout_split["n_internal_tuning"]),
            "n_internal_holdout": int(holdout_split["n_internal_holdout"]),
            "internal_tuning_ids": holdout_split["internal_tuning_ids"],
            "internal_holdout_ids": holdout_split["internal_holdout_ids"],
        },
        "evaluation": "primary = nested CV (3 outer × 2 inner, genuinely nested, same outer splits for all variants); secondary = internal_holdout (100/200)",
        "data_flow": {
            "fit_scope": "A0–A3: no fitting; A4: no IDF; A5: TF-IDF fitted on outer_train/inner_train only",
            "threshold_scope": "A0–A3: fixed any-hit (>0); A4/A5: nested per-category thresholds from inner CV only",
            "external_locked_test": "RESERVED — does not exist",
        },
        "nested": {
            variant: {
                "report": rep.to_dict(orient="records"),
                "accounting": acc,
                "per_fold_macro_f1": [r for r in per_fold_macro if r["variant"] == variant],
            } for variant, rep, acc in [(v, nested_reports[v], nested_accounting[v]) for v in ABLATION_VARIANTS]
        },
        "internal_holdout": {
            variant: {
                "report": rep.to_dict(orient="records"),
                "accounting": acc,
            } for variant, rep, acc in [(v, holdout_reports[v], holdout_accounting[v]) for v in ABLATION_VARIANTS]
        },
        "incremental_deltas": incremental_rows,
        "paired_bootstrap": paired_bootstrap,
        "per_category_deltas": per_cat_rows,
        "boundary_audit": boundary_rows,
        "negative_pattern_audit": neg_audit_rows,
        "threshold_audit": threshold_rows,
        "idf_audit": idf_rows,
        "error_transitions": transition_error_rows,
        "regression_anchors": {
            "A4_expected_nested_macro": 0.942,
            "A5_expected_nested_macro": 0.934,
            "A4_actual_nested_macro": float(nested_reports["A4"].loc[nested_reports["A4"].category == "MACRO AVG", "f1"].iloc[0]),
            "A5_actual_nested_macro": float(nested_reports["A5"].loc[nested_reports["A5"].category == "MACRO AVG", "f1"].iloc[0]),
        },
        "notes": {
            "A3": "OMITTED — lexicon expansion not identifiable (see LEXICON_PROVENANCE.md)",
            "bootstrap": "posting-level paired bootstrap, 10k resamples, seed 42, percentile 95% — supplementary; fold variation primary",
        },
    }

    # Write summary
    (outdir / "lexical_ablation_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # CSV outputs
    # Main nested results table
    nested_table = []
    for variant in ABLATION_VARIANTS:
        rep = nested_reports[variant]
        acc = nested_accounting[variant]
        macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
        micro = float(rep.loc[rep.category == "MICRO AVG", "f1"].iloc[0])
        subset = float(rep.loc[rep.category == "SUBSET ACCURACY", "f1"].iloc[0])
        hamming = float(rep.loc[rep.category == "HAMMING ACCURACY", "f1"].iloc[0])
        # bootstrap CI for nested (supplementary)
        from v4.evaluation.bootstrap import bootstrap_all
        boot = bootstrap_all(y, nested_preds[variant], n_bootstrap=args.n_bootstrap, seed=args.seed)
        nested_table.append({
            "variant": variant,
            "macro_f1": macro, "macro_ci_lower": boot["macro_f1"]["lower"], "macro_ci_upper": boot["macro_f1"]["upper"],
            "micro_f1": micro, "subset_accuracy": subset, "hamming_accuracy": hamming,
            "TP": acc["total_TP"], "FP": acc["total_FP"], "FN": acc["total_FN"], "TN": acc["total_TN"],
            "n_ads_with_error": acc["n_ads_with_at_least_one_error"],
            "avg_labels_pred": acc["avg_labels_per_ad_pred"],
            "avg_labels_true": acc["avg_labels_per_ad_true"],
        })
    pd.DataFrame(nested_table).to_csv(outdir / "lexical_ablation_nested_results.csv", index=False)

    # Internal holdout table
    holdout_table = []
    for variant in ABLATION_VARIANTS:
        rep = holdout_reports[variant]
        acc = holdout_accounting[variant]
        holdout_table.append({
            "variant": variant,
            "macro_f1": float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0]),
            "micro_f1": float(rep.loc[rep.category == "MICRO AVG", "f1"].iloc[0]),
            "subset_accuracy": float(rep.loc[rep.category == "SUBSET ACCURACY", "f1"].iloc[0]),
            "hamming_accuracy": float(rep.loc[rep.category == "HAMMING ACCURACY", "f1"].iloc[0]),
            "TP": acc["total_TP"], "FP": acc["total_FP"], "FN": acc["total_FN"], "TN": acc["total_TN"],
        })
    pd.DataFrame(holdout_table).to_csv(outdir / "lexical_ablation_internal_holdout.csv", index=False)

    pd.DataFrame(incremental_rows).to_csv(outdir / "lexical_ablation_incremental_deltas.csv", index=False)
    pd.DataFrame([{"transition": k, **v} for k, v in paired_bootstrap.items()]).to_csv(outdir / "lexical_ablation_paired_bootstrap.csv", index=False)
    pd.DataFrame(per_cat_rows).to_csv(outdir / "lexical_ablation_per_category.csv", index=False)
    pd.DataFrame(transition_error_rows).to_csv(outdir / "lexical_ablation_error_transitions.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(outdir / "lexical_ablation_boundary_audit.csv", index=False)
    pd.DataFrame(neg_audit_rows).to_csv(outdir / "lexical_ablation_negative_pattern_audit.csv", index=False)
    pd.DataFrame(neg_detail_rows).to_csv(outdir / "lexical_ablation_negative_detail.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(outdir / "lexical_ablation_threshold_audit.csv", index=False)
    pd.DataFrame(idf_rows).to_csv(outdir / "lexical_ablation_idf_audit.csv", index=False)

    # Also per-fold CSV
    pd.DataFrame(per_fold_macro).to_csv(outdir / "lexical_ablation_per_fold_macro.csv", index=False)

    # Per-category reports per variant (nested)
    for variant in ABLATION_VARIANTS:
        nested_reports[variant].to_csv(outdir / f"lexical_ablation_{variant}_nested_per_category.csv", index=False)
        holdout_reports[variant].to_csv(outdir / f"lexical_ablation_{variant}_holdout_per_category.csv", index=False)

    print(f"Wrote ablation outputs to {outdir}")
    # Print regression anchor check
    print(f"Regression anchors: A4 nested macro {nested_reports['A4'].loc[nested_reports['A4'].category=='MACRO AVG','f1'].iloc[0]:.4f} (expected ~0.942)")
    print(f"                   A5 nested macro {nested_reports['A5'].loc[nested_reports['A5'].category=='MACRO AVG','f1'].iloc[0]:.4f} (expected ~0.934)")
    # Print delta table
    print(pd.DataFrame(incremental_rows).to_string(index=False))


if __name__ == "__main__":
    main()
