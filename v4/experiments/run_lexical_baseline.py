"""
v4 experiment runner — three classical baselines with corrected protocol.

What is evaluated
-----------------
- Baseline 0 — unweighted lexical match (no TF-IDF)
- Baseline 1 — cosine TF-IDF (inductive, raw cosine)
- Baseline 2 — weighted lexical TF-IDF (inductive, stable IDF-weighted hit)

All fitted statistics are learned from TRAINING data only (never external data).
Scores are batch-invariant (no max-normalisation).
Thresholds are tuned on internal development data only; model selection is dev-based.

Terminology
-----------
- internal_tuning    : ~100 of the 300 gold postings (historically "dev")
- internal_holdout   : ~200 of the 300 (historically "test") — NOT an external test
- external_locked_test : RESERVED for future independently collected dataset (does not exist yet)
  Only that future dataset should be called "test" in publication-facing tables.

Two operating modes
-------------------
1. --mode internal_holdout  (mirrors v3 split for audit comparability)
   - One stratified internal_tuning / internal_holdout split (100/200, seeded).
   - Fit vectoriser on internal_tuning texts only.
   - Tune thresholds on internal_tuning scores.
   - Select best variant by internal_tuning macro-F1.
   - Report internal_tuning and internal_holdout metrics for all variants; internal_holdout only after freezing.

2. --mode nested_cv  (preferred publication-grade internal development estimate)
   - Outer: stratified GroupKFold (k=3, limited by smallest role_family=3) over the FULL 300
           (treated as development corpus).
   - Inner: for each outer_train, stratified 2-fold CV using ONLY that outer_train's texts/labels
            to tune thresholds; fallback documented if not feasible.
   - For each outer fold: fit vectoriser on outer_train, apply thresholds tuned on inner folds,
            score outer_validation, store predictions.
   - Aggregate nested predictions (each posting predicted exactly once via its outer validation fold)
           → nested-CV internal development estimate + bootstrap CIs (supplementary) + fold variation.
   - Future external_locked_test will use a locked model re-fitted on all 300.

Outputs
-------
results/v4_lexical_summary.json   — machine-readable summary (reproducibility fields)
results/v4_*_report.csv          — per-category reports
results/v4_results.txt           — human-readable summary
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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v4.config import CATEGORIES, TAXONOMY_VERSION
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_dev_test_split, make_cv_splits, RANDOM_SEED
from v4.evaluation.metrics import evaluate, accounting_report, format_report
from v4.evaluation.bootstrap import bootstrap_all
from v4.evaluation.nested import run_nested_cv_for_method
from v4.methods.lexical_baseline import (
    VECTORISER_CONFIG,
    unweighted_lexical_scores,
    tune_thresholds,
    apply_thresholds,
    fit_tfidf_vectoriser,
    cosine_tfidf_scores,
    weighted_lexical_scores,
)


def get_git_commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        return c
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


def run_internal_holdout(gold_df, y, texts, outdir, seed=RANDOM_SEED):
    t_start = time.time()
    split = make_dev_test_split(gold_df, dev_frac=1 / 3, seed=seed)
    is_internal_tuning = split["is_internal_tuning"]
    is_internal_holdout = split["is_internal_holdout"]

    internal_tuning_texts = [texts[i] for i in np.where(is_internal_tuning)[0]]
    internal_holdout_texts = [texts[i] for i in np.where(is_internal_holdout)[0]]
    all_texts = texts

    results = {}

    # Baseline 0 — unweighted lexical (no fitting)
    S0 = unweighted_lexical_scores(all_texts)
    thr0 = tune_thresholds(S0[is_internal_tuning], y[is_internal_tuning])
    pred0 = apply_thresholds(S0, thr0)
    results["unweighted_lexical"] = {
        "scores": S0, "thresholds": thr0, "pred": pred0,
        "fitted_on": "no fitting (lexicon only)",
    }

    # Baseline 1 — cosine TF-IDF fitted on internal_tuning only
    S1, vec1 = cosine_tfidf_scores(internal_tuning_texts, all_texts)
    thr1 = tune_thresholds(S1[is_internal_tuning], y[is_internal_tuning])
    pred1 = apply_thresholds(S1, thr1)
    results["cosine_tfidf"] = {
        "scores": S1, "thresholds": thr1, "pred": pred1, "vec": vec1,
        "fitted_on": "internal_tuning texts only",
    }

    # Baseline 2 — weighted lexical TF-IDF fitted on internal_tuning only
    S2, vec2 = weighted_lexical_scores(internal_tuning_texts, all_texts)
    thr2 = tune_thresholds(S2[is_internal_tuning], y[is_internal_tuning])
    pred2 = apply_thresholds(S2, thr2)
    results["weighted_lexical_tfidf"] = {
        "scores": S2, "thresholds": thr2, "pred": pred2, "vec": vec2,
        "fitted_on": "internal_tuning texts only",
    }

    # Internal comparison: weighted and unweighted are the candidates of interest.
    # Do NOT auto-declare a winner on a 0.001 difference (Fix 3).
    internal_tuning_scores = {}
    for name, r in results.items():
        rep = evaluate(y[is_internal_tuning], r["pred"][is_internal_tuning])
        macro = float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])
        internal_tuning_scores[name] = macro
    # Retain both lexical baselines as serious candidates; no automatic winner.
    selection_status = "undecided"
    comparison_note = "weighted and unweighted lexical baselines are effectively tied under current internal development evaluation"
    candidate_baselines = ["unweighted_lexical", "weighted_lexical_tfidf"]

    elapsed = time.time() - t_start
    return results, split, internal_tuning_scores, selection_status, comparison_note, candidate_baselines, elapsed


def run_nested_cv(gold_df, y, texts, outdir, seed=RANDOM_SEED, n_outer_splits=3, grid=None):
    t_start = time.time()
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)

    # Outer splits: stratified over 300
    outer_splits, outer_meta = make_cv_splits(gold_df, texts=texts, n_splits=n_outer_splits, n_repeats=1, seed=seed)

    methods = ["unweighted_lexical", "weighted_lexical_tfidf", "cosine_tfidf"]
    nested_by_method = {}
    for method in methods:
        res = run_nested_cv_for_method(gold_df, y, texts, outer_splits, method_name=method, seed=seed, grid=grid)
        nested_by_method[method] = res

    elapsed = time.time() - t_start
    return nested_by_method, outer_splits, outer_meta, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO_ROOT / "v3/manual_work/uk_analyst_corpus_v4_clean.csv"))
    ap.add_argument("--gold", default=str(REPO_ROOT / "v3/manual_work/gold_standard_annotation_workbook_v2.xlsx"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "v4/results"))
    ap.add_argument("--mode", choices=["internal_holdout", "nested_cv", "both"], default="both",
                    help="internal_holdout: single stratified split; nested_cv: nested 3-fold CV; both: run both")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n_outer_splits", type=int, default=3, help="outer folds (3 is max feasible for role_family)")
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    args = ap.parse_args()

    mode = args.mode
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gold_df, y, texts = load_gold_with_texts(args.gold, args.corpus)
    n = len(gold_df)

    git_commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pid_hash = hashlib.sha256(",".join(sorted(gold_df["posting_id"].astype(str))).encode()).hexdigest()[:16]

    role_dist = gold_df["role_family"].value_counts().to_dict()
    cat_support = {c: int(y[:, i].sum()) for i, c in enumerate(CATEGORIES)}

    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_source": "v4/config.py (frozen copy of v3/base-model/config.py)",
        "git_commit": git_commit,
        "timestamp_utc": timestamp,
        "random_seed": args.seed,
        "n_examples": int(n),
        "posting_id_hash": pid_hash,
        "role_family_distribution": role_dist,
        "category_support": cat_support,
        "category_prevalence": {c: float(y[:, i].mean()) for i, c in enumerate(CATEGORIES)},
        "vectoriser_config": VECTORISER_CONFIG,
        "package_versions": get_package_versions(),
        "data_flow": {
            "corpus": "uk_analyst_corpus_v4_clean.csv (820 postings, job_summary only field used)",
            "gold": "gold_standard_annotation_workbook_v2.xlsx sheet Annotation (300 postings) — treated as DEVELOPMENT material",
            "internal_split_method": "stratified on role_family (human-confirmed), seed=42",
            "external_locked_test": "RESERVED for future independently collected dataset — does not exist yet; do not fabricate",
            "fit_scope": "TfidfVectorizer fitted on internal_tuning / outer_train texts only; validation/holdout texts only transformed",
            "threshold_scope": "tuned on internal_tuning (holdout) or inner validation folds only (nested CV); outer validation labels never used for thresholds",
            "model_selection": "best variant chosen by internal_tuning macro-F1 only (never internal_holdout, never external_locked_test)",
            "scoring": "batch-invariant: no max-normalisation; cosine is raw, weighted lexical is sum(IDF matched)/sum(IDF lexicon)",
        },
        "modes_run": mode,
    }

    full_text_report = [f"v4 lexical baselines — {timestamp}  commit={git_commit}  seed={args.seed}",
                        f"300 postings treated as DEVELOPMENT material; external_locked_test does not exist yet"]

    if mode in ("internal_holdout", "both"):
        results, split, internal_tuning_scores, selection_status, comparison_note, candidate_baselines, elapsed = run_internal_holdout(
            gold_df, y, texts, outdir, seed=args.seed
        )
        is_internal_tuning = split["is_internal_tuning"]
        is_internal_holdout = split["is_internal_holdout"]

        # Bootstrap both lexical methods (Fix 3) — not just a single "best"
        bootstrap_results = {}
        for name, r in results.items():
            if name in ("unweighted_lexical", "weighted_lexical_tfidf"):
                y_holdout = y[is_internal_holdout]
                p_holdout = r["pred"][is_internal_holdout]
                bootstrap_results[name] = bootstrap_all(y_holdout, p_holdout, n_bootstrap=args.n_bootstrap, seed=args.seed)

        holdout_detail = {}
        for name, r in results.items():
            rep_tuning = evaluate(y[is_internal_tuning], r["pred"][is_internal_tuning])
            rep_holdout = evaluate(y[is_internal_holdout], r["pred"][is_internal_holdout])
            acc_tuning = accounting_report(y[is_internal_tuning], r["pred"][is_internal_tuning])
            acc_holdout = accounting_report(y[is_internal_holdout], r["pred"][is_internal_holdout])

            rep_tuning.to_csv(outdir / f"v4_{name}_internal_tuning.csv", index=False)
            rep_holdout.to_csv(outdir / f"v4_{name}_internal_holdout.csv", index=False)
            (outdir / f"v4_{name}_accounting_internal_tuning.json").write_text(json.dumps(acc_tuning, indent=2))
            (outdir / f"v4_{name}_accounting_internal_holdout.json").write_text(json.dumps(acc_holdout, indent=2))

            holdout_detail[name] = {
                "thresholds": {c: float(r["thresholds"][i]) for i, c in enumerate(CATEGORIES)},
                "thresholds_source": "tuned on internal_tuning split only (grid 0..1, 201 points, per-category max F1)",
                "fitted_on": r["fitted_on"],
                "internal_tuning_macro_f1": float(rep_tuning.loc[rep_tuning.category == "MACRO AVG", "f1"].iloc[0]),
                "internal_tuning_micro_f1": float(rep_tuning.loc[rep_tuning.category == "MICRO AVG", "f1"].iloc[0]),
                "internal_tuning_subset_accuracy": float(rep_tuning.loc[rep_tuning.category == "SUBSET ACCURACY", "f1"].iloc[0]),
                "internal_tuning_hamming_accuracy": float(rep_tuning.loc[rep_tuning.category == "HAMMING ACCURACY", "f1"].iloc[0]),
                "internal_holdout_macro_f1": float(rep_holdout.loc[rep_holdout.category == "MACRO AVG", "f1"].iloc[0]),
                "internal_holdout_micro_f1": float(rep_holdout.loc[rep_holdout.category == "MICRO AVG", "f1"].iloc[0]),
                "internal_holdout_subset_accuracy": float(rep_holdout.loc[rep_holdout.category == "SUBSET ACCURACY", "f1"].iloc[0]),
                "internal_holdout_hamming_accuracy": float(rep_holdout.loc[rep_holdout.category == "HAMMING ACCURACY", "f1"].iloc[0]),
                "internal_tuning_accounting": acc_tuning,
                "internal_holdout_accounting": acc_holdout,
            }
            if name in bootstrap_results:
                holdout_detail[name]["internal_holdout_bootstrap"] = bootstrap_results[name]

            full_text_report.append(format_report(rep_tuning, f"v4 {name} — internal_tuning (n={is_internal_tuning.sum()})"))
            full_text_report.append(format_report(rep_holdout, f"v4 {name} — internal_holdout (n={is_internal_holdout.sum()})  [reported after freezing; NOT external_locked_test]"))
            full_text_report.append(f"  thresholds {name}: " + ", ".join(f"{c}={r['thresholds'][i]:.3f}" for i, c in enumerate(CATEGORIES)))

        holdout_block = {
            "split": {
                "method": "stratified internal_tuning/internal_holdout on role_family, dev_frac=1/3",
                "seed": args.seed,
                "n_internal_tuning": int(is_internal_tuning.sum()),
                "n_internal_holdout": int(is_internal_holdout.sum()),
                "internal_tuning_ids": split["internal_tuning_ids"],
                "internal_holdout_ids": split["internal_holdout_ids"],
                "role_family_internal_tuning": gold_df[is_internal_tuning]["role_family"].value_counts().to_dict(),
                "role_family_internal_holdout": gold_df[is_internal_holdout]["role_family"].value_counts().to_dict(),
            },
            "model_selection": {
                "selection_status": selection_status,
                "comparison": comparison_note,
                "candidate_baselines": candidate_baselines,
                "internal_tuning_macro_f1_by_variant": internal_tuning_scores,
                "note": "no automatic winner — retain both lexical baselines; external_locked_test does not exist",
            },
            "runtime_sec": float(elapsed),
            "results": holdout_detail,
        }
        summary["internal_holdout"] = holdout_block
        full_text_report.append(
            f"\nInternal holdout comparison (internal_tuning macro-F1): {internal_tuning_scores}  "
            f"[{comparison_note}; selection_status={selection_status}]"
        )
        full_text_report.append(f"Holdout runtime {elapsed:.1f}s  (vectoriser fitted on internal_tuning only; no batch-max normalisation)")

        for name, r in results.items():
            dfp = pd.DataFrame(r["pred"], columns=CATEGORIES)
            dfp.insert(0, "posting_id", gold_df["posting_id"].values)
            dfp.insert(1, "split", np.where(is_internal_tuning, "internal_tuning", "internal_holdout"))
            dfp.to_csv(outdir / f"v4_{name}_predictions_gold_internal_holdout.csv", index=False)
            dfp.to_csv(outdir / f"v4_{name}_predictions_gold.csv", index=False)

    if mode in ("nested_cv", "both"):
        grid = np.linspace(0.0, 1.0, 51)
        nested_by_method, outer_splits, outer_meta, elapsed = run_nested_cv(
            gold_df, y, texts, outdir, seed=args.seed, n_outer_splits=args.n_outer_splits, grid=grid
        )
        n_outer = len(outer_splits)
        nested_detail = {}
        per_fold_rows = []
        for method, res in nested_by_method.items():
            nested_pred = res["nested_predictions"]
            nested_scores = res["nested_scores"]
            outer_fold_info = res["outer_fold_info"]
            rep = evaluate(y, nested_pred)
            acc = accounting_report(y, nested_pred)
            # Exclude outer_fold_info thresholds arrays from CSV; write report
            rep.to_csv(outdir / f"v4_{method}_nested_cv_report.csv", index=False)
            (outdir / f"v4_{method}_nested_cv_accounting.json").write_text(json.dumps(acc, indent=2))

            # Bootstrap over nested predictions is supplementary — does NOT capture training variance
            boot = bootstrap_all(y, nested_pred, n_bootstrap=args.n_bootstrap, seed=args.seed)
            nested_detail[method] = {
                "method": method,
                "nested_macro_f1": float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0]),
                "nested_micro_f1": float(rep.loc[rep.category == "MICRO AVG", "f1"].iloc[0]),
                "nested_subset_accuracy": float(rep.loc[rep.category == "SUBSET ACCURACY", "f1"].iloc[0]),
                "nested_hamming_accuracy": float(rep.loc[rep.category == "HAMMING ACCURACY", "f1"].iloc[0]),
                "nested_bootstrap": boot,
                "nested_accounting": acc,
                "per_report": rep.to_dict(orient="records"),
                "outer_fold_info": outer_fold_info,
                "note_bootstrap": "posting-level bootstrap over fixed nested-CV predictions quantifies uncertainty conditional on those predictions but does NOT fully reproduce model-training uncertainty; report fold-to-fold variation as primary",
            }
            # Per-fold breakdown (fold-to-fold variation)
            for info in outer_fold_info:
                outer_val_idx = gold_df[gold_df["posting_id"].isin(info["outer_validation_ids"])].index
                rep_f = evaluate(y[outer_val_idx], nested_pred[outer_val_idx])
                macro = float(rep_f.loc[rep_f.category == "MACRO AVG", "f1"].iloc[0])
                per_fold_rows.append({"fold": info["outer_fold"], "variant": method, "method": method,
                                      "n_validation": info["outer_validation_n"],
                                      "n_train": info["outer_train_n"],
                                      "outer_macro_f1": macro,
                                      "macro_f1": macro,
                                      "inner_strategy": info["inner_strategy"]})
            full_text_report.append(format_report(rep, f"v4 {method} — nested cross-validated internal development estimate (n={n}, {n_outer} outer folds, genuinely nested)"))
            # Show thresholds per outer fold
            for info in outer_fold_info:
                thr_str = ", ".join(f"{c}={info['thresholds'][c]:.3f}" for c in CATEGORIES)
                full_text_report.append(f"  outer_fold {info['outer_fold']} thresholds {method}: {thr_str}  | inner={info['inner_strategy']}")

        per_fold_df = pd.DataFrame(per_fold_rows)
        if not per_fold_df.empty:
            per_fold_df.to_csv(outdir / "v4_nested_cv_per_fold_macro_f1.csv", index=False)

        summary["nested_cv"] = {
            "method": f"nested CV: outer Stratified GroupKFold (k={args.n_outer_splits}) + inner StratifiedKFold (k=2) per outer_train, using ONLY outer_train for fitting/thresholds",
            "outer_splits_meta": {k: v for k, v in outer_meta.items() if k != "group_ids"},
            "outer_meta": {k: v for k, v in outer_meta.items() if k != "group_ids"},
            "limitations": outer_meta["limitations"],
            "n_outer_splits": args.n_outer_splits,
            "n_outer_folds": n_outer,
            "runtime_sec": float(elapsed),
            "results": nested_detail,
            "per_fold_macro_f1": per_fold_rows,
            "grid": {"start": float(grid[0]), "stop": float(grid[-1]), "n_points": int(len(grid))},
            "interpretation": "Each outer validation fold uses thresholds and TF-IDF fitted ONLY on its outer_train (via inner CV), so outer labels never influence their own thresholds.",
        }
        full_text_report.append(f"\nNested CV runtime {elapsed:.1f}s  (each outer fold: fit on outer_train only; thresholds from inner CV only; {', '.join(outer_meta['limitations'][:2])})")

        for method, res in nested_by_method.items():
            dfp = pd.DataFrame(res["nested_predictions"], columns=CATEGORIES)
            dfp.insert(0, "posting_id", gold_df["posting_id"].values)
            dfp.to_csv(outdir / f"v4_{method}_nested_cv_predictions.csv", index=False)

    # Final summary
    summary["runtime_sec_total"] = None
    (outdir / "v4_lexical_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (outdir / "v4_results.txt").write_text("\n".join(full_text_report))

    print("\n".join(full_text_report))
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
