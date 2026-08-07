"""
Experiment 3 — Selective semantic fallback H1/H2 (asymmetric, OFF, nested).

H1 = A1 (frozen whole-word any-hit) + S1 (TF-IDF LR) fallback
H2 = A1 + S3 (frozen NLI) fallback

Identical outer splits seed 42, 3-fold. Thresholds + (for H1) C tuned via inner
CV optimising FULL HYBRID macro-F1, with OFF option and conservative tie-break.
Outer validation never influences thresholds/C. Asymmetric: A1=1 cannot be vetoed.
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

from v4.config import CATEGORIES, TAXONOMY_VERSION, CATEGORY_LABELS
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_dev_test_split, make_cv_splits, RANDOM_SEED
from v4.evaluation.metrics import evaluate, accounting_report
from v4.evaluation.bootstrap import bootstrap_all
from v4.ablation.lexical_ablation import score_for_variant, any_hit_predictions
from v4.evaluation.hybrid_nested import run_nested_hybrid_S1, run_nested_hybrid_S3, _get_nli_scores_cached
from v4.evaluation.nested import run_nested_cv_for_method as run_lexical_nested
from v4.semantic.model_config import S1_VECTORISER_CONFIG, S1_C_GRID, S1_CLASS_WEIGHT, S2_MODEL_ID, S2_REVISION, S3_MODEL_ID, S3_REVISION
from v4.methods.lexical_baseline import VECTORISER_CONFIG
from v4.hybrid.selective_fallback import OFF_THRESHOLD


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
    return versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO_ROOT / "v3/manual_work/uk_analyst_corpus_v4_clean.csv"))
    ap.add_argument("--gold", default=str(REPO_ROOT / "v3/manual_work/gold_standard_annotation_workbook_v2.xlsx"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "v4/results/hybrid"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--n_outer_splits", type=int, default=3)
    ap.add_argument("--skip_s3", action="store_true", help="skip H2 (NLI) for speed")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    gold_df, y, texts = load_gold_with_texts(args.gold, args.corpus)
    n = len(gold_df)
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pkg_versions = get_package_versions()

    outer_splits, _ = make_cv_splits(gold_df, texts=texts, n_splits=args.n_outer_splits, seed=args.seed)

    # Verify identical to ablation
    ablation_summary_path = REPO_ROOT / "v4/results/ablation/lexical_ablation_summary.json"
    outer_ids_match = None
    if ablation_summary_path.exists():
        try:
            abl = json.load(open(ablation_summary_path))
            abl_ids = [set(entry["validation_ids"]) for entry in abl["outer_splits"]["outer_fold_ids"]]
            new_ids = [set(gold_df.iloc[val_idx]["posting_id"]) for _, val_idx in outer_splits]
            outer_ids_match = all(a == b for a, b in zip(abl_ids, new_ids))
            assert outer_ids_match
            print(f"Outer fold IDs IDENTICAL to ablation: {outer_ids_match}")
        except Exception as e:
            print(f"Warning outer check: {e}")

    # Lexical anchors A1 (same as semantic)
    pred_A1 = np.zeros((n, len(CATEGORIES)), dtype=int)
    scores_A1 = np.zeros((n, len(CATEGORIES)), dtype=float)
    for _, val_idx in outer_splits:
        S = score_for_variant("A1", [texts[i] for i in val_idx])
        pred_A1[val_idx] = any_hit_predictions(S)
        scores_A1[val_idx] = S

    # A2/A4/A5 via lexical nested for regression anchor check
    res_A4 = run_lexical_nested(gold_df, y, texts, outer_splits, method_name="unweighted_lexical", seed=args.seed)
    res_A5 = run_lexical_nested(gold_df, y, texts, outer_splits, method_name="weighted_lexical_tfidf", seed=args.seed)
    pred_A4 = res_A4["nested_predictions"]
    pred_A5 = res_A5["nested_predictions"]

    # H1
    print("Running H1 = A1 + S1 (nested hybrid, C+thresholds on hybrid F1)...")
    t0 = time.time()
    res_H1 = run_nested_hybrid_S1(gold_df, y, texts, outer_splits, seed=args.seed)
    t_H1 = time.time() - t0
    pred_H1 = res_H1["nested_predictions"]
    sem_H1 = res_H1["nested_scores_semantic"]

    # H2
    if args.skip_s3:
        print("Skipping H2 per --skip_s3")
        pred_H2 = None
        sem_H2 = None
        res_H2 = None
        t_H2 = 0
        nli_prov = None
    else:
        print("Running H2 = A1 + S3 NLI (nested hybrid, thresholds on hybrid F1, cached NLI)... this may take ~650s without cache")
        t0 = time.time()
        res_H2 = run_nested_hybrid_S3(gold_df, y, texts, outer_splits, seed=args.seed)
        t_H2 = time.time() - t0
        pred_H2 = res_H2["nested_predictions"]
        sem_H2 = res_H2["nested_scores_semantic"]
        nli_prov = res_H2.get("nli_provenance")

    method_preds = {"A1": pred_A1, "A4": pred_A4, "A5": pred_A5, "H1_A1_S1": pred_H1}
    method_scores_sem = {"H1_A1_S1": sem_H1}
    runtimes = {"A1": 0.0, "A4": 0.0, "A5": 0.0, "H1_A1_S1": float(t_H1)}
    outer_infos = {"H1_A1_S1": res_H1["outer_fold_info"], "A4": res_A4["outer_fold_info"], "A5": res_A5["outer_fold_info"]}
    if res_H2 is not None:
        method_preds["H2_A1_S3"] = pred_H2
        method_scores_sem["H2_A1_S3"] = sem_H2
        runtimes["H2_A1_S3"] = float(t_H2)
        outer_infos["H2_A1_S3"] = res_H2["outer_fold_info"]

    # Metrics
    nested_reports = {}
    nested_accounting = {}
    nested_bootstrap = {}
    for m, pred in method_preds.items():
        rep = evaluate(y, pred)
        acc = accounting_report(y, pred)
        nested_reports[m] = rep
        nested_accounting[m] = acc
        nested_bootstrap[m] = bootstrap_all(y, pred, n_bootstrap=args.n_bootstrap, seed=args.seed)

    # Paired deltas vs A1
    from sklearn.metrics import f1_score

    def paired_delta(a, b):
        rng = np.random.default_rng(args.seed)
        n_b = args.n_bootstrap
        deltas = np.empty(n_b)
        for bi in range(n_b):
            idx = rng.integers(0, len(y), size=len(y))
            fa = f1_score(y[idx], a[idx], average="macro", zero_division=0)
            fb = f1_score(y[idx], b[idx], average="macro", zero_division=0)
            deltas[bi] = fb - fa
        return {
            "delta_macro_f1": float(f1_score(y, b, average="macro", zero_division=0) - f1_score(y, a, average="macro", zero_division=0)),
            "ci_lower": float(np.percentile(deltas, 2.5)),
            "ci_upper": float(np.percentile(deltas, 97.5)),
            "n_bootstrap": int(n_b),
        }

    paired = {}
    for m in ["H1_A1_S1", "H2_A1_S3"]:
        if m in method_preds:
            d = paired_delta(pred_A1, method_preds[m])
            # also compute wrong->correct counts
            a_wrong = (pred_A1 != y)
            b_wrong = (method_preds[m] != y)
            w2c = int(((a_wrong) & (~b_wrong)).sum())
            c2w = int(((~a_wrong) & (b_wrong)).sum())
            d.update({"wrong_to_correct": w2c, "correct_to_wrong": c2w, "net_corrected": w2c - c2w})
            paired[f"{m}_vs_A1"] = d

    # Seen/unseen analysis (frozen A1 lexicon as probe)
    S_A1_all = score_for_variant("A1", texts)
    seen_mask = (S_A1_all > 0)  # lexical hit
    # For analysis: seen = lexical hit, but gold positive seen = gold=1 & seen_mask
    overall_seen = {}
    for m, pred in method_preds.items():
        # recall on seen vs unseen positives
        gold_pos = (y == 1)
        seen_pos = gold_pos & seen_mask
        unseen_pos = gold_pos & (~seen_mask)
        n_seen = int(seen_pos.sum())
        n_unseen = int(unseen_pos.sum())
        rec_seen = float((pred[seen_pos] == 1).mean()) if n_seen else 0.0
        rec_unseen = float((pred[unseen_pos] == 1).mean()) if n_unseen else 0.0
        overall_seen[m] = {"n_seen": n_seen, "n_unseen": n_unseen, "recall_seen": rec_seen, "recall_unseen": rec_unseen}

    # Total anchors check
    n_total_pos = int((y == 1).sum())
    n_seen_pos = int(((y == 1) & (S_A1_all > 0)).sum())
    n_unseen_pos = int(((y == 1) & (S_A1_all == 0)).sum())

    total_runtime = time.time() - start_time

    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "git_commit": git_commit,
        "timestamp_utc": timestamp,
        "random_seed": args.seed,
        "n_examples": int(n),
        "outer_splits": {
            "n_outer_splits": int(len(outer_splits)),
            "outer_ids_match_ablation": outer_ids_match,
            "outer_fold_ids": [
                {"fold": int(i), "validation_ids": gold_df.iloc[val_idx]["posting_id"].tolist()}
                for i, (_, val_idx) in enumerate(outer_splits)
            ],
        },
        "hybrid_config": {
            "H1": "A1 (whole-word any-hit) + S1 (TF-IDF LR C in [0.1,1,10], thresholds tuned on hybrid F1, OFF=2.0, asymmetric, conservative tie-break)",
            "H2": "A1 + S3 (distilbert-base-uncased-mnli MAX, thresholds tuned on hybrid F1, OFF, asymmetric, cached)",
            "asymmetric": "A1=1 cannot be vetoed; fallback only when A1=0",
            "off_option": True,
            "threshold_grid": "51 points 0..1 step 0.02 + OFF(2.0)",
            "optimisation": "full hybrid macro-F1 via inner 2-fold CV per outer_train, conservative highest-wins tie-break",
            "batch_invariant": True,
            "S3_cache": nli_prov if res_H2 is not None else None,
        },
        "runtimes_seconds": runtimes,
        "total_runtime_seconds": float(total_runtime),
        "nested": {
            m: {
                "report": nested_reports[m].to_dict(orient="records"),
                "accounting": nested_accounting[m],
                "bootstrap": nested_bootstrap[m],
                "per_fold_macro": [
                    {
                        "fold": int(f),
                        "macro_f1": float(evaluate(y[val_idx], method_preds[m][val_idx]).loc[evaluate(y[val_idx], method_preds[m][val_idx]).category == "MACRO AVG", "f1"].iloc[0]),
                    }
                    for f, (_, val_idx) in enumerate(outer_splits)
                ],
                "outer_fold_info": [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
                    for info in outer_infos[m]
                ] if m in outer_infos else None,
            }
            for m in method_preds
        },
        "paired_deltas": paired,
        "seen_unseen": {
            "n_total_positive_cells": int(n_total_pos),
            "n_seen_positive_cells": int(n_seen_pos),
            "n_unseen_positive_cells": int(n_unseen_pos),
            "per_method_recall": overall_seen,
            "expected_anchors": {"A1_macro": 0.9429, "seen": 1016, "unseen": 39},
        },
        "regression_anchors": {
            "A1_expected_macro": 0.9429,
            "A1_actual": float(nested_reports["A1"].loc[nested_reports["A1"].category == "MACRO AVG", "f1"].iloc[0]),
            "seen_expected": 1016,
            "seen_actual": int(n_seen_pos),
            "unseen_expected": 39,
            "unseen_actual": int(n_unseen_pos),
        },
        "notes": "Hybrid fallback experiment 3 — post-hoc hypotheses H1-H6 documented in HYBRID_FALLBACK.md; nested thresholds cannot be tuned on test/external_locked_test",
    }

    (outdir / "hybrid_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # CSVs
    nested_rows = []
    for m in method_preds:
        rep = nested_reports[m]
        acc = nested_accounting[m]
        boot = nested_bootstrap[m]
        nested_rows.append({
            "method": m,
            "macro_f1": float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0]),
            "macro_ci_lower": float(boot["macro_f1"]["lower"]),
            "macro_ci_upper": float(boot["macro_f1"]["upper"]),
            "micro_f1": float(rep.loc[rep.category == "MICRO AVG", "f1"].iloc[0]),
            "subset_accuracy": float(rep.loc[rep.category == "SUBSET ACCURACY", "f1"].iloc[0]),
            "hamming_accuracy": float(rep.loc[rep.category == "HAMMING ACCURACY", "f1"].iloc[0]),
            "TP": int(acc["total_TP"]), "FP": int(acc["total_FP"]), "FN": int(acc["total_FN"]), "TN": int(acc["total_TN"]),
            "runtime_seconds": float(runtimes.get(m, 0)),
        })
    pd.DataFrame(nested_rows).to_csv(outdir / "hybrid_nested_results.csv", index=False)

    # per_fold
    per_fold_rows = []
    for m in method_preds:
        for f, (_, val_idx) in enumerate(outer_splits):
            rep = evaluate(y[val_idx], method_preds[m][val_idx])
            per_fold_rows.append({"method": m, "fold": int(f), "macro_f1": float(rep.loc[rep.category == "MACRO AVG", "f1"].iloc[0])})
    pd.DataFrame(per_fold_rows).to_csv(outdir / "hybrid_per_fold.csv", index=False)

    # paired deltas
    pd.DataFrame([{"comparison": k, **v} for k, v in paired.items()]).to_csv(outdir / "hybrid_paired_deltas.csv", index=False)

    # seen_unseen detail per category
    seen_rows = []
    for m in method_preds:
        for ci, cat in enumerate(CATEGORIES):
            gold_pos = (y[:, ci] == 1)
            seen_pos = gold_pos & (S_A1_all[:, ci] > 0)
            unseen_pos = gold_pos & (S_A1_all[:, ci] == 0)
            n_seen_c = int(seen_pos.sum())
            n_unseen_c = int(unseen_pos.sum())
            rec_seen = float((method_preds[m][seen_pos, ci] == 1).mean()) if n_seen_c else None
            rec_unseen = float((method_preds[m][unseen_pos, ci] == 1).mean()) if n_unseen_c else None
            seen_rows.append({"method": m, "category": cat, "n_seen_pos": n_seen_c, "n_unseen_pos": n_unseen_c, "recall_seen": rec_seen, "recall_unseen": rec_unseen})
    pd.DataFrame(seen_rows).to_csv(outdir / "hybrid_seen_unseen.csv", index=False)

    # disagreements: where hybrid differs from A1
    disag_rows = []
    for m in ["H1_A1_S1", "H2_A1_S3"]:
        if m not in method_preds:
            continue
        diff = (method_preds[m] != pred_A1)
        for i in range(n):
            for j, cat in enumerate(CATEGORIES):
                if diff[i, j]:
                    disag_rows.append({
                        "posting_id": gold_df.iloc[i]["posting_id"],
                        "category": cat,
                        "gold": int(y[i, j]),
                        "A1_pred": int(pred_A1[i, j]),
                        "hybrid_pred": int(method_preds[m][i, j]),
                        "hybrid_method": m,
                        "semantic_score": float(method_scores_sem[m][i, j]),
                        "threshold": float(outer_infos[m][next(f for f, (_, vi) in enumerate(outer_splits) if i in vi)]["thresholds"][j]) if "thresholds" in outer_infos[m][0] else -1,
                    })
    if disag_rows:
        pd.DataFrame(disag_rows).to_csv(outdir / "hybrid_disagreements.csv", index=False)

    # runtime
    runtime_rows = [{"method": m, "runtime_seconds": float(runtimes[m])} for m in runtimes]
    runtime_rows.append({"method": "TOTAL", "runtime_seconds": float(total_runtime)})
    pd.DataFrame(runtime_rows).to_csv(outdir / "hybrid_runtime.csv", index=False)

    print(f"Wrote hybrid outputs to {outdir}")
    for r in nested_rows:
        print(f"{r['method']:12} macro {r['macro_f1']:.4f} [{r['macro_ci_lower']:.3f},{r['macro_ci_upper']:.3f}]")
    for k, v in paired.items():
        print(f"{k}: Δ {v['delta_macro_f1']:.4f} CI [{v['ci_lower']:.3f},{v['ci_upper']:.3f}] w2c {v['wrong_to_correct']} c2w {v['correct_to_wrong']}")
    print(f"Anchors: A1 {summary['regression_anchors']['A1_actual']:.4f} (exp 0.9429) seen {n_seen_pos} (exp 1016) unseen {n_unseen_pos} (exp 39)")


if __name__ == "__main__":
    main()
