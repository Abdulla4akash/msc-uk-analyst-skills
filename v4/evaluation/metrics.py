"""
Metrics for v4.

Mirrors v3/evaluate.py:evaluate but adds:
- macro/micro F1, exact-match, Hamming
- per-category support / predicted counts
- aggregate rows for reporting
- bootstrap-friendly helpers
- accounting summary
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

from v4.config import CATEGORIES


def evaluate(y_true, y_pred, categories=None):
    """Per-category and aggregate metrics as tidy DataFrame.

    y_true, y_pred: ndarray (n, 13) int 0/1
    Returns DataFrame with rows for each category plus MACRO/MICRO/SUBSET/HAMMING.
    """
    if categories is None:
        categories = CATEGORIES
    n_cats = len(categories)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(n_cats)
    )
    rows = []
    for i, c in enumerate(categories):
        rows.append({
            "category": c,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(y_true[:, i].sum()),
            "predicted": int(y_pred[:, i].sum()),
        })
    for name, avg in (("MACRO AVG", "macro"), ("MICRO AVG", "micro")):
        pp, rr, ff, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        rows.append({
            "category": name,
            "precision": float(pp),
            "recall": float(rr),
            "f1": float(ff),
            "support": int(y_true.sum()),
            "predicted": int(y_pred.sum()),
        })
    rows.append({
        "category": "SUBSET ACCURACY",
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float((y_true == y_pred).all(axis=1).mean()),
        "support": int(len(y_true)),
        "predicted": int(len(y_true)),
    })
    rows.append({
        "category": "HAMMING ACCURACY",
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float((y_true == y_pred).mean()),
        "support": int(y_true.size),
        "predicted": int(y_true.size),
    })
    # Also add Hamming loss as derived value in accounting, not here.
    return pd.DataFrame(rows)


def accounting_report(y_true, y_pred, categories=None):
    """Detailed accounting summary.

    Returns dict with posting*label decisions, TP/FP/FN/TN per category, etc.
    """
    if categories is None:
        categories = CATEGORIES
    n, k = y_true.shape
    total_cells = int(n * k)
    per_cat = {}
    total_fp = 0
    total_fn = 0
    total_tp = 0
    total_tn = 0
    for i, c in enumerate(categories):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        per_cat[c] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        total_fp += fp
        total_fn += fn
        total_tp += tp
        total_tn += tn
    n_ads_with_error = int(((y_true != y_pred).any(axis=1)).sum())
    avg_labels_true = float(y_true.sum(axis=1).mean())
    avg_labels_pred = float(y_pred.sum(axis=1).mean())
    prevalence = {c: float(y_true[:, i].mean()) for i, c in enumerate(categories)}
    return {
        "n_postings": int(n),
        "n_categories": int(k),
        "total_cells": int(total_cells),
        "per_category": per_cat,
        "total_TP": int(total_tp),
        "total_FP": int(total_fp),
        "total_FN": int(total_fn),
        "total_TN": int(total_tn),
        "n_ads_with_at_least_one_error": int(n_ads_with_error),
        "avg_labels_per_ad_true": avg_labels_true,
        "avg_labels_per_ad_pred": avg_labels_pred,
        "label_prevalence": prevalence,
        "hamming_loss": float((y_true != y_pred).mean()),
        "hamming_accuracy": float((y_true == y_pred).mean()),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
    }


def format_report(df, title):
    lines = [f"\n{'=' * 72}", title, "=" * 72,
             f"{'category':22s} {'prec':>7s} {'rec':>7s} {'F1':>7s} {'n_true':>7s} {'n_pred':>7s}"]
    for _, r in df.iterrows():
        if r["category"] in ("MACRO AVG", "MICRO AVG"):
            lines.append("-" * 72)
        pv = "  n/a  " if pd.isna(r["precision"]) else f"{r['precision']:7.3f}"
        rv = "  n/a  " if pd.isna(r["recall"]) else f"{r['recall']:7.3f}"
        lines.append(f"{r['category']:22s} {pv} {rv} {r['f1']:7.3f} {r['support']:7d} {r['predicted']:7d}")
    return "\n".join(lines)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def micro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def subset_accuracy(y_true, y_pred):
    return float((y_true == y_pred).all(axis=1).mean())


def hamming_accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())
