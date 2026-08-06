"""
Evaluation harness for the extraction methods.

Provides:
  - load_gold()          : gold-standard label matrix from the annotation workbook
  - make_split()         : reproducible dev/test split, stratified by role family
  - evaluate()           : per-category and aggregate precision / recall / F1
  - tune_thresholds()    : per-category threshold selection on the dev split only

The dev/test split matters: any method whose decision threshold is tuned must be
tuned on dev and reported on test, otherwise the reported score is optimistically
biased by fitting to the evaluation data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import CATEGORIES

RANDOM_SEED = 42


def load_gold(workbook_path):
    """Return (dataframe, label matrix) for the 300 annotated postings."""
    g = pd.read_excel(workbook_path, sheet_name="Annotation")
    g = g[g[CATEGORIES].notna().all(axis=1)].reset_index(drop=True)
    y = g[CATEGORIES].astype(int).values
    return g, y


def make_split(gold_df, dev_frac=1 / 3, seed=RANDOM_SEED):
    """Stratified dev/test split on role family. Returns boolean masks."""
    rng = np.random.default_rng(seed)
    is_dev = np.zeros(len(gold_df), dtype=bool)
    for _, idx in gold_df.groupby("role_family").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_dev = max(1, int(round(len(idx) * dev_frac)))
        is_dev[idx[:n_dev]] = True
    return is_dev, ~is_dev


def evaluate(y_true, y_pred, categories=CATEGORIES):
    """Per-category and aggregate metrics as a tidy dataframe."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(categories))
    )
    rows = [
        {"category": c, "precision": p[i], "recall": r[i], "f1": f[i],
         "support": int(y_true[:, i].sum()), "predicted": int(y_pred[:, i].sum())}
        for i, c in enumerate(categories)
    ]
    for name, avg in (("MACRO AVG", "macro"), ("MICRO AVG", "micro")):
        pp, rr, ff, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        rows.append({"category": name, "precision": pp, "recall": rr, "f1": ff,
                     "support": int(y_true.sum()), "predicted": int(y_pred.sum())})
    # exact-match (all 13 labels correct) and Hamming accuracy
    rows.append({"category": "SUBSET ACCURACY", "precision": np.nan, "recall": np.nan,
                 "f1": float((y_true == y_pred).all(axis=1).mean()),
                 "support": len(y_true), "predicted": len(y_true)})
    rows.append({"category": "HAMMING ACCURACY", "precision": np.nan, "recall": np.nan,
                 "f1": float((y_true == y_pred).mean()),
                 "support": y_true.size, "predicted": y_true.size})
    return pd.DataFrame(rows)


def tune_thresholds(scores, y_true, grid=None):
    """Pick the per-category threshold maximising F1 on the given (dev) data."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    thresholds = np.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        best_f, best_t = -1.0, 0.5
        for t in grid:
            pred = (scores[:, i] >= t).astype(int)
            _, _, f, _ = precision_recall_fscore_support(
                y_true[:, i], pred, average="binary", zero_division=0
            )
            if f > best_f:
                best_f, best_t = f, t
        thresholds[i] = best_t
    return thresholds


def format_report(df, title):
    """Readable console table."""
    lines = [f"\n{'=' * 72}", title, "=" * 72,
             f"{'category':22s} {'prec':>7s} {'rec':>7s} {'F1':>7s} {'n_true':>7s} {'n_pred':>7s}"]
    for _, r in df.iterrows():
        if r["category"] in ("MACRO AVG", "MICRO AVG"):
            lines.append("-" * 72)
        pv = "  n/a  " if pd.isna(r["precision"]) else f"{r['precision']:7.3f}"
        rv = "  n/a  " if pd.isna(r["recall"]) else f"{r['recall']:7.3f}"
        lines.append(f"{r['category']:22s} {pv} {rv} {r['f1']:7.3f} "
                     f"{r['support']:7d} {r['predicted']:7d}")
    return "\n".join(lines)
