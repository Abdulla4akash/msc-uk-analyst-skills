"""
Agreement and adjudication tooling for external double annotation.

Implements per-category metrics for overlap (100 postings) after human A/B labels exist.
Do NOT fabricate results; calculate after human labels supplied.

Metrics per category:
- raw agreement
- Cohen's kappa
- positive agreement, negative agreement
- prevalence
- annotator-A positive count, annotator-B positive count

Also macro mean, micro label agreement, posting-level exact agreement.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

from v4.config import CATEGORIES

def per_category_agreement(y_a: np.ndarray, y_b: np.ndarray):
    """
    y_a, y_b: (n_overlap, 13) int 0/1 blind double annotations
    Returns DataFrame per category with agreement stats.
    """
    n = y_a.shape[0]
    rows = []
    for j, cat in enumerate(CATEGORIES):
        a = y_a[:, j]
        b = y_b[:, j]
        # raw agreement
        raw = float((a == b).mean())
        # kappa (if both constant, sklearn returns nan or 0)
        try:
            kappa = float(cohen_kappa_score(a, b))
        except Exception:
            kappa = float("nan")
        # positive agreement: 2*TP / (2*TP + FP + FN) where A is "reference"? Use counts where both positive
        tp = int(((a==1) & (b==1)).sum())
        fp = int(((a==0) & (b==1)).sum())  # B positive, A negative
        fn = int(((a==1) & (b==0)).sum())
        tn = int(((a==0) & (b==0)).sum())
        # positive agreement (also Dice)
        denom_pos = 2*tp + fp + fn
        pos_agree = float(2*tp / denom_pos) if denom_pos>0 else float("nan")
        denom_neg = 2*tn + fp + fn
        neg_agree = float(2*tn / denom_neg) if denom_neg>0 else float("nan")
        prevalence = float((a==1).mean())  # could also average of both, but report A prevalence + both
        rows.append({
            "category": cat,
            "n_overlap": int(n),
            "raw_agreement": raw,
            "cohen_kappa": kappa,
            "positive_agreement": pos_agree,
            "negative_agreement": neg_agree,
            "prevalence_A": float((a==1).mean()),
            "prevalence_B": float((b==1).mean()),
            "prevalence_mean": float(((a==1).mean() + (b==1).mean())/2),
            "A_pos": int((a==1).sum()),
            "B_pos": int((b==1).sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
    df = pd.DataFrame(rows)
    return df

def aggregate_agreement(y_a: np.ndarray, y_b: np.ndarray):
    """
    Macro mean agreement, micro label agreement, posting-level exact.
    """
    per_cat = per_category_agreement(y_a, y_b)
    macro_raw = float(per_cat["raw_agreement"].mean())
    # micro: flatten
    micro_raw = float((y_a == y_b).mean())
    # posting-level exact: row all 13 equal
    posting_exact = float((y_a == y_b).all(axis=1).mean())
    # macro kappa (mean of per-cat kappa, ignoring nan)
    macro_kappa = float(per_cat["cohen_kappa"].mean(skipna=True))
    return {
        "macro_mean_raw_agreement": macro_raw,
        "macro_mean_kappa": macro_kappa,
        "micro_label_agreement": micro_raw,
        "posting_level_exact_agreement": posting_exact,
        "per_category": per_cat
    }

def make_adjudication_sheet(y_a: np.ndarray, y_b: np.ndarray, posting_ids, out_path):
    """
    Create adjudication worksheet for disagreement cells.
    No model predictions; blank adjudicated label and note.
    """
    rows = []
    for i, pid in enumerate(posting_ids):
        for j, cat in enumerate(CATEGORIES):
            if y_a[i,j] != y_b[i,j]:
                rows.append({
                    "posting_id": pid,
                    "category": cat,
                    "Annotator_A": int(y_a[i,j]),
                    "Annotator_B": int(y_b[i,j]),
                    "adjudicated_label": "",
                    "adjudication_note": ""
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote adjudication sheet with {len(df)} disagreements to {out_path}")
    return df

# Example usage (not executed now, for future after human labels):
#   gold_df_A, y_A, _ = load...
#   gold_df_B, y_B, _ = load...
#   per_cat = per_category_agreement(y_A, y_B)
#   agg = aggregate_agreement(y_A, y_B)
