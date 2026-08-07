"""
Bootstrap 95% confidence intervals for v4.

Posting-level resampling (resample rows, not individual label cells).
"""

import numpy as np
from sklearn.metrics import f1_score

from v4.config import CATEGORIES


def bootstrap_ci(y_true, y_pred, metric="macro_f1", n_bootstrap=10000, seed=42, ci=95):
    """
    Posting-level bootstrap CI.

    Parameters
    ----------
    y_true, y_pred : ndarray (n,13)
    metric : str in {"macro_f1","micro_f1","subset_accuracy","hamming_accuracy"}
    n_bootstrap : int
    seed : int
    ci : int (e.g. 95)

    Returns
    -------
    dict with point estimate, lower, upper, method, n_bootstrap, seed
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _metric(yt, yp):
        if metric == "macro_f1":
            return f1_score(yt, yp, average="macro", zero_division=0)
        if metric == "micro_f1":
            return f1_score(yt, yp, average="micro", zero_division=0)
        if metric == "subset_accuracy":
            return float((yt == yp).all(axis=1).mean())
        if metric == "hamming_accuracy":
            return float((yt == yp).mean())
        raise ValueError(f"unknown metric {metric}")

    point = float(_metric(y_true, y_pred))
    vals = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals[b] = _metric(y_true[idx], y_pred[idx])
    alpha = (100 - ci) / 2
    lower = float(np.percentile(vals, alpha))
    upper = float(np.percentile(vals, 100 - alpha))
    return {
        "metric": metric,
        "point": point,
        "lower": lower,
        "upper": upper,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "method": "percentile posting-level bootstrap (resample rows with replacement)",
    }


def bootstrap_all(y_true, y_pred, n_bootstrap=10000, seed=42, ci=95):
    """Convenience: bootstrap for all 4 aggregate metrics."""
    out = {}
    for m in ("macro_f1", "micro_f1", "subset_accuracy", "hamming_accuracy"):
        out[m] = bootstrap_ci(y_true, y_pred, metric=m, n_bootstrap=n_bootstrap, seed=seed, ci=ci)
    return out
