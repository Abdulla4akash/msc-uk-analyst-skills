# v4 — Publication-Grade Evaluation Foundation

> **Status:** Internal development evaluation only. The 300 manually annotated postings are treated as **development data** for future work. The future `external_locked_test` (independently collected, independently annotated) **does not exist yet**, so no claim about generalisation to unseen UK job advertisements is made.

This directory rebuilds the classical lexical baselines under a corrected, defensible evaluation protocol. It fixes the methodological issues identified in `v3/base-model/` without improving lexicons or changing gold labels — the goal is a trustworthy baseline, not a higher score.

## Terminology

| Term | Meaning |
|------|---------|
| `internal_tuning` | ~100 of the 300 gold postings (historically `dev`) — used for fitting and threshold tuning |
| `internal_holdout` | ~200 of the 300 (historically `test`) — held-out reporting **after** freezing, **NOT** an external test |
| `external_locked_test` | **RESERVED** for future independently collected dataset — does not exist yet; only this should be called "test" in publication tables |
| `nested cross-validated internal development estimate` | 3 outer folds × 2 inner folds, genuinely nested; each outer validation posting is predicted exactly once with thresholds/IDF from its outer_train only |

If a file or JSON key still exposes `dev`/`test` it is a backward-compat alias; publication-facing output uses `internal_tuning`/`internal_holdout`.

## What was wrong in v3

| Problem | v3 location | Effect |
|---|---|---|
| **TF-IDF fit leakage** — vectoriser fitted on the full 820-posting corpus, including the 200-posting holdout | `v3/base-model/tfidf_baseline.py:59,138` (`cosine_scores(corpus_texts, …)`, `S_full_cos`) | Holdout text influences vocabulary and IDF values |
| **Batch-dependent normalisation** — scores divided by per-category maxima across the evaluation batch (`S / denom` where `denom = S.max(axis=0)`) | `v3/base-model/tfidf_baseline.py:64-66,83-85` | Score for posting X changes if evaluated alongside different postings |
| **Holdout-based model selection** — `best = max(results, key=lambda k: results[k]["test_macro_f1"])` | `v3/base-model/tfidf_baseline.py:137` | Holdout metrics influence which variant's corpus predictions are released |
| **Single 100-posting tuning split + non-nested CV** — no nested threshold isolation | `v3/base-model/evaluate.py:32-41,70-85`; first v4 `tune_thresholds_cv(oof, …)` was still optimistic | Thresholds tuned on the same labels used for final reporting |

See [`EVALUATION_AUDIT.md`](EVALUATION_AUDIT.md) for numbers and why neither v3 nor v4 should be presented as external generalisation.

## What v4 does instead

- **Taxonomy frozen** — `v4/config.py` is an exact copy of `v3/base-model/config.py` at tag `v3-original-baseline` (commit `d7ee030`), with `TAXONOMY_VERSION = "v3-13cat-frozen"`. No import from `v3` at runtime.
- **Inductive TF-IDF** — `TfidfVectorizer` fitted on **outer_train / internal_tuning texts only** (`methods/lexical_baseline.py:fit_tfidf_vectoriser`); validation/holdout texts only transformed.
- **Batch-invariant scores** — cosine returns raw `cosine_similarity`; weighted lexical returns `sum(IDF matched)/sum(IDF lexicon)` — no batch-max division.
- **Genuinely nested threshold selection** — see `evaluation/nested.py`. For each outer fold: fit TF-IDF on outer_train, tune thresholds via **inner 2-fold CV using only outer_train** (StratifiedKFold where feasible; deterministic holdout fallback documented), freeze thresholds, score outer_validation. Outer validation labels never influence thresholds. See data-flow below.
- **Publication-safe split naming** — `internal_tuning` / `internal_holdout` everywhere publication-facing; `external_locked_test` is reserved and never fabricated.
- **Reproducibility** — every run records timestamp, git commit, seed, taxonomy version, split IDs, vectoriser config, package versions, per-outer-fold thresholds + inner strategy, and bootstrap CIs in JSON.

## Data flow (one outer fold, genuinely nested)

```
                         texts / labels
                             │
          ┌──────────────────┼─────────────────────┐
          │ outer_train (200)│                     │ outer_validation (100)
          │                  │                     │  (text+labels invisible until final scoring)
          │        ┌─────────┴─────────┐           │
          │        │ inner CV (k=2)    │           │
          │        │ using ONLY        │           │
          │        │ outer_train       │           │
          │        ├─ inner_train → fit TF-IDF     │
          │        ├─ inner_val   → score          │
          │        └─ tune thresholds              │
          │              │                        │
          │        frozen thresholds (13)          │
          │              │                        │
          │  fit TF-IDF on full outer_train        │
          │  score outer_validation  ──────────────┼──▶ apply frozen thresholds
          │                                        │         │
          │                                        │   store outer predictions
          └────────────────────────────────────────┘         │
                                                         concatenate across 3 outer folds
                                                         → nested-CV internal development estimate
                                                              (each posting predicted exactly once)
                                                              + bootstrap CIs (supplementary)
                                                              + fold-to-fold variation
```

For the single `internal_holdout` (historical dev/test), the analogous flow is simpler: fit on `internal_tuning` (100), tune on `internal_tuning`, score `internal_holdout` (200). Model selection uses `internal_tuning` macro-F1 only.

Enforced structurally in `evaluation/nested.py:run_nested_cv_for_method` and verified by `tests/test_nested_cv.py` (outer-label isolation, outer-text isolation, accounting, reproducibility).

## Structure

```
v4/
  README.md
  EVALUATION_AUDIT.md
  config.py
  evaluation/
    data.py
    splits.py
    nested.py                         genuinely nested CV
    metrics.py
    bootstrap.py
  methods/
    lexical_baseline.py               Baseline 0/1/2
  experiments/
    run_lexical_baseline.py           internal_holdout + nested_cv
  tests/
    test_no_batch_dependence.py
    test_split_isolation.py
    test_threshold_tuning.py
    test_reproducibility.py
    test_nested_cv.py                 outer-label/text isolation, accounting, reproducibility
  results/
```

## Baselines

| Baseline | Fitting | Score |
|---|---|---|
| **0 — unweighted lexical** | no fitting | `(# matched terms) / (lexicon size)` |
| **1 — cosine TF-IDF** | fit on outer_train/internal_tuning | raw cosine posting vs category pseudo-doc |
| **2 — weighted lexical TF-IDF** | fit on outer_train/internal_tuning | `sum(IDF matched)/sum(IDF lexicon)` |

Grid `0…1` (201 pts internal_holdout, 51 pts nested). Fallback for unseen terms: IDF `1.0` in numerator+denominator.

## Running

```bash
pip3 install -r requirements.txt
pip3 install pytest
PYTHONPATH=. python3 v4/experiments/run_lexical_baseline.py --mode both --n_bootstrap 10000
PYTHONPATH=. python3 v4/experiments/run_lexical_baseline.py --mode internal_holdout
PYTHONPATH=. python3 v4/experiments/run_lexical_baseline.py --mode nested_cv --n_outer_splits 3
PYTHONPATH=. python3 -m pytest v4/tests -v
```

Outputs in `v4/results/` (publication-safe names; `dev`/`test` aliases kept for back-compat):

- `v4_lexical_summary.json` — taxonomy, commit, seed, splits, **per-outer-fold thresholds + inner_strategy**, bootstrap CIs
- `v4_*_internal_tuning.csv` / `v4_*_internal_holdout.csv` — per-category reports
- `v4_*_nested_cv_report.csv` — nested estimate
- `v4_*_accounting_*.json` — TP/FP/FN/TN
- `v4_*_predictions_gold_internal_holdout.csv` / `v4_*_nested_cv_predictions.csv`
- `v4_nested_cv_per_fold_macro_f1.csv` — fold variation

## Weighted vs unweighted — ablation hypothesis

Internal development evaluation shows **weighted and unweighted lexical matching are effectively tied** (internal_holdout: 0.938 vs 0.937; nested: 0.934 vs 0.942, see table below). Current evidence suggests most lexical performance comes from lexicon design + whole-word matching + boundary rules + negative-pattern suppression, rather than IDF weighting itself. Formal ablation (`raw match → whole-word → negative patterns → expanded lexicon → IDF → threshold tuning`) is deferred to the next stage.

## Bootstrap clarification

Posting-level bootstrap (percentile, 10k) over fixed nested-CV predictions quantifies uncertainty conditional on those predictions but does **not** fully reproduce training-data uncertainty. Report fold-to-fold variation + nested aggregate as primary; bootstrap as supplementary. For the future `external_locked_test`, bootstrap over that held-out set will be a primary uncertainty estimate.

## What is NOT claimed

- No external generalisation — `external_locked_test` does not exist.
- No lexicon improvement — lexicons frozen to `v3-13cat-frozen`.
