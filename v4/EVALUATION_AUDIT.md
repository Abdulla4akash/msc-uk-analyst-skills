# v4 Evaluation Audit — v3 vs Corrected Protocol

**Date:** 2026-08-07  
**v3 frozen baseline tag:** `v3-original-baseline` (`d7ee030`)  
**v4 methodological implementation reviewed at:** `4207363`  
**v4 cleanup provenance:** see current `git HEAD` (this document’s content is part of the cleanup commit; the SHA is reported in the final handoff)  
**Taxonomy:** `v3-13cat-frozen`  
**Seed:** 42 · **Bootstrap:** 10,000 posting-level resamples, percentile, seed 42

## 1. Original v3 reported result

From `v3/base-model/tfidf_B_weighted_hit_test.csv` and `v3/base-model/results/tfidf_summary.json`, internal holdout `n=200` (historically called "test"):

| Variant | Macro-F1 | Micro-F1 | Subset acc. | Hamming acc. |
|---|---|---|---|---|
| **B — weighted hit** | **0.937** | **0.962** | **0.755** | **0.980** |
| A — cosine | 0.759 | 0.811 | 0.260 | 0.889 |

Best variant chosen by `test_macro_f1` (line 137).

## 2. Corrected v4 internal-development estimates

### 2.1 Internal holdout (same 100/200 stratified split, inductive, correct name)

Fit on `internal_tuning` (100) only; thresholds on `internal_tuning`; `internal_holdout` (200) reported after freezing. **Not** `external_locked_test`.

| Variant | internal_tuning macro-F1 | internal_holdout macro-F1 | internal_holdout 95% CI (macro) | internal_holdout micro-F1 | subset |
|---|---|---|---|---|---|
| **weighted_lexical_tfidf** | **0.951** | **0.938** | **[0.914, 0.955]** | **0.963** | **0.760** |
| unweighted_lexical | 0.950 | 0.937 | — | 0.962 | 0.755 |
| cosine_tfidf | 0.821 | 0.776 | — | 0.824 | 0.305 |

**Fix 2 interpretation:** weighted (0.938) and unweighted (0.937) are **effectively tied** on internal_holdout — do not declare a winner. See §5/§9 for uncertainty.

Per-category internal_holdout for weighted: programming 0.966, sql 1.000, visualisation_bi 0.986, reporting 0.946, excel 0.991, statistics 0.974, ML 0.882, data_cleaning 0.865, etl 0.947, data_modelling 0.976, cloud 0.727, stakeholder_comm 0.988, ethics_governance 0.947. Accounting: 48 ads with ≥1 error, 673 TP / 22 FP / 30 FN / 1875 TN.

### 2.2 Nested cross-validated internal development estimate (preferred)

**Genuinely nested:** 3 outer folds (max feasible; AE n=3) × 2 inner folds per outer_train, thresholds frozen from inner validation only. Each posting predicted exactly once via its outer validation fold.

| Variant | Nested macro-F1 | 95% bootstrap CI* | Nested micro-F1 | Nested subset | Per-outer-fold macro-F1 |
|---|---|---|---|---|---|
| **weighted_lexical_tfidf** | **0.934** | **[0.917, 0.948]** | **0.950** | **0.687** | **0.947 / 0.938 / 0.893** |
| **unweighted_lexical** | **0.942** | **[0.925, 0.955]** | **0.958** | **0.730** | **0.947 / 0.940 / 0.920** |
| cosine_tfidf | 0.682 | [0.640, 0.714] | 0.749 | 0.180 | 0.636 / 0.637 / 0.720 |

\* Bootstrap over fixed nested predictions — supplementary; fold variation is primary (see bootstrap clarification).

Posterior: weighted and unweighted are **effectively tied** (paired outer-fold differences are within fold variation; unweighted is fractionally higher). This suggests lexicon design + negative-pattern suppression drive most performance, not IDF weighting (hypothesis for next-stage ablation).

## 3. Nested threshold-selection correction

### 3.1 How the old CV worked (pre-fix, optimistic)

1. For each of 3 folds: fit TF-IDF on train fold, score val fold → collect **OOF scores for all 300**.
2. Call `tune_thresholds_cv(oof_scores, y, splits)` which averaged F1 across folds for each grid threshold and picked the best. This **inspected the true labels of all 300** (including every posting on which final performance was reported).
3. Apply those thresholds to the same OOF scores and report `OOF macro-F1`.

Model scores were out-of-fold, but **thresholds were not** — they were optimised using the labels of the observations on which performance was measured.

### 3.2 Was GPT-5.6 Pro's concern confirmed?

**Confirmed.** Inspection of `v4/methods/lexical_baseline.py:tune_thresholds_cv` and `v4/experiments/run_lexical_baseline.py:run_cv` shows the call `tune_thresholds_cv(oof, y, splits)` occurs after OOF concatenation and uses `y` for all rows. The implementation was not structurally isolated.

### 3.3 Why it was optimistic

Thresholds are hyperparameters. Optimising them on the same labels used for evaluation leaks label information. With thresholds near the quantised score grid, a threshold tuned to maximise F1 on the evaluation labels will be at least as good as one tuned blind, inflating the reported metric (small here because thresholds are already near-optimal, but the procedure is still invalid).

### 3.4 How the new nested procedure prevents leakage

Per outer fold `k`:

- `outer_train` (200) → inner 2-fold CV **using only outer_train's texts/labels** (`evaluation/nested.py:run_nested_cv_for_method`)
  - inner_train → fit TF-IDF → score inner_val → accumulate inner OOF scores
  - `thresholds = _tune_via_inner_cv(inner_oof_scores, y_outer_train, inner_splits, grid)`
- **Freeze thresholds**
- Fit TF-IDF on **full outer_train** (200) → score **outer_validation** (100) → `pred = scores >= thresholds`
- Store `pred` as that posting's single nested prediction

At no point does `y_outer_validation` or `texts_outer_validation` influence IDF, thresholds, or model selection for that fold. Enforced by passing only `outer_train` indices/texts/labels into inner-CV and threshold code; tests `test_outer_label_isolation` and `test_outer_text_isolation` fail on the old procedure.

### 3.5 How much did the metric change?

- Weighted: old OOF macro-F1 **0.941** → nested **0.934** (Δ = –0.007). The optimistic gap was small because scores are coarse and thresholds near 0.02 already generalise, but the correction is required.
- Unweighted: old 0.942 → new 0.942 (no IDF fitting, so identical).
- Cosine: old 0.717 → new **0.682** (Δ = –0.035) — larger because cosine thresholds are more sensitive to the fitted vocabulary.

## 4. Why internal_holdout barely changed (and why that does not imply correctness)

Weighted internal_holdout test 0.937→0.938 despite three leakages, because binary presence dominates with thresholds ~0.005, and 68 lexicon terms absent from a 100-doc internal_tuning vocab fall back to IDF `1.0` in both versions. The stability is **not** evidence leakage is harmless — next-stage supervised models could show larger bias.

## 5. What can and cannot be claimed

**Can be claimed:**

- Nested cross-validated internal development estimate: weighted 0.934 [0.917, 0.948], unweighted 0.942 [0.925, 0.955] — effectively tied.
- Internal holdout: weighted 0.938 [0.914, 0.955] — **not** `external_locked_test`.
- No exact/normalised duplicates; no TF-IDF pairs >0.90.

**Cannot be claimed:**

- Generalisation to unseen UK ads. The 300 is development material; `external_locked_test` does not exist.
- That one lexical variant meaningfully wins — difference 0.001–0.008 is within fold variation.
- That CV with `k=3` fully characterises rare categories (ethics_governance n=13).

Use: "nested cross-validated internal development estimate" or "internal holdout performance on the existing annotated corpus."

## 6. Neither result is external generalisation

Both v3 0.937 and v4 0.938/0.934 are internal to the 300 that informed lexicon design. Even nested numbers would be optimistic as out-of-distribution estimates. A future `external_locked_test` (new sampling date, unseen companies, independent annotator) is required.

## 7. Thresholds and details

- Internal holdout weighted: `excel 0.040`, others `0.005` (grid 201).
- Nested weighted per outer fold: `(0.020…0.060 for excel/programming)` vs holdout's `0.005`; see `v4_lexical_summary.json → nested_cv.results.*.outer_fold_info`.

## 8. Remaining threats to validity (before publication)

- Single annotator; no kappa.
- Lexicon-data overlap with the 300.
- Temporal/company/source shift not tested (no company-held-out).
- Possible paraphrased duplicate templates (only exact/normalised + TF-IDF 0.90 checked).
- Small tuning population for rare categories (ethics_governance, cloud).
- Fold variation substantial (weighted 0.893–0.947 across outer folds).

## 9. Ablation hypothesis (Fix 2)

Current evidence: weighted (0.934) and unweighted (0.942) are tied, suggesting most lexical performance comes from carefully designed lexicons + whole-word matching + boundary rules + negative-pattern suppression, rather than IDF weighting. Future ablation: `raw match → whole-word → negative patterns → expanded lexicon → IDF → thresholds` (not run now).

## 10. What must happen before publication

1. Freeze taxonomy/lexicons/prompts (done: `v3-13cat-frozen`).
2. Second-annotator reliability on stratified 60–100 sample, oversampling rare/hard categories.
3. Collect ≥200 natural + ≥100 challenge `external_locked_test` sampled after freeze, independently annotated.
4. Run locked model once on `external_locked_test`; report with CIs and slices; never tune on it.
