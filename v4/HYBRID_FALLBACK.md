# Hybrid Selective Fallback — Experiment 3 (Internal Development)

**Date:** 2026-08-07  
**Commit before results:** `afffe18` (semantic baselines + fix) — hybrid models frozen per `v4/semantic/model_config.py` and `v4/hybrid/selective_fallback.py` before running `run_hybrid_fallback.py` (models/threshold grids frozen 2026-08-07, no model shopping after)  
**Taxonomy:** `v3-13cat-frozen` (13 categories, frozen; `CATEGORIES` order frozen)  
**Evaluation:** nested 3 outer ×2 inner (leakage-safe, identical outer splits seed 42, stratified `role_family`, dedup), posting-level bootstrap 10 000 (supplementary); `external_locked_test` does not exist  
**Seed:** 42  
**Status:** Pre-registered methods H1/H2 below were frozen BEFORE running hybrid results (per §34). Post-hoc hypotheses H3-H6 below are labelled post-hoc (written AFTER seeing semantic results) and are exploratory, not confirmatory.

## Research question

Does selective, per-category semantic fallback — only when the frozen lexical gate A1 is silent — improve on the strong whole-word lexical baseline, and if so, where?  The gate is asymmetric: `A1=1` can never be vetoed by semantics; semantics may only add positives where `A1=0`. This preserves lexical precision on explicit tool categories while allowing recall on lexically unseen/paraphrased expressions.

## Frozen hybrid definitions (pre-registered before results)

- **A1 gate (frozen):** `v4/ablation/lexical_ablation score_for_variant("A1")` — final frozen `LEXICONS`, whole-word/phrase-safe (`\bterm\b`, case-insensitive), no `NEGATIVE_PATTERNS`, `any-hit >0`.  Binary gate per posting-category.
- **S1 (frozen):** `v4/semantic/supervised_tfidf` TF-IDF vectoriser `lowercase/english/(1,2)/min_df2/max_df0.9/sublinear` → 13 one-vs-rest `LogisticRegression(C∈{0.1,1.0,10.0}, class_weight="balanced", lbfgs, max_iter 1000)`; C and thresholds selected via inner CV optimising **full hybrid** macro-F1 (not S1-only), per Appendix.  Vectoriser+classifier fitted only on `outer_train`/`inner_train`.
- **S3 (frozen):** `typeform/distilbert-base-uncased-mnli` (66 M, 512 tokens, Apache-2.0) with 13 frozen hypotheses `"This job requires {CATEGORY_LABELS[cat]}."`, premise split into sentences/400-token chunks, MAX entailment aggregation, thresholds only; CPU; cached with provenance hash (`v4/results/semantic/s3_nli_scores_cache.npz` + `s3_nli_scores_provenance.json`; hash = SHA256(model_id + hypotheses + chunk_tokens + texts)[:16]; cache invalidates when texts/model change).  See `SEMANTIC_MODEL_SELECTION.md` for BART-large → DistilBERT switch (infeasible runtime, not result-driven).
- **No S2 in hybrid:** `H1` and `H2` only; S2 embedding not used as fallback in this stage.

### Hybrid decision rule (asymmetric, OFF, batch-invariant)

For each posting `i` and category `c` in `CATEGORIES` order:

```
if A1[i,c] == 1:  hybrid[i,c] = 1               # lexical gate cannot be vetoed
else:             hybrid[i,c] = 1 if sem_scores[i,c] >= thr[c] else 0
                  # OFF represented as thr=2.0 (>1) → never fires, so hybrid = A1
```

- `sem_scores` is `S1` probability (LR `predict_proba`) for H1 or `S3` MAX entailment for H2, both in `[0,1]`, batch-invariant.
- `thr[c]` per-category threshold in `GRID = 0..1 step 0.02 (51 points) + OFF(2.0)`.
- Conservative tie-break: when multiple thresholds tie on inner hybrid macro-F1, the **highest** (most conservative) wins, so `OFF` is favoured on ties.
- No cross-posting normalisation; no lexicon use inside S1/S3 beyond the frozen gate.

### Nested tuning protocol (genuinely nested, identical outer folds)

Identical `outer_splits = make_cv_splits(gold_df, texts, n_splits=3, seed=42)` for every method (hash verified equal to `v4/results/ablation/lexical_ablation_summary.json`; assertion fails otherwise).  For each outer fold `k` (`outer_train`≈200, `outer_val`≈100):

1. Compute lexical A1 gates for `outer_train`/`outer_val` via frozen lexicon only.
2. Build `inner_splits` (2-fold `StratifiedKFold` on `role_family`; fallback `KFold` if rare) **using only `outer_train`** (`seed + 1000*k + 7`).
3. **H1:** For each `C` in grid, compute inner OOF semantic scores for `outer_train` (fit TF-IDF+LR on each `inner_train`, score `inner_val`, concatenate OOF), then tune per-category hybrid thresholds on the OOF hybrid (lexical `outer_train` OR semantic OOF) to maximise hybrid per-category F1 (macro = mean; higher wins, tie → higher threshold).  Score hybrid OOF macro-F1 for that `C`; pick `C` with best hybrid macro (first wins on tie).  Freeze `best_C` and `best_thr[13]` (with OFF) from `outer_train` only.
4. **H2:** Semantic scores are frozen NLI cached scores; for each category and each `thr` candidate, compute mean hybrid F1 across inner validation folds (lexical inner_val OR NLI inner_val >= thr), average across 2 inner folds, pick highest-mean thr (tie → highest).  No classifier fitting.
5. Fit S1 on full `outer_train` with `best_C` (H1) or reuse cached NLI scores (H2); score `outer_val` semantically; apply frozen asymmetric rule with `best_thr` to `outer_val` (`A1_val` OR `sem_val >= thr`).  `outer_val` labels/texts never influence thresholds/C/fitting.

Every posting receives exactly one outer prediction (`prediction_count==1`).  A secondary `internal_holdout` 100/200 (audit only) uses the same rule (tune on `internal_tuning` only, score `internal_holdout`) but all selection uses `internal_tuning` macro-F1 only.

## Hypotheses

### Pre-registered (written BEFORE running hybrid results, per §34)

- **H1 — H1 (A1+S1) may modestly improve recall on lexically unseen positives while largely preserving A1 precision, because S1 learns corpus-specific lexical combinations beyond the manual lexicon but is still text-bound.**  
- **H2 — H2 (A1+S3 NLI) may improve recall on unseen/paraphrased stakeholder/soft-skill expressions (e.g., `stakeholder_comm`) but risks adding false positives on explicit tool categories (SQL, Excel, visualisation) due to NLI over-entailment on long adverts.**  
  *(Both are internal-development hypotheses only; no external generalisation claim — `external_locked_test` does not exist.)*

### Post-hoc (written AFTER seeing semantic results in Experiment 2, clearly labelled exploratory)

- **H3 — OFF will be selected for most explicit-tool categories (SQL, Excel, visualisation_bi, programming) because A1 already achieves F1>0.95 and any semantic fallback adds net false positives; OFF is the conservative fallback for high-precision lexicons.**  
- **H4 — Fallback thresholds, if active, will be high (>0.5) and sparse, reflecting the need to be conservative to avoid swamping lexical precision; low thresholds will be strongly disfavoured in inner CV hybrid-F1.**  
- **H5 — The lexically unseen-positive subset (39/1055 ≈3.7%) will drive most hybrid gains, if any; hybrid improvement on seen positives will be near-zero or negative because A1 is already near-perfect on seen (F1 1.00 on seen by construction of seen definition).**  
- **H6 — Because semantic methods are 0.32–0.55 macro-F1 behind lexical on aggregate, any hybrid that is not strongly regularised (OFF + high thresholds) will be worse than A1 on aggregate macro-F1, even if it helps on unseen.**  
  *All H3-H6 are post-hoc, motivated by Experiment 2 semantic gaps, and should be interpreted as exploratory, not confirmatory; they were not used to change the frozen grid or model IDs.*

## Results — primary nested (300, OOF) — placeholder to be filled after `run_hybrid_fallback.py`

*Run:* `PYTHONPATH=. python3 v4/experiments/run_hybrid_fallback.py --n_bootstrap 10000` (debug `--n_bootstrap 100` then final 10k; H2 benefits from NLI cache after first run).  
*Outputs:* `v4/results/hybrid/hybrid_summary.json`, `hybrid_nested_results.csv`, `hybrid_per_fold.csv`, `hybrid_paired_deltas.csv`, `hybrid_seen_unseen.csv`, `hybrid_disagreements.csv`, `hybrid_runtime.csv`; S3 cache `v4/results/semantic/s3_nli_scores_cache.npz` + provenance JSON.

| Method | Type | Nested macro-F1 | 95% CI* | Micro | Subset | Hamming | FP | FN | TP | TN | Ads with error | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | fixed lexical | **0.9429** | [0.926,0.956] | 0.960 | 0.747 | 0.978 | 45 | 39 | 1016 | 2800 | 76 | 0.0s |
| A4/A2 | contextual lexical | 0.9418 | [0.925,0.955] | 0.958 | 0.730 | 0.977 | 42 | 47 | 1008 | 2803 | 81 | 0.0s |
| H1_A1_S1 | selective fallback S1 | **0.9377** | [0.921,0.951] | 0.958 | 0.737 | 0.977 | 50 | 39 | 1016 | 2795 | 76→77* | 8.8s |
| H2_A1_S3 | selective fallback NLI | **0.9429** | [0.926,0.956] | 0.960 | 0.747 | 0.978 | 45 | 39 | 1016 | 2800 | 76 | 603.8s (3.6s cached) |

* H1 adds 5 FP vs A1 (1 per outer fold avg), 0 new FN; no lexical-failure recovery (0/84 wrong→correct). H2 collapses to A1 (all 13 thresholds OFF in all 3 outer folds) → Δ 0.000 [0.000,0.000], 0 wrong→correct, 0 correct→wrong. Paired CIs via posting-level bootstrap 10k seed 42 on identical outer folds.

Paired deltas vs A1 (same 300, same outer folds, same resampled indices 10k seed 42): H1 Δ -0.0052 CI [-0.011,-0.001] w2c 0 c2w 5 net -5; H2 Δ 0.000 CI [0.000,0.000] w2c 0 c2w 0 net 0.  All thresholds nested per outer fold (OFF + conservative tie-break), asymmetric fallback (A1=1 never vetoed).

### Expected conservatism

No hybrid is claimed to beat A1 on aggregate macro-F1 without evidence; given Experiment 2 gaps (S1 -0.32, S3 -0.55 vs A1) and the fact that only 3.7% of positives are lexically unseen (39/1055), the prior is that hybrid will be near A1 or slightly below on aggregate, with any gains concentrated on the unseen subset.  The OFF option and conservative tie-break are explicitly designed to allow the hybrid to collapse to A1 (thr=OFF for all cats) when fallback hurts.

### Regression anchors (must hold)

- A1 nested macro-F1 `0.9429 ±0.002` (anchor from `84ca602`/`d2b7afc`)
- Lexically seen positives `1016` / unseen `39` (3.7%) — using `score_for_variant("A1")>0` as probe (analysis only, not for S1/S3 prediction)
- A2/A4 `0.9418`, A5 `0.9342` reproduced within `0.001`

## Per-category / per-fold / runtime

*To be filled after run.*  Per-category F1 (`hybrid_seen_unseen.csv`), per-fold macro (`hybrid_per_fold.csv`), lexical-failure recovery (where `A1_wrong 84 → hybrid_correct` vs `new errors`), disagreements where `hybrid != A1` with semantic scores and thresholds, OFF rate per category (how many outer folds choose OFF).

## Audits

- **Outer label/text isolation:** `test_hybrid_fallback.py::test_outer_label_isolation_H1/H2` flips/mutates outer val labels/text and asserts thresholds for that fold unchanged (thresholds/C come from `outer_train` inner CV only).  
- **Asymmetric & OFF:** `test_asymmetric_cannot_be_vetoed`, `test_off_option_never_fallback`.  
- **Batch invariance:** `test_batch_invariant`.  
- **Provenance cache:** `test_nli_cache_provenance` checks `s3_nli_scores_cache.npz` hash `SHA256(model_id+hypotheses+chunk_tokens+texts)[:16]` and that second call is cached.  
- **Taxonomy frozen:** `TAXONOMY_VERSION="v3-13cat-frozen"`.

## Limitations

Nested CV is still internal development (300 postings, stratified on `role_family`; rare labels `ethics_governance` 13, `cloud` 40 have high fold variance).  Bootstrap CIs are supplementary (fold variation primary).  No `external_locked_test` — no external generalisation claim.  IDF fallback for 68 lexicon terms missing from 100-dev vocab (weight 1.0) applies to A5 but not to H1/H2 directly.

## How to run

```bash
PYTHONPATH=. python3 v4/experiments/run_hybrid_fallback.py --n_bootstrap 10000   # full (H2 cached after first)
PYTHONPATH=. python3 v4/experiments/run_hybrid_fallback.py --n_bootstrap 100 --skip_s3   # fast H1-only smoke
PYTHONPATH=. python3 -m pytest v4/tests/test_hybrid_fallback.py -v
```

## Provenance

- Lexicon provenance: `v4/LEXICON_PROVENANCE.md` (authentic seed NOT RECOVERABLE → CASE B)
- Semantic model selection: `v4/SEMANTIC_MODEL_SELECTION.md` (S2 `all-MiniLM-L6-v2` 8b3219a, S3 `distilbert-base-uncased-mnli` 0558d89; BART-large infeasible)
- Semantic baselines: `v4/SEMANTIC_BASELINES.md` (S1 0.624, S2 0.512, S3 0.387 vs A1 0.943)
