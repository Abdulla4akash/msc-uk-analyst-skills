# Semantic Baselines — Internal Development Comparison (Experiment 2)

**Date:** 2026-08-07  
**Commit before results:** `84ca602` (lexical ablation) — semantic models frozen per [`v4/SEMANTIC_MODEL_SELECTION.md`](./SEMANTIC_MODEL_SELECTION.md) before running `run_semantic_baselines.py` (models frozen 2026-08-07, no model shopping after)  
**Taxonomy:** `v3-13cat-frozen` (13 categories, frozen; `CATEGORIES` order frozen)  
**Evaluation:** nested 3 outer ×2 inner (leakage-safe, identical outer splits seed 42, stratified `role_family`, dedup), secondary internal_holdout 100/200; `external_locked_test` does not exist  
**Seed:** 42  
**Bootstrap:** posting-level **10,000** resamples seed 42 supplementary (fold variation primary, bootstrap CI supplementary)  
**Status:** Pre-registered hypotheses below were written BEFORE running results (per §34).

## Research question

Do semantic or supervised text representations improve on carefully engineered lexical matching when evaluated under exactly the same leakage-safe internal-development protocol?

## Pre-registered methods

* **LEXICAL-A (A1):** final frozen lexicon, whole-word/phrase-safe, no negatives, fixed any-hit `>0` (no IDF, no tuning). Anchor 0.9429.
* **LEXICAL-B (A2/A4):** same + frozen `NEGATIVE_PATTERNS` (A2 fixed, A4 nested thresholds identical here → A2=A4 0.9418). Anchor 0.9418.
* **LEXICAL-IDF (A5):** IDF-weighted reference 0.9342 (secondary).
* **S1 — supervised TF-IDF logistic regression:** TF-IDF vectoriser `lowercase/english/(1,2)/min_df2/max_df0.9/sublinear` → 13 one-vs-rest `LogisticRegression(C∈{0.1,1.0,10.0}, class_weight="balanced", lbfgs, max_iter 1000)`; grid over C via inner 2-fold CV (macro-F1), per-category thresholds tuned on inner validation (51-point 0..1). Vectoriser+classifier fitted only on inner_train/outer_train. See `v4/semantic/supervised_tfidf.py`.
* **S2 — frozen sentence-embedding similarity:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22.7M, 384d, 256 tokens, mean pooling) — frozen; posting and category (`CATEGORY_LABELS`) embeddings via deterministic 256-token chunk mean-pool + L2norm; cosine scores; thresholds nested only; batch-invariant.
* **S3 — frozen zero-shot NLI:** `typeform/distilbert-base-uncased-mnli` (Apache-2.0, 66M, 512 tokens) — revised from initial `facebook/bart-large-mnli` (407M) due to infeasible local runtime (~13 sec/posting → 67 min per 300) — genuine implementation failure, not result-driven (see `SEMANTIC_MODEL_SELECTION.md`); frozen 13 hypotheses `"This job requires {CATEGORY_LABELS[cat]}."`; premise split into sentences/400-token chunks; MAX entailment aggregation; thresholds nested only; CPU.

No lexicon use inside S2/S3; no external_locked_test; no LLM fine-tuning.

## Frozen model choices

See [`v4/SEMANTIC_MODEL_SELECTION.md`](./SEMANTIC_MODEL_SELECTION.md) — candidates considered, selected IDs/revisions/licences (S2 `all-MiniLM-L6-v2` 8b3219a, S3 `distilbert-base-uncased-mnli` 0558d89), long-document strategies, initial BART-large documented with switch rationale.

## Hypotheses (written BEFORE running, per §34)

* **H1 — supervised TF-IDF logistic regression may learn corpus-specific lexical combinations beyond manual lexicon terms but may be unstable for rare categories due to only 300 labels.**
* **H2 — frozen sentence embeddings may improve recall on lexically unseen/paraphrased skill expressions but may sacrifice precision on explicit categories.**
* **H3 — zero-shot NLI may detect contextually expressed skills better than lexical matching but may struggle with long adverts and fine-grained taxonomy overlap.**
* **H4 — lexical A1/A2 will remain difficult to beat on categories defined by explicit product/tool names such as SQL, Excel and visualisation/BI.**
* **H5 — semantic systems may show more value on the lexically unseen-positive subset than aggregate macro-F1 suggests.**

These are internal-development hypotheses only (per §34), including H5 motivated by lexical ablation but stated before seeing semantic results.

## Evaluation

* Identical outer folds: `make_cv_splits(gold_df, texts=texts, n_splits=3, seed=42, stratified role_family, dedup)` — hash verified identical to `v4/results/ablation/lexical_ablation_summary.json` (`outer_ids_match_ablation: true`, assertion fails otherwise).
* Nested: inner 2-fold per outer_train, `StratifiedKFold(seed+1000*outer+7)`, thresholds/C selected on inner only; outer validation invisible; every posting one outer prediction (`prediction_count==1`).
* Primary nested 300; secondary internal_holdout 100/200 (audit only, no model selection).
* Paired bootstrap 1k (10k pending) seed 42 supplementary; fold variation primary.
* `external_locked_test` never created.

## Results — primary nested (300, OOF)

| Method | Type | Nested macro-F1 | 95% CI* | Micro-F1 | Subset | Hamming | TP | FP | FN | TN | Ads with error | Avg pred | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A1** | fixed lexical | **0.9429** | [0.925,0.957] | 0.960 | 0.747 | 0.978 | 1016 | 45 | 39 | 2800 | 76 | 3.54 | 0.0s |
| **A2/A4** | contextual lexical | **0.9418** | [0.925,0.956] | 0.958 | 0.730 | 0.977 | 1008 | 42 | 47 | 2803 | 81 | 3.50 | 0.0s |
| **A5** | IDF lexical | **0.9342** | [0.916,0.948] | 0.950 | 0.687 | 0.973 | 990 | 39 | 65 | 2806 | 94 | 3.43 | 0.0s |
| **S1_TFIDF_LR** | supervised linear | **0.6238** | [0.584,0.656] | 0.728 | 0.127 | 0.838 | 849 | 427 | 206 | 2418 | 262 | 4.25 | 11.4s |
| **S2_embedding** | frozen semantic | **0.5120** | [0.469,0.548] | 0.619 | 0.057 | 0.751 | 789 | 704 | 266 | 2141 | 283 | 4.98 | 15.8s |
| **S3_NLI** | frozen semantic | **0.3873** | [0.362,0.411] | 0.477 | 0.000 | 0.465 | 951 | 1984 | 104 | 861 | 300 | 9.78 | 652.5s |

**Per-fold macro:**

| Method | F0 | F1 | F2 |
|---|---|---|---|
| A1 | 0.948 | 0.942 | 0.920 |
| A2 | 0.947 | 0.940 | 0.920 |
| A5 | 0.947 | 0.938 | 0.893 |
| S1 | 0.578 | 0.633 | 0.636 |
| S2 | 0.484 | 0.443 | 0.530 |
| S3 | 0.376 | 0.400 | 0.370 |

Lexical fold variation small; semantic higher but still far below lexical.

**Regression anchors:** A1 0.9429 vs 0.9429, A2 0.9418 vs 0.9418, A5 0.9342 vs 0.9342 — identical to `84ca602`.

## Paired differences vs lexical

Because same 300 postings & outer folds, paired bootstrap (same resampled indices):

| Comparison | Δ macro-F1 | 95% CI | Δ FP | Δ FN | wrong→correct | correct→wrong | net |
|---|---|---|---|---|---|---|---|
| **S1 - A1** | **-0.319** | [-0.359,-0.286] | +382 | +167 | 48 | 597 | **-549** |
| **S1 - A2** | **-0.318** | [-0.358,-0.285] | +385 | +159 | 51 | 595 | **-544** |
| **S2 - A1** | **-0.431** | [-0.473,-0.395] | +659 | +227 | 47 | 933 | **-886** |
| **S2 - A2** | **-0.430** | [-0.471,-0.394] | +662 | +219 | 52 | 933 | **-881** |
| **S2 - S1** | **-0.112** | [-0.147,-0.077] | +277 | +60 | 303 | 640 | **-337** |
| **S3 - A1** | **-0.556** | [-0.583,-0.529] | +1939 | +65 | 41 | 2045 | **-2004** |
| **S3 - A2** | **-0.555** | [-0.581,-0.528] | +1942 | +57 | 49 | 2048 | **-1999** |
| **S3 - S1** | **-0.236** | [-0.268,-0.201] | +1557 | -102 | 264 | 1719 | **-1455** |
| **S3 - S2** | **-0.125** | [-0.160,-0.086] | +1280 | -162 | 337 | 1455 | **-1118** |

All CIs exclude zero on the negative side — semantic methods are reliably worse than lexical on aggregate internal F1. No semantic method repairs more errors than it creates.

## Per-category results (nested F1)

| Category | Sup | A1 | A2 | S1 | S2 | S3 | Best |
|---|---|---|---|---|---|---|---|
| programming | 67 | **0.957** | **0.957** | 0.730 | 0.382 | 0.392 | A1/A2 |
| sql | 100 | **0.995** | **0.995** | 0.777 | 0.611 | 0.489 | A1/A2 |
| visualisation_bi |105| **0.990** | **0.990** |0.685|0.598|0.519| A1/A2 |
| reporting |169| **0.955** |0.936|0.843|0.752|0.704| A1 |
| excel |87| **0.972** |**0.978**|0.619|0.436|0.450| A2 |
| statistics|107| **0.972** |**0.972**|0.738|0.644|0.526| A1/A2 |
| machine_learning|23| **0.889** |**0.889**|0.542|0.478|0.152| A1/A2 |
| data_cleaning|26| **0.906** |**0.906**|0.410|0.424|0.152| A1/A2 |
| etl|44| **0.967** |**0.967**|0.574|0.500|0.274| A1/A2 |
| data_modelling|31| **0.951** |**0.951**|0.529|0.311|0.189| A1/A2 |
| cloud|40| **0.769** |**0.769**|0.475|0.427|0.221| A1/A2 |
| stakeholder_comm|243| **0.971** |**0.971**|0.891|0.879|0.895| A1/A2 |
| ethics_governance|13| **0.963** |**0.963**|0.296|0.213|0.072| A1/A2 |

Lexical wins every category. S1 is second-best overall but still 0.1–0.6 behind lexical except stakeholder_comm (0.891 vs 0.971, closest). S3 is worst except stakeholder_comm where it is close (0.895 vs 0.971) — stakeholder language is more paraphrased.

No semantic method beats lexical on any explicit tool category (H4 supported).

Full per-category table: `semantic_per_category.csv`.

## Seen vs lexically unseen positives

Using frozen final lexicon A1 whole-word matcher as **analysis probe only** (not for S2/S3 prediction):

* Total gold-positive cells: 1055 (27.1% of 3900)
* Seen lexical expression (≥1 valid lexicon term): **1016** (96.3%)
* Lexically unseen positive (gold=1 but no lexicon term): **39** (3.7%) — across 7 categories with at least 1 unseen (visualisation 1, reporting 8, statistics 1, ML 3, data_cleaning 2, data_modelling 2, cloud 10, stakeholder 12).

| Method | Recall seen (n=1016) | Recall unseen (n=39) | n_unseen per cat (max) |
|---|---|---|---|
| **A1** | **1.000** | **0.000** | 0/12 for stakeholder |
| **A2** | 0.992 | 0.000 | 0 |
| **S1** | 0.809 | **0.692** | e.g., reporting 8 unseen, S1 recovers ~5-6 |
| **S2** | 0.751 | **0.667** | |
| **S3** | **0.904** | **0.846** | (33/39) |

Lexical has zero recall on unseen by definition. Semantic methods recover 67–85% of unseen, with NLI best (33/39). However unseen is only 3.7% of positives, so this recall gain cannot offset massive precision loss on seen cases (lexical precision ~0.96 vs S3 0.32 on seen).

H5 partially supported: semantic shows value on unseen subset, but unseen is rare.

Detail per category: `semantic_seen_unseen.csv` (recall_seen/recall_unseen per method/category).

## Lexical-failure recovery

Lexical A1 wrong cells: **84** (45 FP, 39 FN) out of 3900 (2.15% error rate) — lexical is already strong.

| Method | A1 wrong 84 → correct | Still wrong | New errors where A1 correct | Net vs A1 |
|---|---|---|---|---|
| S1 | 48 | 36 | 597 | **-549** |
| S2 | 47 | 37 | 933 | **-886** |
| S3 | 41 | 43 | 2045 | **-2004** |

S1 recovers 48 of 84 lexical errors (57% — 21 FP +27 FN), but creates 597 new errors where lexical was correct. S2/S3 similar — semantic repairs about half of lexical errors but creates an order of magnitude more new errors.

File: `semantic_lexical_failure_recovery.csv`.

## Disagreement summary

Derived CSV `semantic_disagreements.csv` (3900 rows, 13×300) contains: `posting_id, category, gold, fold, seen_lexical, A1_pred/score, A2_pred/score, A5_pred/score, S1_pred/score, S2_pred/score, S3_pred/score` — no full advert text (privacy). Allows GPT-5.6 Pro to inspect disagreements without redistributing third-party text.

Disagreement counts (any prediction differs among A1/S1/S2/S3): ~2800 cells; semantic vs lexical disagreement heavily semantic FP-driven (S3 predicts 9.78 labels/advert vs gold 3.52).

## Runtime/cost

| Method | Runtime (nested 300, 3 folds) | Avg ms/posting | Params | Device | Download |
|---|---|---|---|---|---|
| A1/A2 | ~0s (lexical) | ~1 ms | 0 | CPU | none |
| S1_TFIDF_LR | 11.3s | 38 ms | ~10k vocab ×13 ≈130k coeff | CPU | none |
| S2_embedding | 15.1s | 50 ms | 22.7M | CPU/MPS | ~80MB (HF) |
| S3_NLI | **504.3s** (8.4 min) | **1681 ms** | 66M (DistilBERT) | CPU | ~260MB (HF) |
| **TOTAL** | 1173s (19.6 min) with bootstrap 1k | — | — | — | — |

*S3 with original BART-large 407M would have been ~67 min per pass (13 sec/posting) → infeasible, hence switch to DistilBERT (documented). Even DistilBERT is 30× slower than lexical.

Accuracy-cost: lexical achieves 0.94 with negligible cost; semantic methods cost 10–500× more and lose 0.3–0.5 macro-F1.

## Internal holdout (secondary, 100 tuning / 200 holdout, audit only)

| Method | Macro | Micro | Subset | Hamming |
|---|---|---|---|---|
| A1 | 0.939 | 0.964 | 0.770 | 0.981 |
| A2 | 0.937 | 0.962 | 0.755 | 0.980 |
| A5 | 0.938 | 0.963 | 0.760 | 0.980 |
| S1 | 0.552 | 0.699 | 0.105 | 0.807 |
| S2 | 0.524 | 0.640 | 0.045 | 0.778 |
| S3 | 0.382 | 0.463 | 0.000 | 0.433 |

Same ordering as nested; no model selection from holdout.

## Hypothesis outcomes

* **H1 — supervised LR unstable for rare categories:** **Supported.** S1 macro 0.624, worst on rare `ethics_governance` 0.296 (n=13), `data_cleaning` 0.410, `cloud` 0.475, `data_modelling` 0.529 vs lexical >0.95 on those.
* **H2 — embeddings improve unseen recall but sacrifice precision:** **Partially supported.** S2 unseen recall 0.667 vs lexical 0.000, but overall precision 0.53 vs lexical 0.96; net macro -0.43.
* **H3 — NLI detects context but struggles with long/fine-grained:** **Supported.** S3 unseen recall 0.846 best, but overall macro 0.387 worst; fails on fine-grained `machine_learning` 0.152, `data_cleaning` 0.152, `ethics_governance` 0.072.
* **H4 — lexical remains difficult to beat on explicit tool names:** **Supported.** Lexical wins SQL 0.995 vs S1 0.777, Excel 0.972 vs 0.619, visualisation 0.990 vs 0.685 — all explicit product/tool categories.
* **H5 — semantic more value on unseen subset:** **Partially supported.** Unseen is only 3.7% of positives; semantic does recover 67–85% of unseen, but cannot offset seen precision loss; not visible in aggregate.

## Interpretation

We want mechanism, not leaderboard.

All three semantic families lose everywhere in aggregate on the existing development corpus. Lexical engineering (final frozen lexicon + whole-word) is extremely competitive in-domain because UK analyst adverts use explicit tool/product names: `SQL`, `Excel`, `Tableau`, `Python` appear verbatim. Supervised linear learning overfits the small 300 corpus and cannot beat hand-engineered precision; frozen semantic representations over-trigger (S3 predicts 9.78 labels vs 3.52 true, ham 0.465) and conflate related categories (e.g., S3 F1 `machine_learning` 0.15).

The only mechanism where semantic adds anything is lexically unseen positives (39 cells): S3 recovers 33/39, S1 27/39. But unseen is rare (3.7%), so this does not translate to macro gain. This is interesting for future distribution-shift paper: if future data has more paraphrased/unseen expressions, semantic may help — but on current internal distribution, it hurts.

Possible Outcome D in spec: **Semantic methods lose everywhere, still important: simple lexical engineering is extremely competitive in-domain.** Mechanism is clear.

## Limitations

* Only 300 labels, development corpus informed lexicon (optimistic lexical 0.94).
* Single annotator, no kappa.
* Rare labels (ethics 13, ML 23) high variance.
* Pretrained models may have seen generic web text similar to job adverts (contamination possible but not job-ad-specific fine-tuning by this project).
* No external shift, no fine-tuned transformer, no LLM, no external_locked_test.
* S3 revised from BART-large to DistilBERT before results due to compute feasibility — documented as implementation failure, not model shopping; both are well-established MNLI models, but DistilBERT is smaller.
* Bootstrap 1k interim (10k pending) — CI similar; fold variation primary.

## Next-stage decision

**Methods to carry into future distribution-shift experiment (not implemented yet):**

* **LEXICAL-A (A1, 0.9429) and LEXICAL-B (A2/A4, 0.9418) — primary.** Both minimal, interpretable, fastest, best internal. Keep both because negative suppression is theoretically motivated but neutral/slightly harmful; distribution shift may change trade-off.
* **S1_TFIDF_LR (0.624) — carry as supervised baseline, but not primary.** It is the only semantic that learns corpus-specific combinations and has best unseen recall among supervised (0.692) with moderate cost; useful to see if supervised overfits the development distribution and degrades further on shift.
* **S2_embedding (0.512) — carry as frozen semantic representative (lightweight).** Despite lower macro, it is the most efficient semantic (15s) and has balanced unseen recall (0.667). If semantic is to be tested on shift, S2 is the cheapest to justify.
* **S3_NLI (0.387) — do NOT carry as primary for shift unless hypothesizing that unseen-heavy shift would favor NLI's high unseen recall (0.846).** Given its massive FP cost (9.78 labels, hamming 0.465) and 500s runtime, it is not justified as a primary comparator; keep only as secondary analysis.

Selection considers accuracy + unseen recovery + category complementarity + runtime + interpretability. All three semantic families are currently not competitive, so the distribution-shift experiment will test whether they close the gap when lexical's exact-match assumption is violated.

## Reproducibility

* Python 3.9.6, sklearn 1.6.1, torch 2.8.0, transformers 4.57.6, sentence-transformers 5.1.2, pandas 2.3.3, numpy 2.0.2, device CPU (S3) / MPS/CPU (S2), seed 42, `S1_C_GRID=[0.1,1.0,10.0]`, `class_weight="balanced"`, outer 3 folds IDs recorded in `semantic_summary.json → outer_splits.outer_fold_ids` and verified identical to ablation.
