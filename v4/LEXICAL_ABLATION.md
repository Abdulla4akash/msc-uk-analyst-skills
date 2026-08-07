# Controlled Lexical Ablation — Internal Development Estimates

**Date:** 2026-08-07  
**Commit:** see current `git HEAD` (ablation code committed with this document)  
**Taxonomy:** `v3-13cat-frozen` (13 categories, frozen)  
**Lexicon provenance:** `v4/LEXICON_PROVENANCE.md` — authentic seed **NOT RECOVERABLE**, so ablation follows **CASE B** (final frozen lexicon throughout; A3 omitted)  
**Evaluation:** nested 3 outer × 2 inner (genuinely nested, same 3 outer splits seed 42 for every variant); secondary internal_holdout 100/200; bootstrap 10k posting-level paired; `external_locked_test` does not exist  
**Seed:** 42

## Research question

Which components of the lexical pipeline actually explain the approximately 0.94 nested internal-development macro-F1?

Specifically: how much of performance comes from lexicon coverage, whole-word matching, homonym/negative-pattern suppression, threshold tuning, vs TF-IDF weighting — when each component is changed in isolation.

## Hypotheses (stated before interpreting results)

- **H1 — Whole-word matching:** will improve precision by reducing substring false positives (e.g. `excel` inside `excellent`, `sql` inside `nosql`), thus raising macro-F1.
- **H2 — Negative-pattern suppression:** frozen `NEGATIVE_PATTERNS` will reduce false positives in known ambiguous categories (`excel`/`excellence`, `reporting`/`reports to`, `etl`/`product pipeline`, `programming`/`clinical coding`, `cloud`/`Sales Cloud`, `sql`/`Hyperion`), modestly improving or preserving macro-F1.
- **H3 — Lexicon expansion:** if provenance permitted, expansion would raise recall with possible precision cost. Because provenance is not identifiable, H3 is **not tested**.
- **H4 — Per-category threshold tuning (nested):** will alter precision–recall trade-off vs fixed any-hit; expected modest change given already high any-hit baseline.
- **H5 — IDF weighting:** will provide little or no improvement over tuned unweighted lexical (motivated by prior internal tied baselines 0.934 vs 0.942), not a pretended a-priori hypothesis.

## Method — CASE B cumulative ablation

Final frozen lexicon from `v4/config.py` used throughout (do not edit `v4/config.py`; ablation lexicon would have been in `v4/ablation/lexicon_versions.py` if needed — not needed).

| Variant | Lexicon | Matching | Negatives | Decision | IDF |
|---------|---------|----------|-----------|----------|-----|
| **A0** | final frozen | **case-insensitive substring** | no | **any-hit `>0`** (not `>=0`) | no |
| **A1** | final frozen | **whole-word / phrase-safe** (`\bterm\b` case-insensitive) | no | any-hit `>0` | no |
| **A2** | final frozen | whole-word | **frozen `NEGATIVE_PATTERNS`** (mask before matching) | any-hit `>0` | no |
| **A3** | — | — | — | — | — | **OMITTED — NOT IDENTIFIABLE FROM PROVENANCE** |
| **A4** | final frozen | whole-word | frozen patterns | **continuous unweighted score** = `#matched / lexicon_size` + **genuinely nested per-category thresholds** (inner 2-fold CV per outer_train, 51-point grid 0..1) | no |
| **A5** | final frozen | whole-word | frozen patterns | nested thresholds | **inductive IDF** (`TfidfVectorizer` fitted on inner/outer train only) — `sum(IDF matched)/sum(IDF lexicon)` |

A0–A3: no fitted statistics, no learning, no threshold tuning — score/predict outer validation directly via any-hit.
A4/A5: reuse `evaluation/nested.py:run_nested_cv_for_method` (outer_train fit, inner CV thresholds, outer val invisible).

Every posting receives exactly one prediction per variant per outer fold (checked via `prediction_count==1`). Outer splits identical for all variants (hash/IDs recorded in `lexical_ablation_summary.json → outer_splits`).

## Lexicon provenance

See `v4/LEXICON_PROVENANCE.md` for full evidence. Closest historical artefact is `V1/annotation_scheme_v3.xlsx` / `Tier1_Shortlist` column “Tier 2 lexicon … extend during pilot”, but it defines 11 categories (plus `big data`, `business intelligence`) and has zero terms for `stakeholder_comm`/`ethics_governance`; a deterministic seed→final diff is not computable without invented mappings. Therefore A3 is **not identifiable** and is omitted — A2→A4 transition combines no lexicon change (same final lexicon).

## Evaluation

- Primary nested: 3 outer folds (stratified on `role_family`, deduplicated) × 2 inner folds per outer_train (StratifiedKFold k=2, seed `42 + 1000*outer + 7`); outer 300 are development material.
- Secondary internal_holdout: 100 `internal_tuning` / 200 `internal_holdout` (same seed) — audit comparability only.
- Paired bootstrap for Δ macro-F1: same resampled indices for both methods, 10k resamples, seed 42, percentile 95% — supplementary; fold variation primary.
- No `external_locked_test` — never fabricated, never called “test”.

## Primary results — nested (primary internal development estimate)

| Variant | Macro-F1 | 95% CI | Micro-F1 | Subset acc. | Hamming acc. | TP | FP | FN | TN | Ads with error | Avg labels pred |
|---------|---------:|--------|---------:|------------:|-------------:|---:|---:|---:|---:|---------------:|----------------:|
| **A0** substring | 0.831 | [0.809,0.849] | 0.833 | 0.197 | 0.895 | 1026 | 382 | 29 | 2463 | 241 | 4.69 |
| **A1** whole-word | **0.943** | [0.926,0.956] | 0.960 | 0.747 | 0.978 | 1016 | 45 | 39 | 2800 | 76 | 3.54 |
| **A2** +negatives | **0.942** | [0.925,0.955] | 0.958 | 0.730 | 0.977 | 1008 | 42 | 47 | 2803 | 81 | 3.50 |
| **A4** +thresholds | **0.942** | [0.925,0.955] | 0.958 | 0.730 | 0.977 | 1008 | 42 | 47 | 2803 | 81 | 3.50 |
| **A5** +IDF | **0.934** | [0.917,0.948] | 0.950 | 0.687 | 0.973 | 990 | 39 | 65 | 2806 | 94 | 3.43 |

Per-outer-fold macro:

| Variant | Fold 0 | Fold 1 | Fold 2 |
|---------|-------:|-------:|-------:|
| A0 | 0.838 | 0.816 | 0.833 |
| A1 | 0.942 | 0.945 | 0.942 |
| A2 | 0.942 | 0.940 | 0.942 |
| A4 | 0.942 | 0.940 | 0.942 |
| A5 | 0.947 | 0.938 | 0.893 |

Regression anchors: A4 reproduces `unweighted_lexical` nested 0.942; A5 reproduces `weighted_lexical_tfidf` nested 0.934 (both within 0.000 of approved anchors).

## Secondary results — internal holdout (100/200, audit comparability)

| Variant | Macro-F1 | Micro-F1 | Subset | Hamming |
|---------|---------:|---------:|-------:|--------:|
| A0 | 0.820 | 0.822 | 0.195 | 0.887 |
| A1 | 0.938 | 0.957 | 0.755 | 0.978 |
| A2 | 0.937 | 0.956 | 0.740 | 0.977 |
| A4 | 0.937 | 0.956 | 0.740 | 0.977 |
| A5 | 0.938 | 0.963 | 0.760 | 0.980 |

Same pattern: whole-word dominates, negatives/thresholds/IDF marginal.

## Incremental contributions — nested, paired

| Transition | Component added | Macro before | Macro after | Δ macro | Δ macro 95% CI | Δ FP | Δ FN |
|------------|-----------------|-------------:|------------:|--------:|---------------:|-----:|-----:|
| **A0→A1** | whole-word matching | 0.831 | **0.943** | **+0.112** | [+0.099,+0.126] | **–337** | +10 |
| **A1→A2** | negative suppression | 0.943 | 0.942 | **–0.001** | [–0.003,+0.001] | –3 | +8 |
| **A2→A3** | lexicon expansion | — | — | — | — | — | — | **NOT IDENTIFIABLE** |
| **A2→A4** | threshold tuning (unweighted) | 0.942 | 0.942 | **0.000** | [0.000,0.000] | 0 | 0 |
| **A4→A5** | IDF weighting | 0.942 | **0.934** | **–0.008** | [–0.012,–0.003] | –3 | **+18** |

- A0→A1 is the dominant effect (+11.2 pts macro-F1, FP 382→45).
- A1→A2: Δ macro-F1 ≈ –0.0011 (3 wrong→correct, 8 correct→wrong, net –5 corrected cells) — frozen negative-pattern suppression is theoretically motivated for known homonyms (reporting/`reports to`, excel/`excellence`, etl/`product pipeline`, programming/`clinical coding`, cloud/`Sales Cloud`, sql/`Hyperion`), but on the existing development corpus it is approximately **neutral to slightly harmful overall, driven mainly by reporting suppressions that remove genuine positives** (see audit); it is not costless.
- A2→A4 does nothing — any-hit already at optimal thresholds (all outer thresholds 0.02).
- A4→A5 is negative (CI entirely negative).

Per-category incremental F1 table is in `lexical_ablation_per_category.csv`.

## Per-category effects (nested)

Largest gains A0→A1:

| Category | A0 F1 | A1 F1 | Δ |
|----------|------:|------:|---:|
| excel | 0.611 | 0.972 | **+0.362** |
| etl | 0.524 | 0.967 | **+0.443** |
| reporting | 0.745 | 0.955 | **+0.210** |
| machine_learning | 0.548 | 0.889 | **+0.341** |
| programming | 0.887 | 0.957 | +0.070 |
| sql, visualisation, cloud, stakeholder etc. smaller |

A1→A2 per-category: reporting –0.020 (negative patterns incorrectly suppress 8 TP), excel +0.005 (1 correct FP removal), others ≈0.

A4→A5 (IDF): excel –0.078 (15 FN added), etl –0.011, sql –0.005, visualisation –0.005; no category significantly improves.

## Whole-word boundary audit — A0→A1

| Category | n_changed | FP removed | TP lost | FP added | TP gained |
|----------|----------:|----------:|--------:|---------:|----------:|
| programming | 11 | 11 | 0 | 0 | 0 |
| sql | 0 | 0 | 0 | 0 | 0 |
| visualisation_bi | 1 | 0 | 1 | 0 | 0 |
| **reporting** | **114** | **107** | 7 | 0 | 0 |
| **excel** | **106** | **106** | 0 | 0 | 0 |
| statistics | 3 | 2 | 1 | 0 | 0 |
| **machine_learning** | **28** | **28** | 0 | 0 | 0 |
| data_cleaning | 0 | 0 | 0 | 0 | 0 |
| **etl** | **77** | **77** | 0 | 0 | 0 |
| data_modelling | 0 | 0 | 0 | 0 | 0 |
| cloud | 6 | 5 | 1 | 0 | 0 |
| stakeholder_comm | 1 | 1 | 0 | 0 | 0 |

Overall: 337 FP removed for 10 FN introduced (net 327 posting×category errors corrected). Representative substring triggers removed: `excel` inside `excellence/excellent`, `reporting`/`reports` substrings in general text, `pipeline` substrings, `ml` inside words, etc. No rewriting after inspection.

## Negative-pattern suppression audit — A1→A2

| Category | Suppressed matches | Correct FP removals | Incorrect TP removals | Net errors reduced |
|----------|-------------------:|--------------------:|----------------------:|-------------------:|
| excel | 1 | 1 | 0 | **+1** |
| reporting | 10 | 2 | 8 | **–6** |
| others (sql, programming, etl, cloud, etc.) | 0 | 0 | 0 | 0 |

Only `reporting`’s `reports to` / `direct reports` patterns fired (10 postings); 8 were gold positives where reporting skill was genuinely present but phrasing overlapped suppression pattern — net +5 errors. Frozen patterns were **not** tuned after seeing this; audit is for interpretation only.

Detail CSV: `lexical_ablation_negative_detail.csv` (posting_id, category, gold, suppressed — no full advert text).

## Lexicon-expansion audit

**NOT ESTIMABLE FROM AVAILABLE PROVENANCE** — see `LEXICON_PROVENANCE.md`. The only historical lexicon (`V1/annotation_scheme_v3.xlsx` Tier1_Shortlist) defines 11 categories and misses `stakeholder_comm`/`ethics_governance` entirely; deterministic seed→final diff cannot be computed without invented mappings.

## Threshold-tuning audit — A2→A4

For all 13 categories, nested thresholds were `0.02` on every outer fold (3/3). Transition A2 (any-hit, which is `score>0` equivalent to threshold `0`) to A4 (threshold `0.02`) changed **zero** predictions:

| Category | outer0 | outer1 | outer2 | FP removed | FN added | FP added | TP recovered |
|----------|--------|--------|--------|------------|----------|----------|--------------|
| all 13 | 0.02 | 0.02 | 0.02 | 0 | 0 | 0 | 0 |

Any-hit already operates at the optimal boundary for these discrete lexicon-fraction scores; nested tuning does not move the decision.

## IDF-weighting audit — A4→A5

- Macro-F1: 0.942 → 0.934 (Δ –0.008, CI entirely negative; paired bootstrap excludes zero on the negative side).
- Per-category F1: no category improves; excel worsens –0.078 (15 FN: 15 excel postings lost when IDF down-weights rare `excel` hits below 0.02/0.04 thresholds), others 0 or –0.005…–0.011.
- Predictions changed: 21 posting×category cells, 18 of which are new FN, 3 are FP removals (net –15).
- Fold variation: A5 per-fold 0.947/0.938/0.893 — fold 2 drops substantially, indicating IDF weighting is less stable.

Current evidence: IDF weighting **hurts** internal performance vs tuned unweighted.

## Error-transition analysis — nested posting×category cells (3900)

| Transition | wrong→correct | correct→wrong | unchanged correct | unchanged wrong | net corrected |
|------------|--------------:|--------------:|------------------:|----------------:|--------------:|
| **A0→A1** | **337** | 10 | 3479 | 74 | **+327** |
| **A1→A2** | 3 | 8 | 3808 | 81 | **–5** |
| **A2→A4** | 0 | 0 | 3811 | 89 | 0 |
| **A4→A5** | 3 | 18 | 3793 | 86 | **–15** |

Only A0→A1 genuinely repairs errors; A1→A2 is neutral to slightly harmful (net –5), A2→A4 identical, A4→A5 net-negative — negative-pattern suppression is not costless (see correction above).

## Regression anchors

- A4 nested macro 0.9418 ≈ approved `unweighted_lexical` 0.942 (diff 0.0002, floating-point only)
- A5 nested macro 0.9342 ≈ approved `weighted_lexical_tfidf` 0.934 (diff 0.0000)
- Internal holdout anchors (A4 0.937, A5 0.938) also reproduced

If anchors had not reproduced, investigation would have been required (split/threshold/lexicon/score mismatch) — not needed.

## Limitations

- The 300 postings **informed lexicon development** (corpus-validated) — even nested estimates are optimistic vs `external_locked_test`.
- Single annotator, no second-annotator kappa.
- Rare labels (`ethics_governance` n=13, `machine_learning` 23, `data_cleaning` 26) — thresholds and per-category F1 have high variance.
- No `external_locked_test` — this ablation explains **internal** behaviour only, not external generalisation.
- Low outer/inner fold count (3×2) due to `analytics_engineering` n=3 — limits role-family balance granularity.
- Lexicon expansion cannot be isolated (see provenance).
- This ablation precedes any semantic model — do not infer lexical vs semantic superiority.

## Conclusion

The approximately 0.94 nested internal-development macro-F1 is **not** produced by sophisticated weighting or threshold tuning. It comes almost entirely from **careful lexical engineering** — final frozen lexicon coverage combined with **whole-word / phrase-safe matching** (A0→A1 alone contributes +0.112 macro-F1 by removing 337 false positives). Frozen negative-pattern suppression, genuinely nested threshold tuning, and inductive IDF weighting contribute **negligible or negative** incremental value on the current internal corpus (A1→A2 –0.001, A2→A4 0.000, A4→A5 –0.008).

## Next research implication

For future comparisons retain **BOTH** — **LEXICAL-A (A1):** whole-word, final frozen lexicon, no negative suppression, fixed any-hit (≈0.9429) and **LEXICAL-B (A2/A4):** whole-word, frozen negative suppression, effectively equivalent to A4 here because nested thresholding changed zero predictions (≈0.9418) — also retain **A5 (IDF-weighted, ≈0.9342)** as the historical weighted lexical reference where useful, but do not treat it as the primary lexical method. Frozen negative-pattern suppression is theoretically motivated for known homonyms, but on the existing development corpus it is approximately neutral to slightly harmful overall, driven mainly by reporting suppressions that remove genuine positives — it is not costless. Do not retune the negative patterns. Previously stated next-step wording is superseded by this correction.

Do not begin semantic models until GPT-5.6 Pro reviews this ablation.
