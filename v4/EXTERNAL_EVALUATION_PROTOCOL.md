# External Evaluation Protocol — Locked

**Status:** Pre-registration, methods frozen before external labels. No external model evaluation has occurred.

**Freeze tag:** `v4-preexternal-freeze` (on commit where `v4/EXTERNAL_FREEZE_MANIFEST.json` created with `EXTERNAL_LABELS_ACCESSED=false`)

**Development data:** 300 annotated postings from 820-corpus LinkedIn snapshot (2024-01-12 to 2024-01-17), taxonomy `v3-13cat-frozen`.

**External data:** Not yet collected; two locked sets E1 (natural, N=200) and E2 (challenge, N=100) to be acquired after freeze.

---

## 1. Research Questions

### Primary
> How do transparent lexical, supervised linear, and frozen semantic skill-coding methods behave on genuinely external UK analyst job advertisements collected after the development process, including natural distribution shift and deliberately difficult lexical-coverage cases?

### Secondary
> Do semantic methods recover more lexically unseen skill expressions under external distribution shift, even if their in-domain aggregate precision is lower?

The external stage is the **first independent confirmation stage** (not a model-selection stage).

---

## 2. Hypotheses

**Primary (pre-registered):**
> The lexical system's extremely high internal nested performance (A1 ≈0.9429 macro-F1) will **fall** when terminology, source, company, template, and time distributions change (source shift + temporal shift).

**Secondary:**
> Semantic methods (S1 supervised TF-IDF LR, S2 frozen embedding, S3 zero-shot NLI) will show higher recall on **lexically unseen** skill expressions (no frozen A1 term) under external shift, even if aggregate precision remains lower. If external unseen-positive rate increases from internal 39/1055=3.7%, semantic relative value may increase.

Both are directional but evaluation is two-sided; no p-hacking.

---

## 3. Methods Frozen (deployment configs fitted on all 300 dev postings, evaluation via nested CV was unbiased estimate)

| Group | ID | Method | Key freeze |
|-------|----|--------|------------|
| **Primary** | **P1** | **A1 lexical** | Whole-word/phrase-safe, final frozen lexicon, no negative suppression, fixed any-hit (>0). No training. Hashes locked. |
| | **P2** | **A2 lexical** | A1 + frozen NEGATIVE_PATTERNS suppression, fixed any-hit. Separate because suppression value may change under shift. |
| | **P3** | **S1 supervised TF-IDF LR** | Exact Experiment-2 definition: `TfidfVectorizer(sublinear, english, 1-2grams, min_df2, max_df0.9)` + `LogisticRegression(C∈{0.1,1.0,10.0}, balanced, lbfgs)`; inner 2-fold CV to select C, thresholds on OOF, refit on all 300. |
| | **P4** | **S2 frozen embedding** | `sentence-transformers/all-MiniLM-L6-v2`, frozen, mean-pool 256-token chunks, category embeddings from `CATEGORY_LABELS` only, thresholds dev-only. |
| **Secondary/mechanistic** | **S5** | **A5 IDF lexical** | Historical weighted reference, TF-IDF fitted on all 300, thresholds via OOF. |
| | **S6** | **S3 zero-shot NLI** | `typeform/distilbert-base-uncased-mnli`, 400-token premise chunks, MAX entailment, 13 frozen hypotheses `This job requires {label}.`, thresholds dev-only. Highest lexically-unseen recall internally despite 0.387 macro. |
| | **S7** | **H1 A1+S1 fallback** | A1 gate → S1 fallback if A1=0 and S1≥thr; thresholds via hybrid F1 on OOF, OFF=2.0 conservative tie-break. POST-HOC secondary. |
| | **H2** | **A1+S3 fallback** | Same gate with S3. **If final frozen thresholds are OFF for all 13 categories, H2 is `equivalent_to_A1` and NOT evaluated as distinct** (saves NLI compute). Do not force ON. |

Derivation uses **only 300 development postings**; external text/labels not inspected. Final fitted configs exist **only for external deployment**; their performance on same 300 is **not reported as unbiased** (nested CV remains unbiased-ish internal estimate). See `v4/external/freeze.py` and `EXTERNAL_FREEZE_MANIFEST.json`.

---

## 4. External Data — Two Separate Sets

### E1 — Natural External Set (PRIMARY)
- **Target N=200**
- Purpose: estimate realistic external generalisation (population performance).
- **Selection MUST NOT use model predictions/confidence** (no A1/S1/S2/S3/hybrid scores) or future labels. Allowed: job_title, UK location, source, posting date, role-family inclusion rules, duplicate info.
- Sampling: deterministic stratified on `role_family` (business analyst, data analyst, finance analyst, data scientist, marketing analyst, analytics engineer, other), with minimum coverage per family, seed 42, quotas documented. Do not resample because draw "looks bad".
- **Analysis:** Primary external result (see §7).

### E2 — Challenge Set (POST-HOC STRESS TEST, N≈100, never mixed with E1)
- Purpose: stress specific failure mechanisms discovered during development.
- Explicitly non-natural, reported **separately**.
- Uses **frozen model outputs** to identify difficult **unlabelled** cases (allowed because post-hoc).
- Pre-registered strata (before human annotation):
  - **C1 (~40): lexical-low-coverage / semantic-disagreement** — low A1 coverage (<0.15) OR A1=0 while ≥2 semantics positive. Stress unseen/paraphrased.
  - **C2 (~30): lexical ambiguity / homonym** — frozen text rules: `excellent/Excel` collision, `reports to/direct reports`, pipeline ambiguity, `clinical coding/programme`, `Sales Cloud/Service Cloud`, `Oracle/Hyperion`. Deterministic, no new rules after labels.
  - **C3 (~30): role/terminology edge** — underrepresented families, unusual analyst-adjacent terminology, lower coverage, novel companies/templates. Deterministic.
- Do not hand-pick to favour a method. Report C1/C2/C3 separately.

Selection code: `v4/external/sampling.py` (E1 asserts no model columns; E2 explicitly labelled challenge/post-hoc).

---

## 5. Source and Rights Audit

Development: **LinkedIn** (`job_link` linkedin.com), 820 postings, 2024-01-12 to 2024-01-17 (6-day window), narrow temporal/source coverage.

External **must be source shift + temporal shift** (ideal: different UK source + newer advertisements). Prefer new source + newer dates; if dates unavailable, source-held-out still valid but limitation documented.

Candidate sources audited in `v4/external/SOURCE_AUDIT.md` (Reed API, Adzuna API, Find a Job/DWP, ONS/open dataset, etc.) — each with access mechanism, fields, timestamps, UK coverage, candidate count, licence/redistribution, selected/not selected, reason. Selection must not violate terms, access controls, robots, anti-bot, paywalls. Prefer documented API / open dataset / institutional / permissive.

If selected source forbids redistribution of full advert text: **store raw text locally gitignored**, release only derived IDs/hashes/labels/categories/sampling metadata/code/URLs if permitted. Do not commit copyrighted text.

---

## 6. Deduplication (mandatory)

Dedup within E1, within E2, between E1/E2, and external candidates **against ALL 820 development postings** (not only 300 annotated).

Methods:
- **D1 exact hash** SHA-256 raw text
- **D2 normalised hash** case/whitespace/punctuation normalisation
- **D3 near-duplicate** TF-IDF cosine (threshold 0.90) + token Jaccard / MinHash (documented), review table for suspected repost/template groups.

Only one representative per duplicate group enters locked set unless protocol explicitly studies duplication. Store `duplicate_group_id`. Code: `v4/external/dedup.py`.

---

## 7. Future Analysis — Pre-registered

### 7.1 Primary External Result — E1 NATURAL only
For **every frozen method** (P1-P4 primary, S5-S6/H1/H2 secondary if distinct):
- macro-F1, micro-F1, exact/subset accuracy, Hamming accuracy/loss, per-category P/R/F1, 95% posting-level bootstrap CI, FP/FN counts, ads with ≥1 error.
- **Primary comparisons (paired bootstrap on E1):** A1 vs S1, A1 vs S2, A1 vs A2.
- Also report A5, S3, H1, H2(if distinct) as secondary.

### 7.2 Challenge Results — E2 SEPARATELY
Overall challenge metrics + by C1/C2/C3. **Never mix E1+E2 as natural population.**

### 7.3 Seen vs Lexically-Unseen Positives (CENTRAL)
Using **exact frozen A1 lexicon** (from `v4/config.py`):
- `seen`: gold-positive cell where A1 frozen term exists (`S_A1>0`)
- `unseen`: no A1 term (`S_A1==0`)
Compare **recall** for A1, A2, S1, S2, S3, H1/H2(if applicable) on unseen vs seen.
Internal unseen rate: 39/1055=3.7%. Question: does external proportion increase? If yes, does semantic relative value increase? Definition frozen.

### 7.4 Generalisation Drop
For each method: `external E1 macro-F1 − internal nested macro-F1` = `external_shift_delta` (or `generalisation_drop`). Compare A1, A2, S1, S2, S3. Question: which most stable? Do not claim causality unless design supports.

### 7.5 Company/Source/Terminology Slices (if metadata supports)
Pre-register: unseen-company vs known-company, role_family, seniority, source, lexically seen vs unseen, low vs high lexical coverage, natural vs challenge. Report support with every slice; do not over-interpret tiny slices.

---

## 8. Annotation Protocol (human only)

- Human annotators label from **exact external text that will be passed to models** (`job_summary` only; `job_title` for `role_family` metadata separately). No extra info unavailable to model.
- Materials contain **no model outputs** (no predictions, thresholds, confidence, seen/unseen flags, challenge reason, matched terms if biasing).
- Workbook: `v4/external/annotation/ANNOTATOR_A.xlsx` + `ANNOTATOR_B_OVERLAP.xlsx` (or CSV), 13 categories in exact `CATEGORIES` order plus `other/other_skills_verbatim/notes`, guidelines `annotation_guidelines_v2_0.docx`, semantic definitions frozen (do not revise after seeing model behaviour).
- **Double annotation:** at least 100/300 independently double-annotated (50 E1 + 50 E2), stratified, IDs frozen before annotation, B blind to A and to model outputs.
- **No AI gold:** Muse/ChatGPT/LLM/NLI/lexical must not fill workbooks or infer labels. Humans only; May generate workbook, enforce validation, calculate agreement after humans supply labels, but humans provide gold. If no second annotator, STOP: "External sample and annotation package are ready; independent second annotation is blocking".
- **Agreement:** per-category raw agreement, Cohen's kappa, positive/negative agreement (prevalence-sensitive kappa not alone), prevalence, A/B positive counts, plus macro/micro agreement, posting-level exact agreement. Code implemented now, results not fabricated.
- **Adjudication:** disagreement cells → `posting_id, category, A, B, blank adjudicated, blank note`, no model prediction, human consensus/independent adjudicator, final gold matrix.
- **Guidelines freeze:** record `annotation_guideline_version/hash`, `taxonomy_version`; if ambiguity discovered, log in `PROTOCOL_DEVIATIONS.md`, version guidelines, re-annotate consistently — no silent edits.

---

## 9. Locks and Governance

- **Freeze:** `EXTERNAL_FREEZE_MANIFEST.json` with taxonomy, categories, label/lexicon/pattern hashes, commit, seed, dev id/text hashes, method configs/hashes, package/python/torch/transformers versions, timestamp, `EXTERNAL_LABELS_ACCESSED=false`.
- **Sample lock:** `v4/external/LOCKED_SAMPLE_MANIFEST.csv` (gitignored if restricted text) with `external_id, source, source_posting_id, source_url(if permitted), acquired_at, published_at, role_family_sampling_stratum, challenge_stratum, natural_or_challenge, company_hash, text_sha256, normalised_text_sha256, duplicate_group_id` — no gold labels yet, hash `locked_sample_manifest_sha256` recorded, `SAMPLE_LOCKED=true, LABELS_CREATED=false, MODELS_EVALUATED=false`.
- **Label lock:** `EXTERNAL_LABEL_LOCK.json` after adjudication with sample manifest hash, guideline hash, gold file hash, overlap IDs, agreement summary hash, adjudication hash, timestamp, `LABELS_LOCKED=true`. Methods remain immutable.
- **No tuning after sample lock:** no lexicon/negative-pattern/model/prompt/chunking/C-grid/threshold changes, no retraining on external labels, no new hybrid, no removal of poor methods, no challenge-strata changes — even external **text** not used to tune.

---

## 10. Statistical Note

Once deployment settings fitted on all 300:
- **Do NOT report performance of that final fitted deployment configuration on same 300 as though unbiased.**
- Unbiased-ish internal estimate remains **nested cross-validated internal development performance** (≈0.9429 A1, 0.9377 H1, etc.).
- Newly fitted all-300 config exists **only for external deployment**. Document distinction.

---

## 11. Data Release / Privacy

Separate **safe to release** (code, taxonomy, sampling code, derived IDs/hashes/labels where permitted, role metadata, agreement stats, predictions, metrics) vs **potentially restricted** (full advert text, company/source text, URLs). Do not assume MIT licence applies to third-party advert text. Document source-specific restrictions in `v4/external/DATA_RELEASE_PLAN.md`. Store raw text locally gitignored.

---

## 12. Tests

`v4/tests/test_external_protocol.py` verifies:
- full method freeze manifest present
- taxonomy frozen (13 CATEGORIES)
- E1/E2 no overlap with 300 and 820 (exact/normalised)
- E1 sampling does not use model scores
- challenge sampling declared posthoc
- annotation workbook has no model outputs
- overlap IDs frozen (100 double, independent of labels)
- gold not generated by code (blank cells only)
- no model evaluation has consumed future gold at this stage
- sample manifest hash stable
- raw text not tracked (gitignored)
- freeze manifest deterministic

All prior 79 tests continue to pass.

---

## 13. External State Flags (at end of this task)

```
METHODS_FROZEN = true
SAMPLE_LOCKED = true/false (depends if candidate pool acquired)
LABELS_CREATED = false
LABELS_LOCKED = false
MODELS_EVALUATED = false  # must be false
```

If acquisition blocked: State B — freeze, source audit, acquisition/sampling/dedup code, annotation package generator, agreement/adjudication tooling ready, blocker documented. Prepared protocol preferable to fake external evidence.

---

*Generated before external labels; do not run external model evaluation until human annotations locked.*
