# External Annotation Protocol — Human Only

**Version:** 1.0
**Taxonomy:** `v3-13cat-frozen` (see `v4/config.py`)
**Guidelines source:** `v3/manual_work/annotation_guidelines_v2_0.docx` + `annotation_scheme_v3.xlsx`
**Applies to:** E1 natural (N=200) and E2 challenge (N=100) external sets

> **No AI labelling.** Muse, ChatGPT, LLM, NLI, lexical model, or any automated classifier must not be used as gold-standard annotator. Humans only.

---

## 1. Purpose

Create a defensible external-data + human-annotation package for locked evaluation. Skill labels must be judged from the **same textual evidence models receive**: `job_summary`. `job_title` is used only to assign `role_family` metadata separately.

---

## 2. Annotators

- **Annotator A:** primary annotator, labels all 300 external postings (200 E1 + 100 E2)
- **Annotator B:** independent second annotator, labels **overlap of 100** (50 from E1 + 50 from E2) **before seeing A’s labels**

Overlap IDs are **frozen before annotation** via `v4/external/annotation_package.py:select_overlap_ids` (seed 42, stratified by natural/challenge, role_family, challenge_stratum). B is blind to A; B is blind to model outputs.

If no second human available: **STOP** and report `"External sample and annotation package are ready; independent second annotation is the blocking human step."` Do not substitute AI.

---

## 3. Materials Provided (no model leakage)

For each annotator workbook (`ANNOTATOR_A.xlsx`, `ANNOTATOR_B_OVERLAP.xlsx`):

**Included:**
- `external_id`, `source`, `source_posting_id`, `source_url` (if permitted), `published_at`, `job_title`, `job_summary` (model-visible text), `role_family`, `role_family_sampling_stratum`, `challenge_stratum`, `natural_or_challenge`, `duplicate_group_id`
- 13 skill columns in exact `CATEGORIES` order: `programming, sql, visualisation_bi, reporting, excel, statistics, machine_learning, data_cleaning, etl, data_modelling, cloud, stakeholder_comm, ethics_governance`
- Extra: `other`, `other_skills_verbatim`, `notes`

**Explicitly NOT included:**
- A1/S1/S2/S3 predictions, thresholds, confidence, seen/unseen flags, challenge reason that reveals expected label, lexical matched terms (if biasing), hybrid scores.

Workbooks contain **blank label cells** (0/1 to be filled). Validation enforces 0/1 only. Model outputs stay in separate researcher-only files.

Generation: `v4/external/annotation_package.py:create_blank_workbook` (validates no model columns).

---

## 4. Label Semantics

Use **frozen 13 labels** in exact order. Definitions from `annotation_guidelines_v2_0.docx`; do not revise after seeing external model behaviour. If ambiguity discovered after annotation starts, record in `PROTOCOL_DEVIATIONS.md`, version guidelines, re-annotate affected data consistently — no silent edits.

For each category, `1` if skill **required/evidenced** in `job_summary`; `0` if absent. Whole-word evidence matters for human too, but human may use paraphrase understanding (that is the point of the unseen analysis).

**Boundary rules (examples from guidelines):**
- `programming`: languages, coding, scripting (python, R, SQL not counted here)
- `sql`: database querying with SQL (not Oracle Fusion/Hyperion unless SQL explicitly)
- etc. — refer to full doc.

**Other:** if skill not in taxonomy but evidenced, set `other=1` and describe in `other_skills_verbatim`.

---

## 5. Evidence Scope

Annotate **only from `job_summary`** (the text passed to models). Do not infer from company reputation, job title alone, external knowledge, or information not in the text. If job_summary is truncated/empty, label based on what is present (mostly zeros).

`job_title` is used only to assign `role_family` for stratification; skill labels must be grounded in `job_summary`.

---

## 6. Workflow

1. **Sample lock:** `v4/external/LOCKED_SAMPLE_MANIFEST.csv` committed (no raw text if restricted), hash recorded, `SAMPLE_LOCKED=true`, `LABELS_CREATED=false`.
2. **Workbook generation:** blank workbooks gitignored if full text restricted, IDs frozen, overlap IDs frozen.
3. **Annotation:** A labels all 300; B labels 100 overlap independently.
4. **Agreement analysis:** after both complete, run `v4/external/annotation_package.py` agreement tooling (per-category raw, Cohen’s kappa, positive/negative agreement, prevalence, macro/micro, exact agreement). Do not report kappa alone for rare categories.
5. **Adjudication:** for disagreement cells produce `posting_id, category, A, B, blank adjudicated, blank note` (no model prediction). Consensus or independent adjudicator decides. Produce final gold matrix.
6. **Label lock:** generate `EXTERNAL_LABEL_LOCK.json` with sample manifest hash, guideline hash, gold file hash, overlap IDs, agreement summary hash, adjudication hash, timestamp, `LABELS_LOCKED=true`. Methods remain immutable.

---

## 7. Quality Control

- Excel data validation 0/1 only
- No pre-filled labels (code must not generate gold)
- Hash of workbook before annotation vs after
- Adjudication log required

---

## 8. Ethics / Rights

- Advert text is third-party copyright; annotators view locally, do not republish beyond what source licence permits.
- Store raw text locally gitignored; public release only derived hashes/IDs/labels.

---

## 9. After Human Annotation

Only after `EXTERNAL_LABEL_LOCK.json` (`LABELS_LOCKED=true`) may models be evaluated externally (future step, not this task). Do not run evaluation now.

---

*Human annotation only. Do not ask AI to fill workbooks.*
