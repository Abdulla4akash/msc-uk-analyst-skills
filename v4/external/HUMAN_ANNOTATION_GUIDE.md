# Human Annotation Guide — External UK Analyst Advertisements

**Taxonomy version:** `v3-13cat-frozen` (see `v4/config.py`)
**Guidelines source:** `v3/manual_work/annotation_guidelines_v2_0.docx` + `annotation_scheme_v3.xlsx` — do not redefine after seeing external data.
**Applies to:** E1 natural N=200 + E2 challenge N=100 (total 300), plus 100 double-annotated overlap.

> No AI labelling. Humans only. Do not use Muse/ChatGPT/LLM/NLI/lexical model.

---

## 1. What You Label

For each posting, label whether the **job_summary** (model-visible text) **requires or expects** the skill/category according to the frozen taxonomy. `job_title` and metadata are for orientation only — skill decisions must be supported by `job_summary`. Do not infer from employer reputation, external knowledge, or information not in `job_summary`.

- `1` = category clearly required/expected (explicit requirement, responsibility, tool/skill mention that satisfies the definition).
- `0` = not supported by the text (absent, merely associated, or insufficient evidence).

Do not label something merely because it is semantically associated with the job. Example: a job at a data company does not imply every data skill is required — only label what the summary states is required.

Use whole-word/phrase evidence in spirit, but human may use paraphrase understanding (that is the point of the unseen analysis). If the summary paraphrases a skill without the frozen lexical term, still label `1` if the definition is satisfied.

If `job_summary` is truncated/empty, label based on what is present (mostly zeros) and note in `notes`.

---

## 2. Evidence Scope

- Annotate **only from `job_summary`**.
- Do not search the employer or role online.
- Do not consult the other annotator.
- Do not use LLMs/classifiers.

---

## 3. Categories (exact order, frozen)

Order in workbooks matches `CATEGORIES` exactly:

`programming, sql, visualisation_bi, reporting, excel, statistics, machine_learning, data_cleaning, etl, data_modelling, cloud, stakeholder_comm, ethics_governance`

Definitions from `v4/config.py:CATEGORY_LABELS` (verbatim):

| # | Category | Definition (frozen) | Inclusion rule | Exclusion rule |
|---|----------|---------------------|----------------|----------------|
| 1 | **programming** | programming or scripting languages | Requires coding/scripting languages or explicit programming activity (e.g., Python, R, Java, VBA, scripting). | SQL alone goes to `sql`, not here. General "technical skills" without language not enough. |
| 2 | **sql** | database querying with SQL | Requires SQL or relational database querying. | Oracle Fusion/Hyperion, MongoDB/NoSQL alone without SQL not counted unless SQL explicitly mentioned. `database` alone without SQL is not enough. |
| 3 | **visualisation_bi** | data visualisation and business intelligence tools | Requires BI/visualisation tools (Power BI, Tableau, Looker, Qlik, dashboards). | Static reporting without BI tool goes to `reporting`. |
| 4 | **reporting** | producing reports and management information | Requires producing reports/MI/KPI packs. | Generic "communication" without reporting not enough. |
| 5 | **excel** | spreadsheet software such as Excel | Requires Excel/Google Sheets/pivot/VLOOKUP. | "excellent" is not Excel; `reports to` is not reporting — beware homonyms (see C2). |
| 6 | **statistics** | statistical analysis and forecasting | Requires statistical methods, forecasting, hypothesis testing, econometrics. | General "analysis" without statistical method not enough. |
| 7 | **machine_learning** | machine learning and predictive modelling | Requires ML, predictive modelling, feature engineering, deep learning, NLP/LLM for modelling. | Rule-based analytics without ML not enough. |
| 8 | **data_cleaning** | data cleaning and data quality | Requires cleaning, cleansing, wrangling, quality, validation, integrity, profiling. | General "data management" without quality aspect not enough. |
| 9 | **etl** | data engineering, ETL pipelines and data warehousing | Requires ETL/ELT, pipelines, ingestion, warehousing, lakes, specific tools (Airflow, Kafka, Snowflake etc.) when used for ETL context. | Programme/project pipeline not counted (see ambiguity). |
| 10 | **data_modelling** | data modelling and schema design | Requires data modelling, dimensional modelling, star schema, ERD, schema design. | Database use without modelling not enough. |
| 11 | **cloud** | cloud computing platforms | Requires cloud platforms (AWS, Azure, GCP, Databricks/Snowflake when cloud-context, S3/EC2 etc.). | On-premise servers without cloud not enough. |
| 12 | **stakeholder_comm** | stakeholder communication and presenting findings | Requires stakeholder management/engagement, presenting, storytelling to non-technical audiences. | Internal reporting without stakeholder focus goes to `reporting`. |
| 13 | **ethics_governance** | data governance, privacy and GDPR compliance | Requires governance, GDPR, data protection, privacy, ethics, confidentiality. | Generic "compliance" without data context not enough. |

`other` + `other_skills_verbatim`: if a required skill is clearly evidenced but not in the 13, set `other=1` and describe verbatim. Do not use `other` to duplicate a 13-category skill.

`notes`: free text for uncertainty, ambiguity, or quality issues (e.g., truncated description). Do not put labels in notes.

Full original wording and examples remain in `annotation_guidelines_v2_0.docx`. This guide does not add new lexical terms.

---

## 4. How to Decide 1 vs 0

1. Read the **full** `job_summary` (not just first paragraph). Job summaries in this set are 570–6008 characters (median 2519) and 97–983 words (median 364) — expect multi-paragraph descriptions.
2. For each of the 13 categories **independently**, ask: does the text state the job requires, expects, or will involve this skill? Look for explicit requirement, responsibility, or essential/desirable criterion.
3. Mark `1` only if evidenced. Otherwise `0`. Leave no cell blank — every category must be 0 or 1.
4. Record genuine uncertainty in `notes` (e.g., "unclear if 'pipeline' refers to data pipeline or sales pipeline") but still enter your best 0/1 judgement. Do **not** leave `?` — the workbook validation allows only `0,1,blank`. The `?` workflow is **not** part of the frozen protocol (see §6).

---

## 5. Ambiguity Handling (frozen C2 patterns — do not learn from external data)

Be alert to homonyms that the challenge set deliberately stresses:

- `excellent` / `excellence` **is not** `Excel`
- `reports to` / `direct reports` **is not** `reporting` (MI)
- `sales/drug/product/candidate pipeline` **is not** `ETL pipeline` unless data-pipeline context
- `clinical coding`, `wellbeing programme`, `programme` **is not** `programming`
- `Sales Cloud` / `Service Cloud` **is not** `cloud platform` in the same sense
- `Oracle Fusion/EPM/EPBCS/PBCS`, `Hyperion` **is not** `SQL` unless SQL explicitly

If the text contains such ambiguous spans, label strictly by the skill definition, not keyword collision.

---

## 6. 0 / 1 / Uncertain Procedure (frozen protocol)

The frozen workbooks enforce **binary 0/1 only** (Excel data validation `0,1` with blank allowed). There is **no `?` label** in the schema.

- **Preferred frozen workflow:** `1` = clearly required/expected, `0` = not supported. If uncertain, enter your best 0/1 and describe the uncertainty in `notes` (e.g., "uncertain — 'programme' could be programming or NHS programme"). Flag for adjudication via notes; do not leave the cell blank.
- **Do not introduce `?`** — this would contradict the already-frozen annotation schema enforced by `create_blank_workbook` (validation `0,1` only). The protocol states: *Uncertainty handling is via `notes` + adjudication, not a third label value.*

If the frozen protocol is later amended to support `?`, it would require a documented deviation, guideline version bump, and re-annotation — not a silent change.

---

## 7. Practical Workbook Instructions

- Do not add, remove, or reorder rows. IDs are frozen — `external_id` must remain unchanged.
- Do not change `external_id`, `source`, `published_at`, or taxonomy column order.
- Fill **all 13** skill cells + `other` for every row. `other_skills_verbatim` and `notes` may be blank.
- Use only `0` or `1` (or blank temporarily while working, but none blank at submission).
- Save workbook without changing filename/sheet name (`Annotation`).
- Do not add hidden columns, formulas, or comments containing model hints.

---

## 8. Quality Examples (from frozen guidelines, not external data)

- *Example inclusion:* "Experience with Python and SQL for data extraction" → `programming=1` (Python), `sql=1` (SQL).
- *Example exclusion:* "Excellent communication skills and reports to the Head of Analytics" → `excel=0` (excellent ≠ Excel), `reporting=0` (reports to ≠ MI), `stakeholder_comm` may be `1` if presenting to stakeholders is described elsewhere.
- *Example other:* "Requires knowledge of SAS and SPSS for forecasting" → `statistics=1` (SAS/SPSS are in frozen lexicon), not `other`. "Requires Alteryx" → `visualisation_bi=1`. "Requires Welsh language" → `other=1` + `other_skills_verbatim="Welsh language"`.

---

## 9. Version

- Guide version: 1.0 aligned to `v3-13cat-frozen`, `v4-preexternal-freeze` tag, `EXTERNAL_FREEZE_MANIFEST.json` taxonomy hashes.
- Any ambiguity discovered after annotation starts must be logged in `v4/external/PROTOCOL_DEVIATIONS.md`, not silently fixed.

---

*Annotate from `job_summary` only. When unsure, read the definition again, check the summary evidence, and note your reasoning.*
