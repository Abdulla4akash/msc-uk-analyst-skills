# Annotator Checklist — External UK Analyst Advertisements

**Keep to one page. Do this for every posting.**

1. **Read full `job_summary`** — not just title or first line. Median length 364 words.
2. **Label each of the 13 categories independently** in exact order: `programming, sql, visualisation_bi, reporting, excel, statistics, machine_learning, data_cleaning, etl, data_modelling, cloud, stakeholder_comm, ethics_governance`. Every cell must be `0` or `1`.
3. **Do not infer unstated skills** — label only what the summary explicitly requires/expects. Employer reputation or title alone is not evidence.
4. **Use frozen definitions only** (`HUMAN_ANNOTATION_GUIDE.md`, `v4/config.py`, `annotation_guidelines_v2_0.docx`). Do not add new lexical terms.
5. **Do not search the employer/job online** — annotate only from the provided `job_summary`.
6. **Do not use ChatGPT, Muse, LLMs, or any classifier** — humans only.
7. **Do not consult the other annotator** — A and B are independent; B is blind to A until adjudication.
8. **Record uncertainty/notes using the frozen 0/1 procedure** — enter your best 0/1 and describe uncertainty in the `notes` column (e.g., "uncertain — ambiguous pipeline"). Do not enter `?` — validation allows only `0,1`.
9. **Save workbook without changing IDs, order, or sheet names** — `external_id` and row order are frozen. Keep `Annotation` sheet name.
10. **Do not add or remove postings** — label all assigned rows (A: 300, B: 100). Verify no row left blank at submission. Flag truncated/empty descriptions in `notes` but still label.

**Submission:** return the completed workbook unchanged except filled `0/1` cells (+ `other`/`other_skills_verbatim`/`notes` where needed). Do not add hidden sheets, formulas, or model outputs.
