# Skill Extraction from UK Analyst Job Postings

MSc project. Building and evaluating automated methods for extracting a 13-category
skill taxonomy from UK data/business analyst job adverts, measured against a manually
annotated gold standard.

This repository holds four iterations of the work:

| Directory | Status |
|-----------|--------|
| `V1/` / `v2/` | **Historical provenance** — earlier corpus/annotation versions, frozen |
| `v3/` | **Frozen original MSc baseline** — original reported baseline (historical experiment, preserved for provenance) |
| **`v4/`** | **CURRENT publication-grade evaluation foundation** — corrected, nested, leakage-safe; use this for all future work |

- `V1/` and `v2/` are frozen provenance of early corpora and schemes.
- `v3/manual_work/` holds corpus construction and the 300-posting gold standard; `v3/base-model/` is the original TF-IDF baseline with its historically reported numbers (preserved, not current).
- `v4/` corrects evaluation leakage (inductive TF-IDF, batch-invariant scoring, genuinely nested thresholds, internal development terminology) and is the foundation for all future extraction methods. See `v4/README.md` and `v4/EVALUATION_AUDIT.md`.

**Current development target: `v4/`** — all new methods should integrate with `v4/evaluation/` and `v4/methods/`, not with `v3/base-model/evaluate.py` or `v3/base-model/tfidf_baseline.py`.

`external_locked_test` (future independently collected test set) **does not exist yet** — do not fabricate it and do not call the internal 300-posting splits “test”.

---

## The task

Each job posting is labelled with any number of 13 skill categories (a multi-label
problem). The taxonomy is derived from ESCO and O\*NET, then corpus-validated:

| | | |
|---|---|---|
| `programming` | `sql` | `visualisation_bi` |
| `reporting` | `excel` | `statistics` |
| `machine_learning` | `data_cleaning` | `etl` |
| `data_modelling` | `cloud` | `stakeholder_comm` |
| `ethics_governance` | | |

Category definitions and boundary rules live in
`v3/manual_work/annotation_guidelines_v2_0.docx`; the label space itself is defined
once in `v3/base-model/config.py` (frozen as `v4/config.py` `v3-13cat-frozen` for future work) so that every method and the gold standard share identical columns.
See `v4/config.py` — `v4` does not import from `v3` at runtime.

---

## Where to look

| | |
|---|---|
| [`v4/README.md`](v4/README.md) | **Current** evaluation foundation — start here |
| [`v4/EVALUATION_AUDIT.md`](v4/EVALUATION_AUDIT.md) | v3→v4 corrections, nested estimates, what can/cannot be claimed |
| [`AGENTS.md`](AGENTS.md) | Working context and ground rules — read before changing anything |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Every column in every data file, and what the split is |
| `v3/manual_work/annotation_guidelines_v2_0.docx` | Category definitions and boundary rules |
| `v3/base-model/config.py` / `v4/config.py` | The taxonomy and lexicons (v4 is frozen copy) |

---

## Repository layout

### `v4/` — **current foundation**

| Path | What it is |
|---|---|
| `config.py` | Frozen taxonomy `v3-13cat-frozen` |
| `evaluation/` | Data loading, stratified nested CV, inductive scoring helpers, bootstrap |
| `methods/lexical_baseline.py` | Three baselines (unweighted, cosine TF-IDF, weighted lexical TF-IDF) with batch-invariant scoring |
| `experiments/run_lexical_baseline.py` | Internal holdout + nested CV runner |
| `tests/` | Batch-invariance, isolation, nested-CV, reproducibility tests |
| `EVALUATION_AUDIT.md` | v3 vs v4 vs nested numbers |

### `v3/` — frozen original baseline (historical)

**`v3/manual_work/`** — corpus construction and manual annotation

| File | What it is |
|---|---|
| `filter_corpus.py` | Builds the corpus: filters postings to UK analyst-adjacent titles, joins in the job summaries |
| `uk_analyst_corpus_v4_clean.csv` | The cleaned working corpus (used by v4 as well) |
| `gold_standard_sample_300_v3.csv` | 300-posting sample drawn for manual annotation |
| `gold_standard_annotation_workbook_v2.xlsx` | The completed gold standard annotations (used by v4) |
| `pilot_sample_20_v2.csv` / `pilot_annotation_completed.xlsx` | 20-posting pilot round |
| `annotation_guidelines_v2_0.docx` | Annotation guidelines |
| `annotation_scheme_v3.xlsx` | The coding scheme |
| `postings_reader.html` | Browser viewer for reading postings |

**`v3/base-model/`** — Original Method 1: TF-IDF baseline (historical, preserved)

| File | What it is |
|---|---|
| `config.py` | Shared taxonomy, Tier 2 lexicons, negative patterns (frozen as v4) |
| `tfidf_baseline.py` | The original baseline (batch-max normalisation, corpus-fitted TF-IDF, test-selected winner) — preserved |
| `evaluate.py` | Splitting, per-category threshold tuning, scoring (preserved) |
| `tfidf_A_cosine_test.csv` | Historical test-split results, variant A |
| `tfidf_B_weighted_hit_test.csv` | Historical test-split results, variant B |
| `tfidf_predictions_corpus.csv` | Historical predictions across the full corpus |

### `v2/`, `V1/` — earlier iterations

Same shape, earlier corpus and annotation-scheme versions. Superseded and frozen.

---

## The baseline (historical v3, preserved)

Two variants, both scored per category with thresholds tuned on the holdout’s `internal_tuning` (historically “dev”) and applied to `internal_holdout` (historically called “test”).

**A — cosine.** TF-IDF cosine similarity posting vs category pseudo-document.

**B — weighted hit.** Sum of TF-IDF weights of lexicon terms present, normalised, plus negative-pattern suppression.

**v3 reported (internal holdout n=200, historically called “test”) — preserved:**

| Variant | Macro-F1 | Micro-F1 | Subset acc. | Hamming acc. |
|---|---|---|---|---|
| A — cosine | 0.759 | 0.811 | 0.260 | 0.889 |
| **B — weighted hit** | **0.937** | **0.962** | **0.755** | **0.980** |

**v4 corrected estimates (internal development only; `external_locked_test` does not exist):**

| Variant | internal_holdout macro | Nested CV macro | Interpretation |
|---|---|---|---|
| weighted lexical TF-IDF | 0.938 [0.914,0.955] | 0.934 [0.917,0.948] | tied with unweighted |
| unweighted lexical | 0.937 | 0.942 [0.925,0.955] | tied — lexicon/patterns drive most performance |
| cosine TF-IDF | 0.776 | 0.682 [0.640,0.714] | substantially weaker |

v4 treats the 300 postings as development material; future `external_locked_test` will be independently collected. See `v4/EVALUATION_AUDIT.md`.

---

## Running it

### v4 (current)

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 v4/experiments/run_lexical_baseline.py --mode both --n_bootstrap 10000
PYTHONPATH=. python3 -m pytest v4/tests -v
```

See `v4/README.md`.

### v3 (historical, preserved — reproduced from original)

```bash
pip install -r requirements.txt

cd v3/base-model
python tfidf_baseline.py \
  --corpus ../manual_work/uk_analyst_corpus_v4_clean.csv \
  --gold   ../manual_work/gold_standard_annotation_workbook_v2.xlsx \
  --outdir results
```

Writes per-variant reports into `results/` (historical pipeline, not current foundation).

---

## Data provenance

The corpus is derived from a public dataset of LinkedIn job postings, filtered to UK
analyst-adjacent roles (see `filter_corpus.py` for the exact inclusion and exclusion
patterns). Postings are retained with their original `job_link`. The raw source files
are not included, so `filter_corpus.py` cannot be re-run from a clean clone — the
cleaned corpus is the starting point.

## Licence

Code and documentation: **MIT** ([LICENSE](LICENSE)). The job-posting data is
third-party content and is **not** licensed by this repository — see
[DATA_NOTICE.md](DATA_NOTICE.md).

## Status

v4 evaluation foundation is the current baseline (leakage-safe, nested). Three further extraction methods are planned against the same label space and `v4` framework, for direct comparison with this corrected baseline.

## Reference

Attwood, S. & Williams, A. (2023) — TF-IDF mapping of job listings to CyBOK knowledge
areas, the approach adapted in variant A.
