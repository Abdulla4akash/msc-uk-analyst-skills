# AGENTS.md

Context for AI coding agents working in this repository. Humans: start with `README.md`.

## What this is

An MSc research project, not a product. It extracts a **13-category skill taxonomy**
from UK data/business analyst job adverts and measures the extraction against a
**manually annotated gold standard of 300 postings**. The first extraction method
was a TF-IDF baseline; the **current evaluation foundation is `v4/`** (nested,
leakage-safe). Three further methods are planned.

The point of the repo is *comparability between methods*. Anything that makes two
methods less directly comparable is a regression, even if it improves a score.

## Repository shape

```
V1/  v2/     earlier iterations — provenance only, DO NOT EDIT
v3/           frozen original MSc baseline / historical experiment — DO NOT EDIT
v4/           CURRENT publication-grade evaluation foundation — USE THIS
  manual_work/  corpus construction + human annotation (in v3, reused by v4)
  base-model/   original baseline code (frozen, in v3)
  config.py, evaluation/, methods/, experiments/, tests/  (current, in v4)
```

`V1/`, `v2/`, `v3/` are frozen historical snapshots. Never edit, refactor, "fix" or
delete them, and never import from them. If you notice a bug there, mention it —
don't patch it.

## Current development target

**`v4/`** is current. All new work (methods 2–4, ablations, distribution-shift experiments) must integrate with `v4/`:

- `v4/config.py` — frozen taxonomy `v3-13cat-frozen` (do not import from `v3/base-model/config.py` at runtime)
- `v4/evaluation/` — data loading, `make_dev_test_split` (`internal_tuning`/`internal_holdout`), `make_cv_splits`, `evaluation/nested.py` (genuinely nested CV), metrics, bootstrap
- `v4/methods/` — baselines with batch-invariant, inductive TF-IDF scoring
- `v4/experiments/run_lexical_baseline.py` — runner that enforces inductive fitting and nested threshold isolation

Do **not** build new methods against `v3/base-model/evaluate.py` or `v3/base-model/tfidf_baseline.py` — those files use corpus-fitted TF-IDF, batch-max normalisation, and `test`-selected winners and are retained only for provenance.

## Ground rules

**The label space is defined once.** `v4/config.py` (frozen copy of `v3/base-model/config.py` at tag `v3-original-baseline`, `d7ee030`) holds `CATEGORIES`, and its **order is load-bearing** — it defines the column order of every prediction matrix and must match the gold-standard workbook columns. Never reorder, rename,
insert or drop a category without updating the gold workbook and every results file
in lockstep. All methods import from `v4/config.py`.

**Never tune on held-out data.** Thresholds are selected via genuinely nested CV using only inner training folds (`evaluation/nested.py`) and applied to outer validation folds whose labels were never inspected. `external_locked_test` (future independently collected test set) must never be inspected during tuning; `internal_holdout` metrics are reported only after freezing. Any change that lets held-out `external_locked_test` data influence threshold selection, feature fitting or model choice invalidates the numbers. Never fabricate an `external_locked_test`.

**`external_locked_test` does not exist yet.** It is RESERVED for the future independently collected dataset (new sampling date, unseen companies, independent annotators). Do not create a synthetic one, do not run an experiment called “external test”, and do not call the internal 300 splits (`internal_tuning`/`internal_holdout`, nested CV) “test” in publication-facing tables. The strongest claim currently allowed is “nested cross-validated internal development estimate”.

**The split must stay reproducible.** `make_dev_test_split()` and `make_cv_splits()` are seeded (`RANDOM_SEED = 42`) and stratified on `role_family`. Changing the seed, fraction or stratification key makes new results incomparable. Don't.

**Gold standard is human-produced.** `gold_standard_annotation_workbook_v2.xlsx` is the output of manual annotation. It is data, not a generated artefact — never rewrite it programmatically, and never "correct" labels to improve a score.

**Don't commit generated results.** The pipeline writes into `results/`, which is gitignored. `v3/base-model/*.csv` are deliberately committed snapshots of the historical baseline; `v4/results/` outputs are ignored except `.gitkeep`. Overwrite snapshots only when you intend to change documented numbers.

**Preserve comparability.** Future methods must:
- use the same frozen taxonomy (`v4/config.py`) unless a separately versioned taxonomy study is being performed;
- use leakage-safe (inductive) fitting — `TfidfVectorizer` fitted on `internal_tuning`/`outer_train` only;
- use nested/internal development evaluation (`evaluation/nested.py` or `make_dev_test_split` with `internal_tuning`/`internal_holdout`);
- never inspect `external_locked_test` during tuning;
- emit scores in fixed `CATEGORIES` order as `(n_postings, 13)` then threshold;
- retain exact split/evaluation provenance in JSON (seed, split IDs, inner strategy).

## How the baseline works

Three baselines under `v4/methods/lexical_baseline.py`, all scoring every (posting, category) pair then thresholding:

- **0 — unweighted lexical.** Binary whole-word presence proportion, no IDF.
- **1 — cosine TF-IDF.** TF-IDF vectors for postings and category pseudo-documents; raw cosine (no batch-max division), inductive fitting.
- **2 — weighted hit.** Sum of TF-IDF weights of lexicon terms present, normalised by lexicon IDF sum, plus `NEGATIVE_PATTERNS` suppression for homonyms. Inductive fitting, batch-invariant.

Current internal development estimates (300 postings treated as development material): weighted ≈ 0.934 / unweighted ≈ 0.942 nested macro-F1 — **effectively tied** (retain both as candidates; do not declare a winner on 0.001 differences). Cosine ≈ 0.682. Lexicon design + negative patterns appear to drive most performance, not IDF. See `v4/EVALUATION_AUDIT.md`.

If you touch the lexicons or negative patterns, re-run and expect numbers to move — update `v4/README.md` and `v4/EVALUATION_AUDIT.md` if they do.

## Adding method 2, 3 or 4

Follow `v4`:

1. Import `CATEGORIES` (and `CATEGORY_LABELS` for prompt/hypothesis-based) from `v4/config.py`. Do not redefine the taxonomy.
2. Use `load_gold_with_texts`, `make_dev_test_split` / `make_cv_splits` / `run_nested_cv_for_method` from `v4/evaluation/` — not `v3/base-model/evaluate.py`.
3. Produce a **score matrix** of shape `(n_postings, 13)` in `CATEGORIES` order, then threshold it — don't emit hard labels directly.
4. Thresholds must be tuned genuinely nested (see `evaluation/nested.py`) or on `internal_tuning` only.
5. Write `<method>_<variant>_internal_tuning.csv`, `<method>_<variant>_internal_holdout.csv`, `*_nested_cv_report.csv` and a `<method>_summary.json` matching the baseline's output contract (publication-safe naming).
6. Take `--corpus`, `--gold` and `--outdir` as arguments. No hardcoded absolute paths — derive from `Path(__file__)`.
7. Run against nested CV + internal holdout; report bootstrap CIs as supplementary and fold variation as primary. Never fabricate `external_locked_test`.

## Conventions

- British spelling in category names and labels (`visualisation_bi`,
  `data_modelling`). The lexicons deliberately carry both spellings as *match terms*
  — that's intentional, not an inconsistency to clean up.
- Lexicon terms are lowercase and matched case-insensitively as whole words.
- Plain functions, `argparse`, `pathlib`. No framework, no class hierarchy, no
  config system. Match that — this code is read by examiners.
- Docstrings explain *why* a choice was made (and cite sources where relevant).
  Keep that habit; it's doing real work in a dissertation context.

## Environment note

```bash
pip install -r requirements.txt
```

## Data handling

The corpus derives from a public LinkedIn job-postings dataset, filtered by
`filter_corpus.py`. `filter_corpus.py` expects `linkedin_job_postings.csv` and
`job_summary.csv` in the working directory — **these source files are not in the
repo**, so that script can't be re-run from a clean clone. The postings carry
company names and original `job_link` URLs. Treat as third-party content: don't
redistribute it further, and don't paste posting text into external services.

## Status

`v4` foundation is current (leakage-safe, nested). `external_locked_test` does not exist. Next stage will ablate lexical components and add semantic baselines — but only after this foundation is approved.
