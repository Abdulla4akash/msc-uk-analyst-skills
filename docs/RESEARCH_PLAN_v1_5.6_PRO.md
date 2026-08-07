# Research Plan v1 — 5.6 Pro

**Shared Foundation + Six Publishable Directions for UK Analyst Skill Extraction**

| Field | Value |
|---|---|
| **Version** | `v1 — 5.6 Pro` |
| **Date** | 2026-08-06 |
| **Repository** | `Abdulla4akash/msc-uk-analyst-skills` |
| **Baseline** | `v3/` — weighted lexical baseline, 13-category taxonomy, 300 manually annotated postings, macro-F1 0.937 (test n=200) |
| **Status** | Planning document — no code or data changes beyond preservation |
| **Tag** | `v3-original-baseline` (see Part A Step 1) |

> The key is **not** to build six unrelated projects from scratch. First create one reliable research foundation, then each direction becomes a different paper built on top of it.

Existing assets are already useful: a 13-category taxonomy (`v3/base-model/config.py: CATEGORIES`), 300 manually annotated postings (`v3/manual_work/gold_standard_annotation_workbook_v2.xlsx`), human-confirmed `role_family` and `seniority`, an `other` label with verbatim unmatched skills (`other_skills_verbatim`), and a strong weighted lexical baseline (variant B, `tfidf_B_weighted_hit_test.csv`).

**Related docs:** [DATA_DICTIONARY.md](DATA_DICTIONARY.md) · [../README.md](../README.md) · [../AGENTS.md](../AGENTS.md) · [../v3/base-model/config.py](../v3/base-model/config.py)

---

## Table of Contents

1. [Part A — Shared Foundation (Do Once)](#part-a--shared-foundation-for-every-direction)
2. [Direction 1 — Interpretable vs Semantic Models under Distribution Shift](#direction-1--interpretable-versus-semantic-models-under-distribution-shift)
3. [Direction 2 — Role-Aware and Hierarchical Taxonomy](#direction-2--role-aware-and-hierarchical-analyst-skill-taxonomy)
4. [Direction 3 — From Mentions to Actual Requirements](#direction-3--from-skill-mentions-to-actual-job-requirements)
5. [Direction 4 — Open-Set Skill Discovery & Taxonomy Maintenance](#direction-4--open-set-skill-discovery-and-taxonomy-maintenance)
6. [Direction 5 — Temporal Skill Bundles & Role Convergence](#direction-5--temporal-skill-bundles-and-role-convergence)
7. [Direction 6 — Cross-Source / Cross-Country Transfer](#direction-6--cross-source-or-cross-country-transfer)
8. [Minimum Viable Paper per Direction](#minimum-viable-paper-for-each-direction)
9. [Recommended Execution Order](#recommended-execution-order)
10. [Machine-Readable Config Template](#appendix-a--machine-readable-config-template)
11. [Evaluation Harness Contract](#appendix-b--evaluation-harness-contract)
12. [Provenance & Release Checklist](#appendix-c--provenance--release-checklist)

---

# Part A — Shared Foundation for Every Direction

> Do this once before starting any of the six papers.

## Step 1 — Preserve the Present Result

Keep the current `v3/` work unchanged as the original dissertation baseline.

Create a new directory or branch, for example:

```text
v4/
  data/
  annotation/
  evaluation/
  methods/
  experiments/
  reports/
```

Tag the current commit:

```bash
git tag -a v3-original-baseline -m "v3 original dissertation baseline — 0.937 macro-F1 (weighted lexical, test n=200)"
git push origin v3-original-baseline
```

This prevents later corrections from destroying the provenance of the reported **0.937 macro-F1** result.

**Rule:** `V1/` and `v2/` are frozen provenance — never edit. `v3/` becomes read-only after the tag; all new work goes in `v4/` (or a feature branch).

## Step 2 — Correct the Evaluation Pipeline

The current code should be changed before any publication experiment because:

* TF-IDF is fitted on the full corpus;
* score normalisation depends on all evaluated documents;
* the best variant is chosen using test macro-F1.

Correct it as follows:

1. Remove maximum-score normalisation across the evaluation batch.
2. Fit TF-IDF **only** on the development or training documents.
3. Tune thresholds only using development folds.
4. Select model variants using development performance.
5. Lock the final model.
6. Run the locked external test set **only once**.
7. Record every configuration in a machine-readable file.

Example `config.json`:

```json
{
  "taxonomy_version": "2.0",
  "split_version": "external_test_v1",
  "model": "weighted_lexicon",
  "idf_fit_data": "development_only",
  "threshold_source": "nested_cv",
  "random_seed": 42,
  "categories": 13,
  "split": { "method": "stratified_role_family", "dev": 100, "test": 200, "stratify_on": "role_family" }
}
```

See [Appendix A](#appendix-a--machine-readable-config-template) for the full schema.

**Code locations to fix:** `v3/base-model/tfidf_baseline.py` (vectoriser fitting), `v3/base-model/evaluate.py` (`make_split`, `tune_thresholds`, normalisation).

## Step 3 — Freeze the Taxonomy and Lexical Resources

Before collecting the external test set, freeze:

* the 13 category definitions (`CATEGORIES` / `CATEGORY_LABELS` in `config.py`);
* lexicon terms (`LEXICONS`);
* negative patterns (`NEGATIVE_PATTERNS`);
* boundary rules (`annotation_guidelines_v2_0.docx`);
* role-family definitions;
* model prompts;
* thresholds and hyperparameter-search procedures.

Do not add a word to the lexicon after looking at the external test advertisements.

Any later additions should create a new version:

```text
taxonomy_v2.0   # frozen for external test
taxonomy_v2.1   # synonym / pattern addition
taxonomy_v3.0   # structural change (new category, hierarchy)
```

Record the taxonomy version in every results file and in `DATA_DICTIONARY.md`.

## Step 4 — Remove Duplicate and Near-Duplicate Advertisements

Job advertisements are frequently reposted or copied by recruitment agencies. A random split can therefore place nearly identical advertisements in both development and test sets.

Create deduplication using:

1. exact text hashes (`job_summary` SHA-256);
2. normalised text hashes (lowercased, whitespace/punctuation normalised);
3. TF-IDF or embedding similarity (cosine > 0.90 threshold, tuned on dev);
4. company and job-title similarity;
5. repeated `job_link` values.

Assign all duplicates to the same split.

Create a `template_group_id` so that related postings remain together:

```text
posting_id | company | template_group_id | split
P001       | ADLIB   | T018              | development
P104       | ADLIB   | T018              | development
P205       | ADLIB   | T018              | development
```

**Implementation:** new script `v4/evaluation/deduplicate.py` that outputs `corpus_dedup.csv` with columns `posting_id, company, template_group_id, split, duplicate_reason`. Grouped splitting via `GroupKFold` / `StratifiedGroupKFold` on `template_group_id`.

## Step 5 — Establish Annotation Reliability

A publishable dataset should not rely only on one person's labels without measuring consistency.

1. Select a stratified subset of approximately **60–100** advertisements.
2. Include all major role families (`role_family`).
3. Oversample rare and difficult categories (`cloud`, `ethics_governance`, `data_modelling`, `data_cleaning`).
4. Have a **second annotator** label them independently.
5. Do not let the second annotator see the original labels.
6. Calculate:
   * Cohen's kappa per category;
   * positive agreement (e.g. prevalence-adjusted);
   * negative agreement;
   * overall micro and macro agreement.
7. Adjudicate disagreements.
8. Revise unclear guidelines.
9. Reannotate affected examples where necessary.

For rare categories, raw percentage agreement can be misleading, so report **category prevalence alongside kappa**.

Report template:

```text
category            prevalence  kappa  pos_agree  neg_agree
programming         0.34        0.88   0.90       0.96
...
cloud               0.08        0.61   0.68       0.97
ethics_governance   0.05        0.54   0.58       0.98
MACRO AVG           —           0.74   0.76       0.96
```

Target: macro kappa ≥ 0.70, no critical category < 0.50 without guideline revision.

## Step 6 — Build Two External Evaluation Sets

Ideally create two different test sets.

### Natural-distribution test set

A random sample of **new** UK analyst advertisements (same inclusion rules as `filter_corpus.py`, but drawn after the freeze date, unseen companies/periods). This answers:

> How well does the system perform in normal deployment?

Size: ≥ 200 postings, independently annotated, same 13 labels + `other`.

### Challenge test set

A deliberately difficult sample containing:

* "excellent" without Excel;
* "reports to" without report production;
* product or sales pipelines without ETL;
* Salesforce products without infrastructure-cloud requirements;
* Snowflake or Databricks in ambiguous contexts;
* skills expressed using unseen synonyms;
* rare labels such as ethics and governance;
* advertisements with **no** taxonomy labels.

This answers:

> Can the system survive the cases that defeat simple keyword matching?

The current results suggest that `cloud`, `reporting`, `data_cleaning`, and `machine_learning` deserve particular attention. `cloud` is currently the weakest label, with **F1 0.727**.

Size: ~100–150 postings, balanced toward hard negatives. Annotate `difficulty_reason` per posting.

## Step 7 — Create One Common Evaluation Harness

Every method should produce the same output:

```text
posting_id
score_programming
score_sql
...
score_ethics_governance
prediction_programming
...
prediction_ethics_governance
```

Report:

* macro-F1;
* micro-F1;
* per-category precision, recall, F1 and **PR-AUC**;
* subset (exact-match) accuracy;
* Hamming loss / Hamming accuracy;
* calibration error (ECE);
* abstention performance (if applicable);
* runtime;
* memory use;
* cost per 1,000 advertisements;
* **95% bootstrap confidence intervals** (e.g. 10k resamples, stratified by posting).

Also report results sliced by:

* role family;
* seniority;
* source;
* company seen versus unseen;
* known versus unseen terminology;
* natural versus challenge set.

**Contract:** see [Appendix B](#appendix-b--evaluation-harness-contract). All four planned methods (`base-model` + 3 new) import `CATEGORIES` from `config.py` and use `load_gold`, `make_split`, `tune_thresholds`, `evaluate`, `format_report` unchanged.

## Step 8 — Prepare the Ethical Release Strategy

The job-advertisement text is third-party material and is not licensed by the repository.

Release:

* code (MIT);
* annotation guidelines (`annotation_guidelines_v2_0.docx` + future versions);
* taxonomy (`config.py`, `CATEGORIES`, `CATEGORY_LABELS`, `LEXICONS`, `NEGATIVE_PATTERNS`);
* binary labels (`posting_id × 13` matrix + `other`);
* derived metadata (`role_family`, `seniority`, `template_group_id`);
* posting identifiers (`posting_id`, `job_link` hashes);
* hashes (exact + normalised text hashes);
* data-processing scripts (`filter_corpus.py`, dedup, split);
* evaluation results (`*_test.csv`, `*_summary.json`).

Only release complete advertisement text where the source licence and institutional guidance clearly permit it.

See `DATA_NOTICE.md` and `LICENSE`.

---

# Direction 1 — Interpretable versus Semantic Models under Distribution Shift

## Main question

> When do lexical methods outperform language models, and when do semantic models become more reliable?

This is the **strongest immediate paper**.

## Step 1 — Define hypotheses

Write the hypotheses **before** running the models.

For example:

* **H1:** Weighted lexical matching performs best on in-domain advertisements containing explicit tool names.
* **H2:** Semantic models perform better on unseen synonyms and indirectly expressed skills.
* **H3:** Lexical performance falls more sharply under source and temporal shift.
* **H4:** A hybrid model achieves the best accuracy–cost trade-off.
* **H5:** LLMs improve recall but generate more contextual false positives.

Pre-register hypotheses and the evaluation plan (OSF or institutional registry) before external-test scoring.

## Step 2 — Construct Four Evaluation Conditions

Use the **same** models in four settings:

1. **In-domain:** development and test advertisements from similar sources and dates.
2. **Company-held-out:** test companies do not appear in development (`GroupKFold` on `company`).
3. **Source-held-out:** train/tune on one source and test on another (e.g. LinkedIn → Indeed/Adzuna).
4. **Temporal-held-out:** tune on earlier advertisements and test on later ones (`first_seen` split).

This gives the paper an actual research contribution beyond comparing model scores.

## Step 3 — Implement the Model Families

Use at least the following systems (all share `CATEGORIES` order; all output `(n_postings, 13)` scores):

#### Model A — Basic lexicon

Binary match if any valid term appears. Simplest lower baseline.

#### Model B — Corrected weighted lexical system

Present weighted-hit method, but with leakage removed (Step 2 of Part A).

#### Model C — Supervised TF-IDF classifier

Train one binary logistic-regression classifier per category:

```text
TF-IDF features → 13 one-vs-rest classifiers
```

Tests whether supervised lexical patterns outperform manually designed matching.

#### Model D — Sentence-embedding classifier

Represent advertisements and category descriptions as embeddings, then calculate similarity (cosine or learned threshold; e.g. `all-MiniLM-L6-v2` + per-category threshold tuned on dev).

#### Model E — Zero-shot NLI classifier

For each category, test a hypothesis such as:

```text
"This job requires cloud computing skills."
```

Threshold tuned on dev; no target fine-tuning.

#### Model F — Fine-tuned transformer

Fine-tune a compact encoder (e.g. `distilbert-base-uncased` or `DeBERTa-v3-small`) for multi-label classification using **grouped** cross-validation (`template_group_id`).

#### Model G — Structured-output LLM

Ask a local or approved LLM to return:

```json
{
  "sql": {
    "present": true,
    "evidence": "Strong SQL experience",
    "confidence": 0.97
  }
}
```

Evidence span must be a verbatim substring; reject otherwise.

#### Model H — Hybrid system

For example:

```text
High-confidence lexical match  → accept prediction
Ambiguous or unseen wording    → semantic model or LLM
```

Gate by lexical score margin or embedding novelty.

## Step 4 — Conduct Development-Only Tuning

For each model:

1. tune hyperparameters using **grouped** cross-validation;
2. tune thresholds **separately per label**;
3. select the final variant using **mean development macro-F1**;
4. freeze the final model;
5. never choose a model based on external-test performance.

## Step 5 — Create Unseen-Expression Subsets

For every external test example, mark whether the positive skill expression:

* appears in the development lexicon;
* is a morphological variation;
* is a new product or tool;
* is a paraphrase;
* is implied rather than directly stated.

Compare model performance separately on:

```text
seen expression
unseen expression
contextual ambiguity
explicit requirement
indirect requirement
```

This is likely where the semantic models will show their value.

Annotation column: `expression_type` ∈ {`seen`, `morphological_variant`, `new_tool`, `paraphrase`, `implied`}.

## Step 6 — Perform Error Analysis

Manually code every disagreement between the strongest lexical and semantic methods.

Use categories such as:

* homonym;
* product-name ambiguity;
* taxonomy overlap;
* unseen synonym;
* negation;
* incidental company description;
* skill required by another team;
* vague soft-skill wording;
* annotation disagreement;
* model hallucination.

Then report which model fails for which reason (confusion matrix of error types × model).

## Step 7 — Measure Operational Cost

For each model report:

* CPU or GPU requirement;
* total inference time (per 1k postings, with hardware spec);
* cost per 1,000 advertisements (compute + API if any);
* model size (parameters / MB);
* energy or hardware requirements where measurable;
* whether advertisement text leaves the local environment;
* reproducibility (seed, version, determinism).

A lexical system that is slightly less accurate but 1,000× cheaper may still be the better deployment choice.

## Step 8 — Test Statistical Significance

Use:

* paired bootstrap confidence intervals for macro-F1 (10k resamples);
* per-category bootstrap intervals;
* McNemar tests for paired binary decisions (per category);
* correction for multiple per-category comparisons (Holm or Benjamini–Hochberg).

Do not interpret a difference such as 0.937 versus 0.941 as meaningful unless the uncertainty supports it.

## Step 9 — Write the Paper around Behaviour, Not Rankings

Structure:

1. problem and research gap;
2. taxonomy and gold standard;
3. leakage-controlled evaluation;
4. model families;
5. in-domain results;
6. distribution-shift results;
7. unseen-expression analysis;
8. cost and interpretability;
9. error taxonomy;
10. recommendations for hybrid deployment.

### Minimum publishable result

At least one genuinely external test source or time period, independent annotation, confidence intervals, and a clear explanation of **why** model performance changes under shift.

---

# Direction 2 — Role-Aware and Hierarchical Analyst-Skill Taxonomy

## Main question

> Can one flat 13-category taxonomy represent all analyst occupations, or do different analyst roles require different skill structures?

## Step 1 — Audit Current Taxonomy Coverage by Role

For every role family, calculate:

* mean number of taxonomy labels;
* percentage with `other = 1`;
* most common unmatched skills (`other_skills_verbatim`);
* categories rarely or never used;
* common category combinations (co-occurrence matrix).

Example output:

```text
Role family             % with unmatched skills
Business analyst        68%
Data analyst            21%
Analytics engineering   30%
Data science            24%
```

A high `other` rate in one role family indicates poor taxonomy coverage.

## Step 2 — Review All Unmatched Skills

Use:

* `other` (binary);
* `other_skills_verbatim` (free text);
* annotator `notes`;
* advertisement text around each unmatched skill.

Remove items that are:

* job benefits;
* personality adjectives with no occupational meaning;
* domain descriptions;
* duplicated forms of an existing category;
* clearly outside the study's intended scope.

## Step 3 — Conduct Independent Open Coding

Two researchers independently group the unmatched skills.

Possible groups may include:

* requirements elicitation;
* process modelling;
* project management;
* change management;
* commercial awareness;
* financial analysis;
* experimentation;
* version control;
* API knowledge;
* leadership;
* domain-specific regulation.

Do not decide these categories in advance. Let them emerge from the data first. Measure inter-coder agreement on the grouping.

## Step 4 — Map Proposed Categories to External Frameworks

For each proposed group, record:

```text
candidate category
definition
example phrases
included concepts
excluded concepts
ESCO equivalent
O*NET equivalent
reason for inclusion
```

A category should be added only if it is:

* frequent enough (e.g. ≥ 5% of relevant role postings or ≥ 15 occurrences);
* conceptually coherent;
* distinguishable from existing categories (kappa ≥ 0.60 in pilot);
* consistently annotatable;
* relevant to the study question.

## Step 5 — Build a Hierarchical Taxonomy

A possible structure is:

```text
Analyst skills
├── Shared core
│   ├── stakeholder communication
│   ├── reporting
│   └── data governance
│
├── Data analysis
│   ├── SQL
│   ├── statistics
│   ├── visualisation
│   └── data cleaning
│
├── Analytics engineering
│   ├── ETL
│   ├── cloud
│   ├── data modelling
│   └── orchestration
│
├── Business analysis
│   ├── requirements elicitation
│   ├── process modelling
│   ├── change management
│   └── option appraisal
│
└── Data science
    ├── machine learning
    ├── experimentation
    └── model deployment
```

The exact structure must come from the evidence, not this illustrative hierarchy. Version as `taxonomy_v3.0` if hierarchy is adopted.

## Step 6 — Write New Annotation Guidelines

For every category specify:

* formal definition;
* positive examples;
* negative examples;
* overlap rules;
* role-specific interpretation;
* whether generic phrases count;
* whether tool names alone are sufficient.

Pay special attention to existing overlaps:

* SQL versus general databases;
* ETL versus cloud (Snowflake, Databricks, Redshift);
* statistics versus machine learning;
* reporting versus visualisation;
* stakeholder communication versus generic communication.

## Step 7 — Run a Second Annotation Pilot

Select a **role-balanced** sample (stratified by `role_family`, `seniority`).

Annotate it independently with both:

1. the original flat taxonomy;
2. the proposed hierarchical taxonomy.

Measure:

* coverage (`other` rate before vs after);
* inter-annotator agreement (per category + hierarchical);
* number of unresolved examples;
* time per advertisement;
* frequency of category overlaps;
* proportion assigned to `other`.

The new taxonomy should not merely contain more labels — it should improve coverage **without destroying reliability**.

## Step 8 — Compare Role-Aware Classification Designs

Test four systems:

1. **Universal flat model:** one model for every role.
2. **Role-specific thresholds:** same categories, different thresholds per role.
3. **Role-conditioned model:** role information is included as an input feature/prompt.
4. **Hierarchical model:** predict shared skills and role-specific branches (e.g. hierarchical loss).

At deployment time, human-confirmed `role_family` will not normally be available. Therefore compare:

* **oracle role:** human label supplied;
* **predicted role:** role inferred from title and text (rule-based or classifier).

The gap between these two reveals whether the role-aware approach is practically usable.

## Step 9 — Test Construct Validity

Ask:

* Are the categories distributed as occupational theory predicts?
* Are business-analysis skills concentrated in business-analyst roles?
* Are engineering skills concentrated in analytics-engineering roles?
* Do senior roles contain more governance, communication and leadership?
* Does the hierarchy improve explanatory value (e.g. better role prediction, tighter skill bundles)?

## Step 10 — Finalise the Contribution

The paper should contribute:

1. a critique of flat analyst taxonomies;
2. a role-aware hierarchical taxonomy (versioned);
3. annotation guidelines;
4. agreement and coverage evidence;
5. role-conditioned extraction experiments;
6. practical implications for labour-market analytics.

### Minimum publishable result

A substantially revised taxonomy with independent annotation, improved coverage, acceptable agreement (macro kappa ≥ 0.65), and evidence that role-aware modelling changes or improves extraction.

---

# Direction 3 — From Skill Mentions to Actual Job Requirements

## Main question

> Can the model distinguish genuine candidate requirements from incidental mentions of technologies and activities?

## Step 1 — Define the New Annotation Unit

Move from only advertisement-level binary labels to **exact textual evidence**.

For every skill mention annotate:

```text
span                       # exact character offsets in job_summary
normalised category        # one of the 13 (or extended taxonomy)
mention type               # tool / method / task / ...
requirement status         # required / preferred / incidental / ...
proficiency                # e.g. expert, familiar, none stated
experience duration        # e.g. "3 years" or null
negation                   # negated or not
subject                    # candidate vs company vs other team
evidence section           # responsibilities / requirements / benefits / boilerplate
```

## Step 2 — Define Requirement Status

Use mutually exclusive labels such as:

* required;
* preferred;
* beneficial;
* training provided;
* future responsibility;
* incidental context;
* not required;
* unclear.

Example:

> "Experience with Tableau would be advantageous."

```text
skill = visualisation_bi
status = preferred
```

> "You will work with the Tableau development team."

```text
skill = visualisation_bi
status = incidental
```

## Step 3 — Define Mention Type

Classify each extracted phrase as:

* tool or platform;
* programming language;
* analytical method;
* task;
* process;
* knowledge area;
* interpersonal capability;
* regulation;
* domain expertise.

This prevents a platform name, activity and capability from being treated as identical objects.

## Step 4 — Annotate Relations

Connect related elements.

Example:

> "Three years' experience building ETL pipelines with Python and Airflow."

Relations:

```text
three years → ETL            (duration)
Python → implements → ETL
Airflow → implements → ETL
ETL → status → required
```

Schema: `(head_span, relation, tail_span)` with a closed relation set.

## Step 5 — Run a Small Pilot

Start with **20–30** diverse advertisements.

Ensure the pilot contains:

* cloud ambiguity;
* reporting-line language ("reports to");
* tool names in company descriptions;
* preferred versus required wording;
* negation ("no prior SQL experience required");
* multiple skills in one sentence;
* requirements applying to another team.

After annotation:

1. list disagreements;
2. revise rules;
3. repeat the pilot;
4. freeze schema version 1.0.

## Step 6 — Build the Main Annotated Dataset

Sample advertisements across:

* role family;
* seniority;
* label frequency;
* source;
* sentence complexity;
* positive and negative cases.

Use **double annotation** on a substantial subset (≥ 60 postings).

Measure:

* span-level agreement (exact + partial overlap F1);
* category agreement;
* requirement-status agreement;
* relation agreement.

## Step 7 — Build Baseline Systems

#### Baseline A — Regex and lexicon

Extract known terms and infer requirement status using nearby cue words (`required`, `essential`, `desirable`, `advantageous`, `training provided`).

#### Baseline B — Sentence classification

Classify each sentence by skill category and requirement status (multi-label sentence classifier).

#### Baseline C — Token-level transformer

Extract exact skill spans using sequence labelling (BIO scheme, e.g. `bert-base-uncased` + CRF).

#### Baseline D — Span-plus-relation model

Extract spans, attributes and relations jointly or in a pipeline (e.g. SpERT or PURE-style).

#### Baseline E — Structured-output LLM

Require exact evidence spans and reject outputs unsupported by the advertisement (verbatim check).

## Step 8 — Create a Hard-Negative Benchmark

Create a separate challenge set specifically for misleading mentions:

```text
excellent communication
reports to the finance director
sales pipeline
Salesforce Service Cloud
Python engineering team
training in Power BI will be provided
no prior SQL experience required
```

Test whether each model incorrectly classifies them as candidate requirements.

## Step 9 — Evaluate at Multiple Levels

Report:

#### Span level

* exact-match precision, recall and F1;
* partial-overlap F1.

#### Attribute level

* requirement-status accuracy;
* mention-type accuracy;
* negation accuracy.

#### Relation level

* relation precision, recall and F1.

#### Advertisement level

Aggregate the extracted requirements back into the original 13 labels and compare with the existing benchmark (does contextual filtering improve corpus-level estimates?).

## Step 10 — Demonstrate Substantive Impact

Compare two estimates of labour demand:

1. skill merely **mentioned**;
2. skill genuinely **required or preferred**.

For example:

```text
Cloud mentioned in 28% of advertisements
Cloud genuinely required in 17%
```

This demonstrates why contextual extraction matters beyond improving an NLP score.

### Minimum publishable result

A new evidence-level annotated corpus, a clear requirement-status schema, strong agreement (span F1 ≥ 0.70, status kappa ≥ 0.65), and proof that mention counting materially overestimates or mischaracterises skill demand.

---

# Direction 4 — Open-Set Skill Discovery and Taxonomy Maintenance

## Main question

> Can the system recognise that an advertisement contains an important skill outside the existing taxonomy and propose a defensible taxonomy update?

## Step 1 — Define "Unknown Skill"

An unknown should be a meaningful occupational requirement that:

* is not covered by any current category;
* cannot be treated as a synonym of an existing category;
* is not merely a company, product or benefit;
* occurs in a candidate-relevant context.

Create three classes:

```text
known skill
unknown but valid skill
not a skill
```

## Step 2 — Build a Controlled Open-Set Benchmark

Create two kinds of unknowns.

#### Artificial unknowns

Remove selected known categories or terms during training.

Example:

* hide `cloud`;
* hide all Airflow mentions;
* hide process-modelling skills (if taxonomy extended).

The system must recognise that these examples do not fit the remaining taxonomy.

#### Naturally occurring unknowns

Use the existing `other_skills_verbatim` field and newly annotated external advertisements. These are the real evaluation target.

## Step 3 — Extract Candidate Phrases

Compare:

* noun-phrase extraction (spaCy NP chunks);
* lexicon expansion (embedding nearest neighbours);
* token-level skill extractor;
* sentence embeddings (novelty via distance to taxonomy);
* LLM candidate generation (with verbatim grounding).

Every candidate must include its **exact supporting sentence** (or character offsets).

## Step 4 — Attempt Taxonomy Linking

For each candidate phrase:

1. compare with current category definitions (`CATEGORY_LABELS` + lexicons);
2. compare with known lexicon terms (embedding similarity);
3. retrieve nearest external taxonomy concepts (ESCO/O*NET);
4. assign a similarity score;
5. decide:
   * link to existing category;
   * suggest new category;
   * reject as non-skill.

## Step 5 — Create a Novelty Score

Combine signals such as:

* low similarity to current categories;
* high classifier uncertainty (entropy / margin);
* disagreement between models;
* repeated appearance across advertisements;
* coherent occupational context.

Example:

```text
Novelty score =
  0.30 semantic distance
+ 0.25 model disagreement
+ 0.20 frequency
+ 0.15 cross-company diversity
+ 0.10 annotator confidence
```

The exact weights should be tuned on development data (optimise ranking precision).

## Step 6 — Cluster Unknown Skills

Cluster accepted unknown candidates using embeddings and contextual evidence (e.g. HDBSCAN on sentence embeddings + TF-IDF).

The system may discover groups such as:

* requirements engineering;
* process mapping;
* Git and version control;
* model deployment;
* experimentation;
* project management.

Review whether each cluster is:

* one coherent concept;
* several concepts incorrectly combined;
* a synonym of an existing category;
* too role-specific;
* too rare to add.

Report cluster purity and coverage.

## Step 7 — Introduce Human Review

Create a review interface showing:

```text
candidate phrase
supporting sentences (2–3 examples)
number of companies
number of role families
nearest existing categories (with similarity)
nearest external concepts (ESCO/O*NET)
proposed label
model confidence
```

The reviewer selects:

* accept new category;
* add as synonym to existing category;
* merge with another proposal;
* defer (needs more evidence);
* reject.

Log every decision with rationale (audit trail for `taxonomy_v2.1`).

## Step 8 — Measure Human-Effort Reduction

Compare:

1. manual reading from scratch;
2. candidate ranking only;
3. candidate ranking plus clustering;
4. full human-in-the-loop system.

Measure:

* minutes per accepted skill;
* number of advertisements reviewed;
* precision of top-ranked suggestions (P@k for k=10, 20, 50);
* taxonomy coverage gained (`other` rate reduction);
* annotator agreement on accept/reject.

## Step 9 — Perform One Taxonomy-Update Cycle

Run the complete loop:

```text
taxonomy v2.0
    ↓
unknown discovery (Steps 3–6)
    ↓
human validation (Step 7)
    ↓
taxonomy v2.1
    ↓
retrain and reevaluate (all 13+new categories)
```

Measure whether v2.1 improves external coverage without reducing precision or annotation reliability. Report per-category delta and `other` rate.

## Step 10 — Frame the Paper as Taxonomy Maintenance

The contribution is not just unknown detection. It is:

* detection;
* linking;
* clustering;
* expert validation;
* versioning;
* measured coverage improvement;
* measured human-effort reduction.

Emphasise the **maintenance workflow**, not just the detector.

### Minimum publishable result

A benchmark containing genuine unknown skills, quantitative unknown-detection results (AUROC / P@k), expert validation of proposed additions, and at least one documented taxonomy-update cycle with measured coverage gain.

---

# Direction 5 — Temporal Skill Bundles and Role Convergence

## Main question

> Are analyst occupations becoming more similar over time, or are they separating into increasingly specialised skill bundles?

This direction requires substantially more longitudinal data.

## Step 1 — Audit the Existing Date Coverage

Use `first_seen` to inspect:

* earliest date;
* latest date;
* postings per month;
* postings per source;
* role-family distribution over time;
* missing periods.

Visualise with a heatmap (month × role_family) and a coverage table.

Do not begin trend analysis merely because a date column exists — the present data may cover too narrow or uneven a window for credible trends.

## Step 2 — Build a Longitudinal Corpus

Collect repeated cross-sections, for example:

```text
Period 1   2022-H1   (e.g. 1,500 postings)
Period 2   2022-H2   (e.g. 1,500 postings)
Period 3   2023-H1
Period 4   2023-H2
...
```

Each period should contain comparable:

* sources;
* role families;
* regions;
* seniority levels;
* company types (agency vs direct employer).

Avoid comparing one year of LinkedIn data with another year of mostly Indeed data without controlling for source. Match sampling fractions or include source as a covariate.

## Step 3 — Deduplicate Repostings Over Time

The same vacancy may appear repeatedly across months.

Create:

* exact duplicate groups (text hash);
* near-duplicate groups (TF-IDF / embedding similarity);
* company-title-location groups;
* likely reposting groups (same `job_link` or near-identical text within 60 days).

Decide whether to count:

* unique **vacancies** (deduplicated);
* unique **advertisement versions**;
* **posting events** (every observation).

For labour-demand analysis, **unique vacancies** are generally more defensible. Report all three as a sensitivity check.

## Step 4 — Annotate a Validation Sample from Every Period

Do not assume the model's 2024 accuracy applies to later years.

For every time period:

1. sample advertisements (≥ 50 per period, stratified);
2. annotate them manually (same guidelines, same 13 labels);
3. calculate per-period performance (macro-F1, per-category);
4. examine new terminology (neologisms, new tools);
5. check whether category definitions remain valid (concept drift).

This separates genuine labour-market change from model degradation.

## Step 5 — Estimate Skill Prevalence with Uncertainty

For each skill and period calculate:

* raw prevalence (model predictions);
* model-corrected prevalence (using validation confusion matrix / calibration);
* 95% confidence intervals (bootstrap or binomial with finite-sample correction);
* prevalence by role;
* prevalence by seniority;
* prevalence by region.

Where the classifier is imperfect, use its validation confusion matrix to estimate corrected prevalence (Rogan–Gladen or Bayesian correction). Report both raw and corrected.

## Step 6 — Model Temporal Change

For each skill, fit a model such as:

```text
skill_present ~ time + role_family + seniority + region + source + (1 | company)
```

Potentially include company or sector random effects.

Report:

* direction of change (increase / decrease / flat);
* effect size (odds ratio per year);
* 95% confidence interval;
* p-value with multiple-testing correction;
* whether the result survives source and seniority controls.

Do not rely only on line charts — every claimed trend needs a model and an uncertainty interval.

## Step 7 — Discover Skill Bundles

Instead of only counting individual skills, analyse **combinations**.

Possible methods:

* co-occurrence networks (skill × skill, weighted by phi or Jaccard);
* association rules (Apriori, support/confidence/lift);
* latent class analysis;
* non-negative matrix factorisation (posting × skill matrix);
* community detection (Louvain / Leiden on co-occurrence graph).

Examples of bundles might include:

```text
SQL + Power BI + Excel + reporting                    (traditional BI)
Python + statistics + machine learning                (data science)
Cloud + ETL + data modelling + SQL                    (analytics engineering)
Stakeholder communication + reporting + governance    (business-facing)
```

Track bundle prevalence over time — do bundles strengthen, weaken, merge, or split?

## Step 8 — Measure Occupational Convergence

For every period, calculate the distance between role-family skill distributions.

For example:

```text
distance(data analyst, analytics engineer)
distance(data analyst, business analyst)
distance(data scientist, analytics engineer)
```

Use a distribution-distance measure such as **Jensen–Shannon divergence** or cosine distance on the mean skill vector.

Interpretation:

* decreasing distance → convergence;
* increasing distance → specialisation.

Also inspect whether convergence concerns only shared tools (e.g. SQL, Excel) or deeper task capabilities (e.g. governance, modelling).

## Step 9 — Conduct Robustness Checks

Repeat the analysis:

* excluding recruitment agencies;
* using only companies observed in several periods (balanced panel);
* controlling for source;
* controlling for seniority;
* excluding duplicate templates;
* using only manually validated labels (not model predictions);
* using different trend windows (e.g. 12m vs 24m rolling).

If a trend survives most checks, it is more credible; if it vanishes when controlling for source, it was likely a dataset artefact.

## Step 10 — Write the Substantive Paper

The paper should answer:

1. which skills changed (and by how much, with uncertainty);
2. which bundles changed (emergence / decline);
3. which roles converged or diverged;
4. whether changes are technical, interpersonal or governance-related;
5. how much uncertainty comes from extraction error vs sampling variation.

Lead with the labour-market story; methods and validation are the foundation that makes the story credible.

### Minimum publishable result

Several genuinely comparable time periods (≥ 3, ideally 4+), manual validation within each period, source-controlled trend models, and analysis of **skill combinations** rather than only keyword frequencies.

---

# Direction 6 — Cross-Source or Cross-Country Transfer

## Main question

> Does a skill-extraction model trained on one labour market, platform or country remain reliable somewhere else?

## Step 1 — Choose One Primary Transfer Question

Do not combine every possible domain shift initially.

Choose **one**:

### Option A — Cross-source UK transfer

```text
UK LinkedIn → UK Indeed / Adzuna / other source
```

This isolates platform and writing-style differences. **Cleanest first paper.**

### Option B — Cross-country English transfer

```text
United Kingdom → United States
United Kingdom → Australia
```

This studies occupational and terminology differences without translation.

### Option C — Cross-language transfer

```text
English UK → another language (e.g. French, German)
```

This is much larger because translation and multilingual modelling become additional research questions. Save for later.

**Recommendation:** start with **Option A**.

## Step 2 — Harmonise the Datasets

Create a common schema:

```text
posting_id
source
country
job_title
company
location
date
job_summary
role_family           # provisional + human-confirmed
seniority             # human-confirmed
```

Normalise:

* HTML entities and tags;
* bullet points and list markers;
* duplicated headers/footers;
* salary information (mask or standardise);
* company boilerplate (detect via near-duplicate footer hashing);
* location formats;
* dates (ISO 8601).

Do not remove meaningful requirement text.

## Step 3 — Match the Samples

Distribution differences can make transfer results difficult to interpret.

Create a **matched** evaluation sample based on:

* role family;
* seniority;
* broad region;
* sector where possible;
* advertisement length (quantile-matched);
* publication period (same quarter).

This makes it clearer whether performance loss comes from source/platform or from occupational composition. Use propensity-score or coarsened exact matching; report standardised mean differences before/after matching.

## Step 4 — Independently Annotate Every Target Domain

Annotate a gold-standard sample from **each** source or country (≥ 150 per domain, stratified).

Use the same frozen guidelines initially (`taxonomy_v2.0`).

Record:

* examples that fit the taxonomy cleanly;
* examples requiring local terminology (e.g. US-specific tools, UK-specific regulation);
* categories that do not transfer (e.g. GDPR vs US privacy);
* new occupational concepts;
* disagreements caused by cultural or institutional context.

Report per-domain prevalence and inter-annotator agreement.

## Step 5 — Quantify the Distribution Shift

Measure differences in:

* vocabulary (top terms, TF-IDF divergence, embedding centroid distance);
* advertisement length (tokens, sentences);
* label prevalence (per category);
* role-family prevalence;
* common tools (lexicon hit rates);
* skill co-occurrence (correlation matrices);
* sentence structure (avg sentence length, bullet ratio);
* requirement wording ("must have" vs "desirable" ratios);
* company boilerplate (footer length, duplication rate).

This helps explain **why** transfer succeeds or fails — not just that it does.

## Step 6 — Run a Transfer Matrix

At minimum run:

| Training / Development | Test | Purpose |
|---|---|---|
| Source A | Source A | In-domain baseline |
| Source A | Source B | **Zero-shot transfer** (the key result) |
| Source B | Source B | Target-domain upper baseline |
| A + B | B | Pooled training |
| A + small B sample | B | Few-shot adaptation |

For few-shot adaptation, test increasing target-data amounts, such as:

```text
0 examples  → zero-shot
10 examples → few-shot
25 examples
50 examples
100 examples
```

This creates a useful **adaptation curve**. Plot macro-F1 vs target shots with bootstrap CIs.

## Step 7 — Compare Adaptation Techniques

Test:

* no adaptation (zero-shot);
* threshold recalibration (Platt / isotonic on small target sample);
* source-specific thresholds;
* lexicon expansion (add target-specific synonyms);
* importance weighting (domain-adversarial or instance weighting);
* fine-tuning on small target samples;
* prompt examples from the target source (few-shot LLM);
* hybrid lexical-semantic adaptation.

A strong finding may be that **simple threshold or lexicon adaptation recovers most of the lost performance** — this is a valuable, practical result.

## Step 8 — Evaluate Transfer Fairly

Report:

* absolute target-domain F1 (macro, micro, per-category);
* relative performance drop from in-domain testing (ΔF1);
* per-category transfer loss (which categories degrade most);
* calibration shift (ECE before/after);
* unseen-term performance (terms not in source lexicon);
* cost of adaptation (annotation hours, compute);
* number of target labels required to recover 90% of in-domain performance.

Example:

```text
In-domain macro-F1 (A→A):        0.91  [0.88, 0.93]
Zero-shot target (A→B):          0.72  [0.68, 0.76]   Δ = -0.19
50-example adaptation (A+50→B):  0.84  [0.81, 0.87]
100-example adaptation:           0.88  [0.85, 0.91]
Target-only upper bound (B→B):   0.90  [0.87, 0.93]
```

## Step 9 — Analyse Local Terminology and Taxonomy Mismatch

For every major transfer error, determine whether it comes from:

* new synonym (same skill, different wording);
* different product name;
* different occupational meaning (same term, different role);
* different regulation (GDPR vs SOX, etc.);
* different role composition (more/fewer BAs vs DAs);
* different advertisement style (terse vs verbose);
* taxonomy genuinely not applying (construct failure, not model failure).

This distinguishes **model failure** from **construct failure** — a critical distinction for labour-market research.

## Step 10 — Frame the Contribution around Portability

The paper should answer:

1. how much accuracy is lost under transfer (absolute + relative);
2. which categories transfer well vs poorly;
3. which categories do not transfer at all;
4. how much target annotation is needed to recover performance;
5. whether lexicon, transformer or LLM methods adapt most efficiently;
6. whether the taxonomy itself is portable or needs localisation.

### Minimum publishable result

Independent gold-standard samples from both domains (≥ 150 each), bidirectional or carefully matched transfer tests, a few-shot adaptation curve, and qualitative evidence explaining the transfer gap (terminology / taxonomy mismatch analysis).

---

# Minimum Viable Paper for Each Direction

| Direction | Minimum Evidence Needed | Main Risk |
|---|---|---|
| **1. Models under shift** | External test set, several model families (A–H), 95% bootstrap CIs, shift and error analysis | Becoming only another model leaderboard without behavioural insight |
| **2. Role-aware taxonomy** | New hierarchy (`taxonomy_v3.0`), double annotation, coverage and agreement improvement, role-aware vs universal model comparison | Adding categories without conceptual discipline (taxonomy bloat) |
| **3. Actual requirements** | Span + status annotations, hard negatives, contextual extraction models (A–E), labour-demand impact (mentioned vs required) | Annotation workload and overly complicated schema |
| **4. Open-set discovery** | Genuine unknown benchmark (artificial + natural), novelty detection (AUROC/P@k), expert-validated taxonomy additions, one update cycle | Treating every unfamiliar noun phrase as a skill |
| **5. Temporal bundles** | Multiple comparable periods (≥3), validation per period, source-controlled trend models, bundle analysis (not just keyword counts) | Confusing dataset drift with labour-market change |
| **6. Cross-domain transfer** | Gold data from both domains, transfer matrix, adaptation curve, taxonomy portability analysis | Domain samples not being genuinely comparable (confounding) |

---

# Recommended Execution Order

## First: Shared Foundation

Correct the evaluation, freeze the taxonomy, deduplicate the data, measure annotation agreement, and build the external test set.

**Deliverable:** `v4/` with `taxonomy_v2.0`, `corpus_dedup.csv`, reliability report, two external test sets, and the locked evaluation harness.

## Second: Direction 1

This is the most direct publication from the existing repository. It uses the current baseline as the central puzzle:

> Why does a transparent lexical method perform so strongly, and does it survive realistic distribution shift?

**Deliverable:** In-domain vs shift results, hybrid recommendation, cost analysis. Strongest immediate paper.

## Third: Direction 2 or Direction 3

Choose based on the intended research identity:

* **Direction 2** for occupational taxonomies, labour-market intelligence and information systems.
* **Direction 3** for a stronger core NLP and information-extraction contribution.

Recommendation: **Direction 2** if targeting labour-market / IS venues; **Direction 3** if targeting NLP / IE venues. Either builds naturally on Direction 1's error analysis.

## Fourth: Direction 4

Open-set discovery naturally follows the role-aware taxonomy or compositional-extraction work (needs a mature taxonomy to define "unknown").

## Fifth: Directions 5 and 6

These become stronger once the extraction system has already been externally validated and you possess additional time periods or sources.

The cleanest overall research programme is therefore:

```text
Reliable evaluation  (Part A)
        ↓
Interpretable vs semantic comparison  (Direction 1)
        ↓
Role-aware or contextual skill representation  (Direction 2 or 3)
        ↓
Open-set taxonomy maintenance  (Direction 4)
        ↓
Temporal and cross-domain labour-market studies  (Directions 5 & 6)
```

**Timeline guidance (indicative, post-foundation):**

| Phase | Direction | Effort | Dependencies |
|---|---|---|---|
| 1 | Foundation (Part A) | 4–6 weeks | None — do first |
| 2 | Direction 1 | 6–8 weeks | Foundation |
| 3 | Direction 2 or 3 | 8–12 weeks | Foundation + D1 error analysis |
| 4 | Direction 4 | 6–10 weeks | D2/D3 taxonomy + D1 models |
| 5 | Directions 5 & 6 | 12–20 weeks (parallelisable) | Foundation + D1 validation, extra data collection |

---

# Appendix A — Machine-Readable Config Template

Every experiment run should write a `run_config.json` alongside its results. Never rely on filenames or README text alone.

```json
{
  "run_id": "2026-08-06_weighted_lexicon_v2.0",
  "taxonomy_version": "2.0",
  "taxonomy_file": "v4/config/taxonomy_v2.0.json",
  "split_version": "external_test_v1",
  "split_seed": 42,
  "split_method": "stratified_group_kfold",
  "split_stratify_on": "role_family",
  "split_group_on": "template_group_id",
  "model": "weighted_lexicon",
  "model_variant": "B_corrected",
  "idf_fit_data": "development_only",
  "threshold_source": "nested_cv_dev_only",
  "thresholds": {
    "programming": 0.42,
    "sql": 0.38,
    "visualisation_bi": 0.35,
    "reporting": 0.51,
    "excel": 0.61,
    "statistics": 0.44,
    "machine_learning": 0.39,
    "data_cleaning": 0.48,
    "etl": 0.37,
    "data_modelling": 0.55,
    "cloud": 0.58,
    "stakeholder_comm": 0.46,
    "ethics_governance": 0.62
  },
  "categories": [
    "programming", "sql", "visualisation_bi", "reporting", "excel",
    "statistics", "machine_learning", "data_cleaning", "etl",
    "data_modelling", "cloud", "stakeholder_comm", "ethics_governance"
  ],
  "categories_order_matters": true,
  "random_seed": 42,
  "evaluation": {
    "metrics": ["macro_f1", "micro_f1", "per_category_prf", "pr_auc", "subset_accuracy", "hamming_loss", "ece"],
    "bootstrap_ci": { "method": "percentile", "n_resamples": 10000, "alpha": 0.05, "stratify_on": "posting" },
    "slices": ["role_family", "seniority", "source", "company_seen", "terminology_seen", "natural_vs_challenge"]
  },
  "environment": {
    "python": "3.11.0",
    "sklearn": "1.4.0",
    "commit": "dd7c0b7",
    "tag": "v3-original-baseline"
  }
}
```

Store one such file per run in `v4/experiments/<run_id>/run_config.json`. The `thresholds` block is written by `tune_thresholds` on dev only.

---

# Appendix B — Evaluation Harness Contract

Every method produces a CSV with **exactly** these columns, in this order, no extra columns:

```text
posting_id,
score_programming, score_sql, score_visualisation_bi, score_reporting, score_excel,
score_statistics, score_machine_learning, score_data_cleaning, score_etl,
score_data_modelling, score_cloud, score_stakeholder_comm, score_ethics_governance,
prediction_programming, prediction_sql, prediction_visualisation_bi, prediction_reporting, prediction_excel,
prediction_statistics, prediction_machine_learning, prediction_data_cleaning, prediction_etl,
prediction_data_modelling, prediction_cloud, prediction_stakeholder_comm, prediction_ethics_governance
```

* `score_*` ∈ [0, 1] (continuous, comparable across categories only after per-category thresholding).
* `prediction_*` ∈ {0, 1} (thresholded on dev thresholds).
* Column order follows `CATEGORIES` in `config.py` — **never reorder**.
* One row per `posting_id`; join to gold on `posting_id`.

The harness (`v4/evaluation/harness.py`) computes:

* macro-F1, micro-F1, per-category P/R/F1 + PR-AUC, support, predicted count;
* subset accuracy, Hamming accuracy/loss;
* calibration (ECE, reliability diagram);
* bootstrap 95% CIs (macro-F1, per-category F1);
* slice breakdowns (role, seniority, source, seen/unseen);
* runtime, memory, cost per 1k.

Output files (per variant):

```text
v4/experiments/<run_id>/
  run_config.json
  predictions.csv          # the contract above
  report_dev.csv           # per-category dev metrics
  report_test.csv          # per-category test/external metrics (written once)
  report_slices.csv        # slice breakdowns
  summary.json             # aggregate metrics + CIs + timings
```

---

# Appendix C — Provenance & Release Checklist

## Provenance

* `V1/`, `v2/` — frozen. Never edit, import from, or delete.
* `v3/` — tagged `v3-original-baseline` at commit `dd7c0b7`. Read-only after tag; provenance for the 0.937 macro-F1 claim.
* `v4/` — all new work (foundation fixes, external data, new methods). Branch from `v3-original-baseline` or from `main` immediately after the tag.

## Before collecting external test data

- [ ] Tag `v3-original-baseline` pushed.
- [ ] Taxonomy frozen (`taxonomy_v2.0` committed, `config.py` hash recorded).
- [ ] Lexicons, negative patterns, prompts, thresholds frozen.
- [ ] Evaluation pipeline corrected (TF-IDF fit, normalisation, dev-only tuning).
- [ ] Deduplication script and `template_group_id` assignment complete.
- [ ] Annotation guidelines versioned (`annotation_guidelines_v2.0.docx` → `v2.1` if revised post-reliability).

## Before publishing any external-test result

- [ ] Two external sets collected and independently annotated (natural + challenge).
- [ ] Inter-annotator agreement reported (kappa + agreement + prevalence).
- [ ] `run_config.json` written for every variant (no test-leaked tuning).
- [ ] External test scored **once** per locked model (no re-tuning on test).
- [ ] Bootstrap CIs and significance tests reported.
- [ ] Slice breakdowns and error analysis included.

## Release (per `DATA_NOTICE.md`)

- [ ] Code: MIT.
- [ ] Guidelines, taxonomy, labels, derived metadata: released.
- [ ] Posting identifiers + hashes: released.
- [ ] Full posting text: only if licence permits; otherwise hashes + scripts to rebuild.

---

## References (Internal)

* `v3/base-model/config.py` — taxonomy and lexicons (single source of truth).
* `v3/base-model/evaluate.py` — splitting, threshold tuning, scoring.
* `v3/base-model/tfidf_baseline.py` — baseline variants A (cosine) and B (weighted hit).
* `v3/manual_work/annotation_guidelines_v2_0.docx` — category definitions and boundary rules.
* `v3/manual_work/gold_standard_annotation_workbook_v2.xlsx` — gold standard (read-only).
* `docs/DATA_DICTIONARY.md` — column definitions and split contract.
* Attwood, S. & Williams, A. (2023) — TF-IDF mapping to CyBOK knowledge areas (adapted in variant A).

---

*End of Research Plan v1 — 5.6 Pro. Next step: execute Part A (Foundation) — tag `v3-original-baseline`, freeze `taxonomy_v2.0`, and correct `v4/evaluation/` before any new data collection.*

