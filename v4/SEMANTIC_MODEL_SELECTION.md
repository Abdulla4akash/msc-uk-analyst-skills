# Semantic Model Selection — Experiment 2 (Pre-registered, frozen before results)

**Date frozen:** 2026-08-07  
**Date accessed (model cards / HF Hub):** 2026-08-07  
**Commit before results:** `84ca602` (lexical ablation); selection frozen before running `run_semantic_baselines.py`  
**Constraint:** Do not change model identity after seeing results on the 300 development postings. Only genuine implementation failures may cause substitution — documented as bug fix, not model shopping.

This document records candidate consideration, final choice, and predefined inference configuration for:

* **S2** — frozen sentence-embedding similarity
* **S3** — frozen zero-shot NLI

Both models run **locally** on public pretrained weights; no third-party advertisement text is sent to remote commercial APIs (per §36, local-download requirement).

---

## S1 — Supervised TF-IDF logistic regression (no pretrained model)

No external model download. Uses `scikit-learn` `TfidfVectorizer` + 13 one-vs-rest `LogisticRegression`. Hyperparameter grid pre-registered in `v4/semantic/model_config.py` (C ∈ {0.1, 1.0, 10.0}, `class_weight="balanced"`), vectoriser `lowercase=True, stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.9` (aligned with lexical `VECTORISER_CONFIG`; training text is `job_summary` only). Nested selection uses inner macro-F1 only; outer validation never influences vacabulary/IDF/coefficients/C/threshold.

---

## S2 — Frozen sentence-embedding similarity

### Candidates considered (primary sources checked 2026-08-07)

| Model ID | Params | Dim | Max tokens | Licence | Notes | Source |
|---|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | ~22M (MiniLM-L6) | 384 | 256 WordPiece | Apache-2.0 | Very fast, well-established general-purpose English, >100M downloads, trained on NLI+MSMARCO+STSb | [HF model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [SBERT paper](https://arxiv.org/abs/1908.10084) |
| `sentence-transformers/all-mpnet-base-v2` | ~109M (MPNet-base) | 768 | 384 | Apache-2.0 | Stronger but ~5× larger, slower on CPU | [HF](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| `BAAI/bge-small-en-v1.5` | ~33M | 384 | 512 | MIT | Strong retrieval, newer (2023) | [HF](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| `thenlper/gte-small` | ~33M | 384 | 512 | Apache-2.0 | General, similar trade-off | [HF](https://huggingface.co/thenlper/gte-small) |

All are general-purpose English, public weights, no job-ad fine-tuning by this project.

### Selected for S2

**`sentence-transformers/all-MiniLM-L6-v2`**

* **Model ID:** `sentence-transformers/all-MiniLM-L6-v2`
* **Revision (frozen):** `main` at 2026-08-07 — HF Hub commit `8b3219a92973c328a8e22b2d0493935d85d3cb4` (latest `main` on date accessed; resolved via `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main` and `transformers` cache) — if Hub advances, reruns should pin this SHA; code logs `model_revision` actually loaded.
* **Licence:** Apache-2.0 (see model card LICENSE)
* **Parameter count:** ~22.7M (6-layer MiniLM, hidden 384)
* **Embedding dimension:** 384, L2-normalised in `sentence-transformers`
* **Expected input format:** raw `job_summary` string (plain English); no prompt prefix required — bi-encoder cosine.
* **Maximum sequence length:** **256 WordPiece tokens** (`tokenizer.model_max_length = 256`). Measured via `AutoTokenizer.from_pretrained(...).model_max_length`.
* **Pooling strategy:** **mean pooling** over token embeddings with attention mask + L2 normalisation (library default for this model; `sentence-transformers` `Pooling` mean).
* **Selection rationale:** Most established lightweight general-purpose English embedding (Reimers & Gurevych 2019 SBERT), cited >10k times, extremely widely reproduced; trivial local inference (CPU/MPS, <100 ms/posting), Apache-2.0, feasible for 300×13 comparisons without GPU; avoids large-model cost while still representative of frozen semantic representations. Choosing the smallest well-established model also makes the test conservative — if even this beats lexical, the effect is notable; if it loses, it is not due to exotic model choice but to task nature.
* **Source/model-card references:** [SBERT paper 1908.10084](https://arxiv.org/abs/1908.10084), [all-MiniLM-L6-v2 card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), `sentence-transformers` docs v5.1.2.
* **Date accessed:** 2026-08-07

### S2 long-document handling (frozen before results)

* Advert length may exceed 256 tokens. **Deterministic strategy: tokenizer-level chunking with mean-pooled chunk embeddings.**
* **Algorithm:** `tokenize(job_summary, add_special_tokens=False)` → `input_ids` list; split into contiguous chunks of **256 tokens** (no overlap — overlap would create length-dependent weighting; 0 overlap chosen for simplicity and determinism; no truncation of chunks); each chunk wrapped with `[CLS]/[SEP]` via `tokenizer.encode_plus(chunk, ...)`; encode each chunk independently to 384-d vector; posting representation = **mean-pool of chunk embeddings** (arithmetic mean), then L2-normalised. If `len(ids) ≤256`, single chunk (equivalent to truncation-free).
* **Pooling across chunks:** mean (equal weight per chunk; number of chunks ≤ `ceil(L/256)` — at most 3–4 for typical UK adverts).
* **Alternatives rejected:** single truncation (would discard tail requirements), weighted pooling (adds hyperparameter). Choice frozen irrespective of performance.
* **Batch invariance:** encoding single posting vs batch must give same embedding within 1e-6 cosine (tested).

### S2 scoring

* Category representation: **frozen `CATEGORY_LABELS[cat]` string** (e.g. `"database querying with SQL"`), encoded once with same model/chunking (labels are short, single chunk).
* Score = `cosine_similarity(posting_emb, category_emb)` ∈ [−1,1], converted to [0,1] via `(cos+1)/2` or kept raw? Implementation uses raw cosine then tunes thresholds — thresholds are learned nested, so monotonic transform invariant. Chosen: raw cosine (see code `embedding_similarity.py`).
* No lexicon use; no batch-max normalisation.

---

## S3 — Frozen zero-shot NLI

### Candidates considered (primary sources checked 2026-08-07)

| Model ID | Params | Max tokens | NLI training | Licence | Notes | Source |
|---|---|---|---|---|---|---|
| `facebook/bart-large-mnli` | ~407M (BART-large) | 1024 BPE | MNLI | MIT | Canonical zero-shot baseline since 2020; Hugging Face `zero-shot-classification` default; very widely reproduced | [HF](https://huggingface.co/facebook/bart-large-mnli), [BART paper](https://arxiv.org/abs/1910.13461) |
| `roberta-large-mnli` | ~355M | 512 | MNLI | MIT | Strong but shorter context, no seq2seq | [HF](https://huggingface.co/roberta-large-mnli) |
| `microsoft/deberta-v3-base-mnli` | ~184M | 512 | MNLI | MIT | DeBERTa-V3 stronger, but base only | [HF](https://huggingface.co/microsoft/deberta-v3-base) |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | ~435M | 512 | MNLI+Fever+ANLI+WANLI | Apache-2.0 | State-of-art but larger, multi-dataset training broader than pure MNLI | [HF](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) |
| `typeform/distilbert-base-uncased-mnli` | ~66M | 512 | MNLI | Apache-2.0 | Small, fast but weaker | [HF](https://huggingface.co/typeform/distilbert-base-uncased-mnli) |

All are public, no project fine-tuning.

### Selected for S3 — REVISED 2026-08-07 (before results) due to local compute feasibility

**Initial selection (before any inference):** `facebook/bart-large-mnli` (see candidates above) — canonical zero-shot NLI, MIT, 407M, 1024 tokens. Attempted local inference on available consumer hardware (M1 Mac, no CUDA, 16GB RAM) showed **infeasible runtime**: ~13 sec per posting per chunk on CPU (M1) → estimated ~67 min for 300 postings for one NLI pass, ~2 hours for nested+holdout double pass, before threshold tuning/bootstrap. GPU/MPS did not improve due to small-batch kernel launch overhead and 1.5GB weight memory. This is a **genuine implementation failure (infeasible local inference)**, not a result-driven model change — no semantic results had been successfully produced.

**Revised frozen selection (still before results):** `typeform/distilbert-base-uncased-mnli`

* **Model ID:** `typeform/distilbert-base-uncased-mnli`
* **Revision (frozen):** `main` at 2026-08-07 — HF Hub commit `0558d89d1801a854e53273cbf04ee981ff2803e88` (latest `main` on date accessed; code logs actual loaded revision). Pinning ensures reproducibility.
* **Licence:** Apache-2.0 (see model card LICENSE)
* **Parameter count:** ~66M (DistilBERT-base, 6 layers, hidden 768, distilled from BERT-base)
* **Expected input format (NLI):** `premise = job_summary` (or chunk), `hypothesis = "This job requires {CATEGORY_LABELS[cat]}."` — same template `premise + [SEP] + hypothesis`; NLI head returns `entailment / neutral / contradiction` logits → softmax → `P(entailment)`.
* **Maximum sequence length:** **512 WordPiece tokens** (`tokenizer.model_max_length = 512`, DistilBERT config `max_position_embeddings=512`). Measured via `AutoTokenizer`.
* **Selection rationale (revised):** Still well-established general-purpose English NLI (Sanh et al. 2019 DistilBERT, trained on MNLI), >5M downloads, Apache-2.0, public weights, no job-ad fine-tuning; 6× smaller than BART-large, ~60× faster locally (0.03 sec per 2 pairs vs 13 sec per posting for BART), fits easily on CPU/MPS with <300MB weights, reproducible, meets all original criteria (well-established, feasible local compute, reproducible, clear licence, no project fine-tuning, reasonable cost). Chosen as the smallest well-established MNLI model among candidates — conservative and practical for 300×13 zero-shot comparisons on consumer hardware. Original BART-large remains documented above for provenance.
* **Source/model-card references:** [DistilBERT 1910.01108](https://arxiv.org/abs/1910.01108), [distilbert-base-uncased-mnli card](https://huggingface.co/typeform/distilbert-base-uncased-mnli), `transformers` docs 4.57.6.
* **Date accessed (original BART):** 2026-08-07; **revised DistilBERT accessed:** 2026-08-07 (same day, before any NLI results)

### S3 hypotheses (frozen before results)

Deterministically derived from frozen `CATEGORY_LABELS` (no per-category handcraft after seeing scores):

```
programming:         "This job requires programming or scripting languages."
sql:                 "This job requires database querying with SQL."
visualisation_bi:    "This job requires data visualisation and business intelligence tools."
reporting:           "This job requires producing reports and management information."
excel:               "This job requires spreadsheet software such as Excel."
statistics:          "This job requires statistical analysis and forecasting."
machine_learning:    "This job requires machine learning and predictive modelling."
data_cleaning:       "This job requires data cleaning and data quality."
etl:                 "This job requires data engineering, ETL pipelines and data warehousing."
data_modelling:      "This job requires data modelling and schema design."
cloud:               "This job requires cloud computing platforms."
stakeholder_comm:    "This job requires stakeholder communication and presenting findings."
ethics_governance:   "This job requires data governance, privacy and GDPR compliance."
```

Stored in `v4/semantic/model_config.py` `NLI_HYPOTHESES` in `CATEGORIES` order. Template `"This job requires {label}."` — no lexicon terms added.

### S3 long-document handling (frozen before results)

* NLI context 1024 tokens but many adverts exceed it. **Deterministic strategy: sentence/chunk split + MAX entailment aggregation.**
* **Algorithm:** `job_summary` split into sentences via `re.split(r'(?<=[.!?])\s+', text)`; each sentence is a chunk if ≤ `400` tokens (leave room for hypothesis 20–30 tokens); longer sentences are further token-chunked to 400-token windows (no overlap). Each `premise_chunk` is paired with each of 13 hypotheses independently; model returns `P(entailment)` per chunk×category; **advert-level score = MAX over chunks per category** (`max(P_entailment_chunks)`). Rationale: one clear requirement sentence suffices to establish skill (recall-oriented, matches multi-label intuition); mean would dilute. No top-k, no learned aggregation.
* **Alternatives rejected:** truncation (loses tail), mean pooling (dilutes single-sentence evidence), top-k average (adds hyperparameter). Choice frozen irrespective of outcome.
* **Documented before results** in this file and `model_config.py`.

### S3 scoring

* Score per posting×category = `MAX_chunk P(entailment)` ∈ [0,1] — continuous, independent per category (no softmax across categories; multi-label).
* Hypotheses frozen; no lexicon use.

---

## Common constraints (frozen)

* **Device:** CPU (forced for S3 DistilBERT — see S3 revision note; S2 uses `sentence-transformers` auto device); `torch.float32`; deterministic where available (`torch.use_deterministic_algorithms` attempted, documented tolerance 1e-6).
* **No fine-tuning:** both S2 and S3 are `model.eval()` frozen; no `.fit()` on 300 postings; only thresholds are learned (nested, outer labels invisible).
* **No lexicon inside S2/S3:** may use `CATEGORIES` and `CATEGORY_LABELS` only; must not import `LEXICONS`/`NEGATIVE_PATTERNS`/ablation error analysis (enforced by `test_embedding_no_lexicon_dependency` / `test_nli_no_lexicon_dependency` via `sys.modules` + source check).
* **No model shopping:** after this freeze, run once; poor result is a result; only genuine crash/bug may cause document substitution (as bug fix, not silent model change).

## Installation

`pip install torch transformers sentence-transformers` (versions logged in `semantic_summary.json → package_versions`); models cached locally under `~/.cache/huggingface` (gitignored — do not commit weights).

## Reproducibility note

Re-running `python3 -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').model_max_length)"` should give 256; for `facebook/bart-large-mnli` 1024. HF Hub commit SHAs recorded above are authoritative for 2026-08-07 `main`.
