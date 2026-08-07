# Data Release / Privacy Plan

**Principle:** Do not assume repository MIT licence applies to third-party advert text. Source terms govern.

---

## Safe to Release (public repository)

Likely permitted, source-agnostic:

- All code: `v4/config.py`, `v4/methods/`, `v4/evaluation/`, `v4/external/*.py`, `v4/experiments/`, `v4/tests/`
- Taxonomy: `v4/config.py` (13 CATEGORIES, CATEGORY_LABELS, LEXICONS, NEGATIVE_PATTERNS) and hashes in `EXTERNAL_FREEZE_MANIFEST.json`
- Sampling code: `v4/external/sampling.py`, `dedup.py`, `acquisition.py` (structure, not raw text)
- Derived IDs/hashes: `external_id`, `source_posting_id` (if stable), `text_sha256`, `normalised_text_sha256`, `duplicate_group_id`
- Sampling metadata: `role_family_sampling_stratum`, `challenge_stratum`, `natural_or_challenge`, `sampling_seed`, `quotas`, `pool distribution`
- Labels: human-annotated gold matrix (posting_id × 13 categories) where permitted by source (most sources permit sharing derived labels)
- Categories, role-family metadata, agreement statistics (kappa, positive/negative agreement), adjudication logs (without raw text)
- Method predictions (binary matrices) and evaluation metrics (macro-F1, per-category P/R/F1, bootstrap CI)
- Manifest hashes: `LOCKED_SAMPLE_MANIFEST.csv` hash, `EXTERNAL_LABEL_LOCK.json`, `EXTERNAL_FREEZE_MANIFEST.json`

Include explicit licence note: "Third-party advert text not included; see source-specific restrictions."

---

## Potentially Restricted (do NOT commit to public repo)

- **Full advert text** (`job_summary`, `jobDescription`, `description`): copyrighted employer content, not covered by MIT. Reed/Adzuna terms forbid verbatim bulk redistribution. Even if source is GOV.UK Find a Job, advert text is third-party.
- **Company/source text** that reproduces advert verbatim
- **Source URLs** — check per-source terms: Reed/Adzuna `jobUrl`/`redirect_url` may be shareable as factual reference, but verify current terms before publishing URLs that expose full text via scraping.
- **Raw acquisition dumps**: `v4/external/raw/*`, `v4/external/candidates_with_text.csv`, `v4/external/annotation/*.xlsx` with full text

**Handling:**
- Store locally, **gitignored** (`v4/external/raw/`, `*_raw.json`, `*candidates*`, `*annotation*.xlsx` if contains text)
- Public `LOCKED_SAMPLE_MANIFEST.csv` should contain **hashes, not raw text** if redistribution restricted.
- For reproducibility, release `text_sha256` so others with access can verify.

---

## Source-Specific Notes

### Reed API
- Terms: bulk redistribution of `jobDescription` not permitted; local retention for research with attribution allowed; respect rate limits.
- Public: IDs/hashes/labels/metrics OK; URLs maybe OK (factual), but do not bundle full text.

### Adzuna API
- Similar: attribution required, description redistribution restricted.

### Find a Job / ONS
- GOV.UK OGL 3.0 covers site structure, **not third-party advert text**; do not republish full text.

---

## Gitignore Additions

Already in `.gitignore` (`results/`), add for external:

```
# External raw data — copyrighted third-party advert text, do not commit
v4/external/raw/
v4/external/candidates*.csv
v4/external/candidates*.json
v4/external/annotation/*.xlsx
v4/external/annotation/*.csv
external_raw/
*_raw.json
```

Public safe manifests (`LOCKED_SAMPLE_MANIFEST.csv` **without** raw text) may be committed if they contain only IDs/hashes/strata.

---

## Checklist Before Public Push

- [ ] Does commit contain any file with full `job_summary` text beyond 300 development corpus (which is already limited release)? If yes, gitignore it.
- [ ] Does `LOCKED_SAMPLE_MANIFEST.csv` contain only hashes/IDs/strata (no raw text) if source forbids?
- [ ] Are `EXTERNAL_FREEZE_MANIFEST.json`, protocol docs, sampling code free of raw text?
- [ ] Is `annotation` workbook gitignored if full text?

---

*When in doubt, keep raw text local and release derived artefacts.*
