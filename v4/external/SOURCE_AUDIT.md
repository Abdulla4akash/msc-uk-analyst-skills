# Source Audit — External UK Analyst Advertisements

**Date checked:** 2026-08-07
**Development source:** LinkedIn (v3/manual_work/uk_analyst_corpus_v4_clean.csv), 820 postings, 2024-01-12 to 2024-01-17, UK analyst roles.

**Requirement:** External set preferably **different source + newer dates** (source shift + temporal shift) for defensible generalisation estimate.

---

## Candidate Sources Considered

### 1. Reed API (reed.co.uk) — **SELECTED (pending API key)**
- **Access mechanism:** Official Reed Jobseeker API (https://www.reed.co.uk/developers) — `GET https://www.reed.co.uk/api/1.0/search` with HTTP Basic Auth (API key as username, blank password), paginated with `keywords`, `locationName`, `resultsToTake`, `resultsToSkip`
- **Details endpoint:** `GET https://www.reed.co.uk/api/1.0/jobs/{jobId}` for complete job description
- **Authentication:** HTTP Basic Auth, `auth=(api_key, "")` (Jobseeker API, not Recruiter API)
- **Date checked:** 2026-08-07
- **Fields available:** `jobId`, `jobTitle`, `locationName`, `date` (posting date), `jobDescription` (full HTML/text), `employerName`, `jobUrl`, `expirationDate`, `minimumSalary`/`maximumSalary`
- **Timestamp availability:** Yes (`date`)
- **UK coverage:** UK national, high volume
- **Estimated candidate count:** 1000+ UK analyst roles at any time; filtered to analyst titles, deduplicated, ~500-800 with full description
- **Licence/terms:** Requires API key; Terms state bulk redistribution of `jobDescription` verbatim is not permitted; research retention with attribution allowed, must respect rate limits, do not exceed 1 req/s, cache locally, no resale. Check latest https://www.reed.co.uk/policies
- **Redistribution limitations:** Full advert text restricted; **IDs, hashes, labels, categories, sampling metadata, source URLs (if permitted), derived features releasable**; store raw text locally gitignored (`v4/external/raw/`).
- **Selected/not selected:** **Selected as primary external source** (different from LinkedIn, stable IDs/URLs, sufficient text, timestamps, UK coverage, research use defensible)
- **Reason:** Best balance of legal API, stable IDs, full text, UK coverage, analyst role density; temporal shift achievable by requesting jobs after 2024-01-17; source shift from LinkedIn.

### 2. Adzuna API (adzuna.co.uk) — Viable alternative
- **Access mechanism:** Adzuna API (https://developer.adzuna.com) with `app_id`/`app_key`, endpoint `https://api.adzuna.com/v1/api/jobs/gb/search/{page}?app_id=...&app_key=...&what=analyst&where=UK`
- **Date checked:** 2026-08-07
- **Fields available:** `id`, `title`, `description`, `company.display_name`, `location.display_name`, `created`, `redirect_url`, `category.label`
- **Timestamp availability:** Yes (`created`)
- **UK coverage:** UK national
- **Estimated candidate count:** 1000+ analyst
- **Licence/terms:** Requires key; attribution required; forbids bulk redistribution of `description`; research allowed.
- **Redistribution limitations:** Full text restricted; derived releasable.
- **Selected/not selected:** Not selected as primary (redundancy), but suitable fallback if Reed inaccessible.
- **Reason:** Similar coverage to Reed; either provides source shift. Keep as fallback.

### 3. Find a Job (DWP) / GOV.UK — Not selected as primary
- **Access mechanism:** HTML search `https://findajob.dwp.gov.uk/search?loc=86383&pp=25&q=analyst` with pagination; no official public API as of 2026-08-07; `robots.txt` allows crawling but terms state scraping without permission may be disallowed; requires checking GOV.UK terms.
- **Date checked:** 2026-08-07
- **Fields available:** `title`, `location`, `description`, `datePosted`, `employer`, `url` (after following detail page)
- **Timestamp availability:** Yes (`datePosted`)
- **UK coverage:** UK national (government service)
- **Estimated candidate count:** High
- **Licence/terms:** Crown copyright, OGL 3.0 for GOV.UK content but **advert text is third-party employer content** not covered; redistribution of full text not permitted without employer permission.
- **Redistribution limitations:** Full text restricted.
- **Selected/not selected:** Not selected
- **Reason:** No stable API, higher terms uncertainty, requires HTML scraping which risks violating anti-bot/access controls; prefer API source.

### 4. ONS / UK Data Service / Open Jobs Dataset — Not selected as primary
- **Access mechanism:** Public open dataset via ONS or UK Data Service (application, approved researcher)
- **Date checked:** 2026-08-07
- **Fields available:** Depends on dataset; often limited to aggregated counts or truncated description; may lack full `job_summary` needed for model input.
- **Timestamp availability:** Varies
- **UK coverage:** UK
- **Estimated candidate count:** Variable; current open vacancy datasets insufficient for 300 analyst full-text postings.
- **Licence/terms:** Open data licence if published; generally permissive for research.
- **Redistribution limitations:** Depends on licence; often permissive for derived.
- **Selected/not selected:** Not selected
- **Reason:** Insufficient full job-description text for skill coding; coverage uncertain.

### 5. Indeed UK (indeed.co.uk) — Not selected
- **Access mechanism:** No public API; scraping Indeed violates `robots.txt` and terms (`Indeed prohibits scraping without permission`, uses anti-bot); requires publisher API which is not openly available for research bulk.
- **Selected/not selected:** Not selected — would violate terms.

### 6. LinkedIn (original source) — Excluded for external
- **Access mechanism:** LinkedIn API not open for bulk job-ad retrieval; scraping violates LinkedIn User Agreement and anti-bot.
- **Reason:** Must be **different source** from development to test generalisation; reuse would not test source shift.

---

## Selected Source for External Acquisition

**Primary:** `Reed API (reed.co.uk)` with API key (human-provided).
**Fallback:** `Adzuna API` if Reed unavailable.

Both provide:
- Stable posting IDs/URLs
- Full job-description text (HTML stripped to `job_summary` for model input, same preprocessing as development: plain text)
- Timestamps (`date`/`created`)
- UK location
- Enough analyst-role coverage
- Research use defensible with key, attribution, rate limiting, local caching, no verbatim redistribution.

---

## Acquisition Rules (from acquisition.py)

- Do NOT scrape in violation of access controls, terms, robots, anti-bot, paywalls.
- Prefer documented API; store raw JSON locally gitignored (`v4/external/raw/`, `*_raw.json`).
- Filter to UK: `locationName` contains United Kingdom / England/Scotland/Wales/UK city, or `searchCountry==United Kingdom`.
- Filter to analyst roles: `jobTitle` contains `analyst` (case-insensitive) or role-family keywords (business/data/finance analyst etc); include `data scientist`/`analytics engineer` where defined.
- Require full `jobDescription` present (≥200 characters, not truncated); strip HTML to text for `job_summary`.
- Deduplicate via `v4/external/dedup.py` (exact, normalised, TF-IDF near-duplicate) within and against 820 development postings.
- Temporal shift: **only advertisements with `date`/`created` > 2024-01-17** (after development window); ideally newer (e.g., 2024-02 onward, 2025-2026), source shift already satisfies but temporal adds strength.
- If exact development dates unavailable, source-held-out remains valid but document limitation (not applicable here — dates available).

---

## Rights / Retention

- **Raw text locally:** `v4/external/raw/reed_raw_*.json` and `v4/external/candidates.csv` (if full text) are **gitignored**.
- **Public repository:** release only what rights allow: `LOCKED_SAMPLE_MANIFEST.csv` with `external_id, source, source_posting_id, source_url(if permitted), acquired_at, published_at, role_family, challenge_stratum, text_sha256, normalised_sha256, duplicate_group_id`, plus `labels`, `categories`, `sampling metadata`, `code`, `agreement stats`, `predictions/metrics` — **no full advert text** unless source explicitly permits.

See `v4/external/DATA_RELEASE_PLAN.md`.

---

## Next Human Step

Human must:
1. Obtain Reed API key (free at reed.co.uk/developers) or Adzuna credentials.
2. Run `PYTHONPATH=. python3 msc-uk-analyst-skills/v4/external/acquisition.py` with key, respecting rate limits, to collect candidates after freeze.
3. Then sampling/dedup will produce locked E1/E2 manifests.

Until then, acquisition is **BLOCKED** — no fake external data is created.

---

*Audit completed before external sampling; do not commit copyrighted third-party advert text.*
