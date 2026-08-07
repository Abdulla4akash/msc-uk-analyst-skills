"""
Acquisition for external UK analyst advertisements.

Priority sources (different from development LinkedIn):
1. Different UK job-ad source (e.g., Reed, Adzuna, Find a Job/DWP, ONS, Indeed UK via API, Guardian Jobs)
2. Legally/technically accessible (documented API, open dataset, institutional access, permissive research source)
3. Stable posting IDs/URLs
4. Sufficient full job-description text
5. Timestamps
6. UK location
7. Enough analyst-role coverage
8. Research use and retention defensible

Do NOT scrape in violation of access controls, terms, robots, anti-bot, paywalls.
Prefer documented API / open dataset / institutional / permissive source.

This module provides code structure for acquisition but does NOT fabricate external data.
If acquisition is blocked (no legal source, access denied), it reports blocker rather than faking.

The 820 development postings must NOT be reused as external.
"""

import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()

CANDIDATE_SOURCES = [
    {
        "source": "Reed API (reed.co.uk)",
        "access_mechanism": "Reed API (https://www.reed.co.uk/developers) with API key, paginated job search",
        "fields_available": "jobId, jobTitle, locationName, date, jobDescription, employerName, jobUrl, expirationDate",
        "timestamp_availability": "date field (posting date)",
        "uk_coverage": "UK national",
        "licence_terms": "Requires API key, terms forbid redistribution of full advert text verbatim; research retention allowed with attribution, check current terms",
        "est_candidate_count": "1000+ analyst roles in UK at any time",
        "redistribution": "Full text redistribution restricted; IDs/hashes/labels releasable"
    },
    {
        "source": "Adzuna API (adzuna.co.uk)",
        "access_mechanism": "Adzuna API (https://developer.adzuna.com) with app_id/app_key",
        "fields_available": "id, title, description, company, location, created, redirect_url, category",
        "timestamp_availability": "created field",
        "uk_coverage": "UK national",
        "licence_terms": "API terms require attribution, forbid bulk redistribution of descriptions; research use allowed with key",
        "est_candidate_count": "1000+ analyst roles",
        "redistribution": "Full text restricted; derived data releasable"
    },
    {
        "source": "Find a Job (DWP) / GOV.UK",
        "access_mechanism": "No public API; HTML search with pagination; check robots.txt and terms; may require scraping permission",
        "fields_available": "title, location, description, datePosted, employer, url",
        "timestamp_availability": "datePosted",
        "uk_coverage": "UK national (government service)",
        "licence_terms": "Crown copyright, Open Government Licence 3.0 for metadata but advert text is third-party employer content; redistribution of full text not permitted without permission",
        "est_candidate_count": "High",
        "redistribution": "Full text restricted"
    },
    {
        "source": "ONS / Open Jobs Dataset (if available)",
        "access_mechanism": "Public open dataset via ONS or UK Data Service (requires application)",
        "fields_available": "Depends on dataset; often limited description; may lack full text",
        "timestamp_availability": "Varies",
        "uk_coverage": "UK",
        "licence_terms": "Open data licence if published; research use generally permissive",
        "est_candidate_count": "Variable, may be insufficient",
        "redistribution": "Depends on licence; often permissive for derived"
    },
]

def audit_development_source():
    """
    Audit development corpus source coverage.
    Returns dict for temporal/source-shift doc.
    """
    import pandas as pd
    from v4.tests._paths import CORPUS_PATH, GOLD_PATH
    from v4.evaluation.data import load_gold_with_texts
    corpus = pd.read_csv(CORPUS_PATH)
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    gold_ids = set(gold_df["posting_id"].astype(str))
    corp_sub = corpus[corpus["posting_id"].astype(str).isin(gold_ids)]
    return {
        "development_source": "LinkedIn (via v3/manual_work/uk_analyst_corpus_v4_clean.csv, job_link contains linkedin.com)",
        "development_earliest": str(corpus["first_seen"].min()),
        "development_latest": str(corpus["first_seen"].max()),
        "gold_earliest": str(corp_sub["first_seen"].min()),
        "gold_latest": str(corp_sub["first_seen"].max()),
        "gold_n": int(len(corp_sub)),
        "corpus_n": int(len(corpus)),
        "acquisition_context": "Snapshot of 820 UK analyst postings filtered from LinkedIn search in Jan 2024 (12-17 Jan), 300 annotated",
        "company_coverage_sample": corp_sub["company"].value_counts().head(10).to_dict(),
        "location_coverage_sample": corp_sub["job_location"].value_counts().head(5).to_dict(),
        "role_family_provisional": corpus["role_family_provisional"].value_counts().to_dict(),
        "note": "Temporal window extremely narrow (6 days); external should use newer advertisements + different source for shift"
    }

def acquire_external_candidates(source_name="Reed API", uk_filter=True, role_query="analyst", max_candidates=500, api_key=None, out_path=None):
    """
    Attempt to acquire external candidates from selected source.

    This function is a STUB that documents acquisition logic without violating terms.
    It does NOT scrape illegally. It expects caller to provide API key and respect rate limits.

    If source requires API key and none provided, it returns BLOCKED status.

    Returns dict with status, candidate_df (if acquired), and metadata.
    """
    # This is intentionally not implementing live HTTP scraping without key.
    # Instead we provide structure and return BLOCKED if no key / no access.
    audit = audit_development_source()
    # Check for selected source
    src = next((s for s in CANDIDATE_SOURCES if s["source"]==source_name), None)
    if src is None:
        return {"status": "BLOCKED", "reason": f"source {source_name} not in audit list", "candidate_df": None}

    if source_name in ["Reed API (reed.co.uk)", "Adzuna API (adzuna.co.uk)"] and api_key is None:
        return {
            "status": "BLOCKED",
            "reason": f"Acquisition requires API key for {source_name}; no key provided. Human must obtain key and run acquisition with proper attribution and rate limiting. Do not scrape without key. See v4/external/SOURCE_AUDIT.md",
            "candidate_df": None,
            "source_audit": src,
            "development_audit": audit
        }
    # If we had a key, we would implement paginated API calls here, respecting terms:
    # - Use official API endpoint with query role_query and location UK
    # - Filter to job_title containing analyst (case-insensitive)
    # - Collect jobId, title, description (job_summary), company, location, date, url
    # - Deduplicate via dedup.py
    # - Filter to UK location (search_country==United Kingdom or locationName contains UK)
    # - Ensure full job_description present (not truncated)
    # - Respect pagination and rate limits, store raw JSON locally gitignored
    # For now, since no key, we return blocked.
    return {
        "status": "BLOCKED",
        "reason": "No acquisition executed in automated environment; human step required to provide API key and run with legal access. See SOURCE_AUDIT.md and DATA_RELEASE_PLAN.md",
        "candidate_df": None,
        "source_audit": src,
        "development_audit": audit
    }

def save_candidate_pool(candidate_df, out_path):
    """
    Save candidate pool locally (gitignored if contains full text).
    Public repo should only release derived IDs/hashes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(out_path, index=False)
    print(f"Saved candidate pool to {out_path} ({len(candidate_df)} rows) — ensure gitignored if full text present")

def acquisition_blocker_report():
    """
    Generate blocker report for State B when external data cannot be acquired today.
    """
    audit = audit_development_source()
    return {
        "state": "B",
        "blocked_reason": "External acquisition requires human-provided API key and legal source access; automated acquisition not executed to avoid terms violation. Protocol, sampling, dedup, annotation package are ready.",
        "development_audit": audit,
        "candidate_sources": CANDIDATE_SOURCES,
        "recommended_source": "Reed API or Adzuna API with key (source shift from LinkedIn, newer dates, UK analyst coverage)",
        "next_human_step": "Obtain API key, run acquisition.py with proper rate limiting, then sampling/dedup/annotation"
    }
