"""
Acquisition for external UK analyst advertisements — Reed Jobseeker API (real client).

Terminology fix: Uses Reed Jobseeker API (https://www.reed.co.uk/developers), not Recruiter API.
Endpoints:
  GET https://www.reed.co.uk/api/1.0/search
  GET https://www.reed.co.uk/api/1.0/jobs/{jobId}
Auth: HTTP Basic Auth, api_key as username, blank password: auth=(api_key, "")

The 820 development postings must NOT be reused as external.
"""

import hashlib
import json
import re
import os
import time
import html as html_lib
from pathlib import Path
from datetime import datetime, timezone, date
import pandas as pd
import requests
from typing import Optional, List, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()

CANDIDATE_SOURCES = [
    {
        "source": "Reed API (reed.co.uk)",
        "access_mechanism": "Reed Jobseeker API (https://www.reed.co.uk/developers) — GET https://www.reed.co.uk/api/1.0/search with HTTP Basic Auth (API key as username, blank password), paginated with keywords, locationName, resultsToTake, resultsToSkip; details via GET https://www.reed.co.uk/api/1.0/jobs/{jobId} (Jobseeker API, not Recruiter API)",
        "fields_available": "jobId, jobTitle, locationName, date, jobDescription, employerName, jobUrl, expirationDate, minimumSalary, maximumSalary, currency, contractType, jobType",
        "timestamp_availability": "date field (posting date)",
        "uk_coverage": "UK national",
        "licence_terms": "Requires API key via Basic Auth; terms forbid verbatim bulk redistribution of jobDescription; research retention allowed with attribution, rate limiting, local caching",
        "est_candidate_count": "1000+ analyst roles in UK at any time",
        "redistribution": "Full text restricted; IDs/hashes/labels releasable"
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

# Frozen query list before bulk collection (do not add based on model results)
REED_QUERIES = [
    "data analyst",
    "business analyst",
    "business intelligence analyst",
    "BI analyst",
    "insight analyst",
    "insights analyst",
    "reporting analyst",
    "MI analyst",
    "finance analyst",
    "financial analyst",
    "commercial analyst",
    "marketing analyst",
    "data scientist",
    "analytics engineer",
]

REED_SEARCH_BASE = "https://www.reed.co.uk/api/1.0/search"
REED_DETAILS_BASE = "https://www.reed.co.uk/api/1.0/jobs"

# ---------- Helpers ----------
def get_reed_api_key() -> str:
    key = os.environ.get("REED_API_KEY")
    if not key:
        raise RuntimeError("REED_API_KEY not set in environment; export REED_API_KEY and retry (never commit the key)")
    # Do not log key
    return key

def strip_html(text: str) -> str:
    if not text:
        return ""
    # Unescape html entities first
    text = html_lib.unescape(text)
    # Replace block tags with newlines to preserve paragraphs/lists
    # Simple approach: replace <br>, <p>, <li>, <ul>, <ol>, <div> etc with newline
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(p|div|h[1-6]|li|tr|ul|ol)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li[^>]*>", "- ", text, flags=re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace but keep paragraphs
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()

def is_uk_location(location_name: str) -> bool:
    if not location_name:
        return False
    low = location_name.lower()
    # Must be UK; reject clearly non-UK (if location contains USA, Australia etc but not UK)
    # Simple: if contains uk, england, scotland, wales, northern ireland, or known UK cities without other country
    uk_indicators = ["united kingdom", "uk", "england", "scotland", "wales", "northern ireland", "london", "manchester", "birmingham", "leeds", "glasgow", "bristol", "liverpool", "edinburgh", "cardiff", "belfast", "sheffield", "nottingham", "southampton", "newcastle", "leicester", "coventry", "bradford", "derby"]
    # If location explicitly mentions non-UK country, reject
    non_uk = ["united states", "usa", "australia", "canada", "germany", "france", "ireland,"]  # careful
    for n in non_uk:
        if n in low and "united kingdom" not in low:
            return False
    for ind in uk_indicators:
        if ind in low:
            return True
    # If location is like "London" alone, assume UK in Reed context with locationName=United Kingdom
    # But if query used United Kingdom, results should be UK anyway; be permissive
    return True

def is_analyst_title(title: str) -> bool:
    if not title:
        return False
    low = title.lower()
    # Must be analyst-adjacent inclusion scope
    analyst_terms = ["analyst", "analytics", "data scientist", "insight", "reporting", "mi ", "business intelligence", "bi "]
    for term in analyst_terms:
        if term.strip() in low:
            return True
    return False

# ---------- Core client ----------
def reed_get(
    endpoint: str,
    *,
    params: dict = None,
    api_key: str = None,
    timeout: float = 30.0,
) -> dict:
    """
    GET with Basic Auth (api_key, ""), raise_for_status, retries for 429/5xx/network.
    Never includes key in URL or logs.
    """
    if api_key is None:
        api_key = get_reed_api_key()
    # Build URL
    # endpoint is like "https://www.reed.co.uk/api/1.0/search" or ".../jobs/{id}"
    # Do not put key in params
    max_retries = 3
    backoff = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(
                endpoint,
                params=params,
                auth=(api_key, ""),
                timeout=timeout,
            )
            # Never log key
            if resp.status_code in (401, 403):
                raise RuntimeError(f"Reed authentication failed: HTTP {resp.status_code} (check REED_API_KEY, not logged)")
            if resp.status_code == 404:
                # For details, 404 is permanent missing job
                resp.raise_for_status()
            if resp.status_code == 429:
                if attempt < max_retries:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
            if 500 <= resp.status_code < 600:
                if attempt < max_retries:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
            # For other 4xx, do not retry indefinitely
            resp.raise_for_status()
            # Parse JSON
            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Malformed JSON from Reed {endpoint}: {e}")
            return data
        except requests.exceptions.RequestException as e:
            # Network/timeout or raise_for_status for 429/5xx already handled
            # For 401/403/404 we already raised RuntimeError, not retry
            if "authentication failed" in str(e).lower():
                raise
            if isinstance(e, requests.exceptions.HTTPError):
                # Check status code
                status = getattr(e.response, "status_code", None)
                if status in (401, 403, 404):
                    raise RuntimeError(f"Reed request failed: HTTP {status} for {endpoint} (redacted)")
                # For other 4xx, don't retry
                if status and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"Reed request failed: HTTP {status} for {endpoint} (redacted)")
            # Transient network or 429/5xx: retry if attempts left
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"Reed request failed after {max_retries} retries for {endpoint}: {type(e).__name__} (redacted, key not logged)")
    raise RuntimeError(f"Reed request exhausted retries for {endpoint} (redacted)")

def search_reed_jobs(
    *,
    keywords: str,
    location_name: str = "United Kingdom",
    results_to_take: int = 100,
    results_to_skip: int = 0,
    api_key: str = None,
) -> dict:
    """
    Search Reed Jobseeker API.
    Correct endpoint: GET https://www.reed.co.uk/api/1.0/search
    Params: keywords, locationName, resultsToTake, resultsToSkip, etc.
    """
    if api_key is None:
        api_key = get_reed_api_key()
    params = {
        "keywords": keywords,
        "locationName": location_name,
        "resultsToTake": int(results_to_take),
        "resultsToSkip": int(results_to_skip),
    }
    return reed_get(REED_SEARCH_BASE, params=params, api_key=api_key)

def get_reed_job_details(
    job_id: object,
    *,
    api_key: str = None,
) -> dict:
    """
    Details: GET https://www.reed.co.uk/api/1.0/jobs/{jobId}
    """
    if api_key is None:
        api_key = get_reed_api_key()
    endpoint = f"{REED_DETAILS_BASE}/{job_id}"
    return reed_get(endpoint, params=None, api_key=api_key)

def acquire_reed_candidate_pool(
    *,
    queries: list,
    target_candidates: int = 800,
    api_key: str = None,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    This helper is used for bulk search pagination (search only).
    It collects unique job IDs across queries, respecting per-query caps.
    For full detail fetching, use the checkpointed loop in the script.
    Returns DataFrame of search hits (not yet detailed).
    """
    if api_key is None:
        api_key = get_reed_api_key()
    all_hits = []
    seen_ids = set()
    for q in queries:
        results_to_skip = 0
        per_query_count = 0
        per_query_cap = 500  # reasonable cap per query
        while True:
            data = search_reed_jobs(keywords=q, location_name="United Kingdom", results_to_take=100, results_to_skip=results_to_skip, api_key=api_key)
            results = data.get("results") or data.get("Results") or []
            # Reed returns results as list under "results"
            if not results:
                break
            for r in results:
                jid = r.get("jobId") or r.get("JobId") or r.get("id")
                if jid is None:
                    continue
                if jid not in seen_ids:
                    seen_ids.add(jid)
                    # keep raw hit with query_origin
                    r["_query_origin"] = q
                    all_hits.append(r)
                    per_query_count += 1
            # Check pagination stop
            total_results = data.get("totalResults") or data.get("TotalResults") or data.get("totalResultsCount") or 0
            # If returned count < requested, no more pages
            if len(results) < 100:
                break
            results_to_skip += 100
            if results_to_skip >= total_results:
                break
            if per_query_count >= per_query_cap:
                break
            if len(seen_ids) >= target_candidates:
                # Continue to collect a bit more for dedup headroom, but check outer
                pass
            time.sleep(sleep_seconds)  # be conservative between search pages
            if len(seen_ids) >= target_candidates and per_query_count >= 100:
                # Allow early break if target reached, but finish current query's dedup
                if len(seen_ids) >= target_candidates:
                    break
        # Respect target overall
        if len(seen_ids) >= target_candidates:
            break
        time.sleep(sleep_seconds)
    df = pd.DataFrame(all_hits)
    return df

# ---------- Audit and legacy helpers ----------
def audit_development_source():
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
    audit = audit_development_source()
    src = next((s for s in CANDIDATE_SOURCES if s["source"]==source_name), None)
    if src is None:
        return {"status": "BLOCKED", "reason": f"source {source_name} not in audit list", "candidate_df": None}
    if source_name in ["Reed API (reed.co.uk)", "Reed API"] and api_key is None:
        try:
            api_key = get_reed_api_key()
        except Exception as e:
            return {
                "status": "BLOCKED",
                "reason": f"Acquisition requires API key for {source_name}; {e}",
                "candidate_df": None,
                "source_audit": src,
                "development_audit": audit
            }
    return {
        "status": "BLOCKED",
        "reason": "Use acquire_reed_candidate_pool and detail fetching with REED_API_KEY; see acquisition.py real client",
        "candidate_df": None,
        "source_audit": src,
        "development_audit": audit
    }

def acquisition_blocker_report():
    audit = audit_development_source()
    return {
        "state": "B",
        "blocked_reason": "External acquisition requires human-provided API key and legal source access; automated acquisition not executed to avoid terms violation. Protocol, sampling, dedup, annotation package are ready.",
        "development_audit": audit,
        "candidate_sources": CANDIDATE_SOURCES,
        "recommended_source": "Reed API (Jobseeker, GET /api/1.0/search with Basic Auth) or Adzuna API with key (source shift from LinkedIn, newer dates, UK analyst coverage)",
        "next_human_step": "Obtain API key, run acquisition.py with proper rate limiting, then sampling/dedup/annotation"
    }
