"""
Real Reed acquisition pipeline — bulk collection with checkpointing.

Usage:
  REED_API_KEY=... PYTHONPATH=. python3 msc-uk-analyst-skills/v4/external/run_reed_acquisition.py

This script is the post-freeze operational stage (not part of frozen methods).
It does NOT evaluate frozen models against external gold.
"""

import os
import json
import time
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone, date
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from v4.external.acquisition import (
    REED_QUERIES, search_reed_jobs, get_reed_job_details, get_reed_api_key,
    strip_html, is_uk_location, is_analyst_title
)
from v4.external.dedup import compute_dedup, dedup_against_development
from v4.external.sampling import assign_role_family
from v4.tests._paths import CORPUS_PATH

# ---------- Config ----------
TARGET_UNIQUE_SEARCH_HITS = 1000
TARGET_ELIGIBLE_CANDIDATES = 700  # aim 600-800
PER_QUERY_CAP = 500
SLEEP_SEARCH = 1.0
SLEEP_DETAILS = 1.0
DEVELOPMENT_CUTOFF = date(2024, 1, 17)

def parse_reed_date(s):
    if not s:
        return None
    # Reed date formats: "01/08/2026" or "2026-08-01T..." or "01/08/2026 12:00:00"
    s = str(s).strip()
    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(s.split("T")[0] if "T" in s else s, fmt)
            return dt.date()
        except:
            continue
    # Try parsing with dateutil fallback
    try:
        # Extract dd/mm/yyyy
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(y, mo, d)
        m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m2:
            y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            return date(y, mo, d)
    except:
        pass
    return None

def is_newer_than_cutoff(posted_date_str):
    d = parse_reed_date(posted_date_str)
    if d is None:
        return False
    return d > DEVELOPMENT_CUTOFF

def main():
    print("=== Reed acquisition pipeline ===")
    # Verify key without exposing
    try:
        key = get_reed_api_key()
        print(f"REED_API_KEY is loaded (len={len(key)}, not displayed)")
    except Exception as e:
        print(f"Key error: {e}")
        return 1
    # Setup paths
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    raw_dir = REPO_ROOT / "v4" / "external" / "raw"
    private_dir = REPO_ROOT / "v4" / "external" / "private"
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    search_raw_path = raw_dir / f"reed_search_raw_{today}.jsonl"
    details_raw_path = raw_dir / f"reed_details_raw_{today}.jsonl"
    candidates_private_path = private_dir / "reed_candidates_private.csv"
    checkpoint_path = private_dir / "reed_details_checkpoint.json"
    progress_log = private_dir / "acquisition_progress.log"

    # ---------- Search ----------
    print(f"Starting search across {len(REED_QUERIES)} queries, target {TARGET_UNIQUE_SEARCH_HITS} unique hits")
    all_hits = []
    seen_ids = set()
    # Load existing checkpoint if exists
    if search_raw_path.exists():
        print(f"Found existing search raw {search_raw_path}, will append")
    # For each query
    for qi, query in enumerate(REED_QUERIES):
        print(f"[{qi+1}/{len(REED_QUERIES)}] Query: '{query}'")
        results_to_skip = 0
        per_query = 0
        while True:
            try:
                data = search_reed_jobs(keywords=query, location_name="United Kingdom", results_to_take=100, results_to_skip=results_to_skip, api_key=key)
            except Exception as e:
                print(f"  search error for '{query}' skip {results_to_skip}: {type(e).__name__} (redacted)")
                break
            # Log raw
            with open(search_raw_path, "a") as f:
                f.write(json.dumps({"query": query, "resultsToSkip": results_to_skip, "response": data}) + "\n")
            results = data.get("results") or data.get("Results") or []
            total = data.get("totalResults") or data.get("TotalResults") or 0
            if not results:
                print(f"  no results at skip {results_to_skip}, total {total}")
                break
            new_ids = 0
            for r in results:
                jid = r.get("jobId")
                if jid is None:
                    continue
                if jid not in seen_ids:
                    seen_ids.add(jid)
                    r["_query_origin"] = query
                    all_hits.append(r)
                    new_ids += 1
                    per_query += 1
            print(f"  skip {results_to_skip}: got {len(results)}, new unique {new_ids}, total unique {len(seen_ids)}, per_query {per_query}, totalResults {total}")
            # Log progress
            with open(progress_log, "a") as pf:
                pf.write(f"{datetime.now(timezone.utc).isoformat()} query='{query}' skip={results_to_skip} got={len(results)} new={new_ids} total_unique={len(seen_ids)}\n")
            if len(results) < 100:
                break
            results_to_skip += 100
            if results_to_skip >= total:
                break
            if per_query >= PER_QUERY_CAP:
                print(f"  per-query cap {PER_QUERY_CAP} reached")
                break
            if len(seen_ids) >= TARGET_UNIQUE_SEARCH_HITS and per_query >= 100:
                # Allow to finish current query but check overall
                pass
            time.sleep(SLEEP_SEARCH)
            # Early break if overall target far exceeded? Continue to get diversity
        time.sleep(SLEEP_SEARCH)
        if len(seen_ids) >= TARGET_UNIQUE_SEARCH_HITS:
            print(f"Reached target unique {TARGET_UNIQUE_SEARCH_HITS}, continuing to finish remaining queries for diversity but will cap")
            # Don't break entirely, continue for diversity but we can break if we have enough
            if qi >= 5 and len(seen_ids) >= TARGET_UNIQUE_SEARCH_HITS + 200:
                print("Enough diversity, stopping early")
                break
    print(f"Search complete: {len(seen_ids)} unique job IDs from {len(all_hits)} hits")
    # Save hits summary
    hits_df = pd.DataFrame(all_hits)
    if not hits_df.empty:
        hits_df.to_csv(private_dir / "reed_search_hits_private.csv", index=False)
        print(f"Saved search hits to {private_dir / 'reed_search_hits_private.csv'}")

    # ---------- Details fetching with checkpoint ----------
    print(f"Fetching details for {len(seen_ids)} unique IDs with checkpoint {checkpoint_path}")
    # Load checkpoint
    checkpoint = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
            print(f"Loaded checkpoint with {len(checkpoint)} entries")
        except:
            checkpoint = {}
    # Determine IDs to fetch
    ids_to_fetch = [jid for jid in seen_ids if str(jid) not in checkpoint]
    print(f"Need to fetch {len(ids_to_fetch)} details (already have {len(checkpoint)} checkpointed)")

    # Fetch details
    fetched = 0
    failed = 0
    for idx, jid in enumerate(sorted(ids_to_fetch)):
        try:
            details = get_reed_job_details(jid, api_key=key)
            # Store raw
            with open(details_raw_path, "a") as f:
                f.write(json.dumps({"jobId": jid, "details": details}) + "\n")
            checkpoint[str(jid)] = {"status": "success", "fetched_at": datetime.now(timezone.utc).isoformat()}
            fetched += 1
        except Exception as e:
            # Distinguish 404 vs other
            err_type = type(e).__name__
            # Never log key
            checkpoint[str(jid)] = {"status": "failed", "error": err_type, "at": datetime.now(timezone.utc).isoformat()}
            failed += 1
            # print without key
            if "404" in str(e):
                print(f"  {jid}: 404 not found")
            elif "401" in str(e) or "403" in str(e):
                print(f"  {jid}: auth error {err_type} (redacted)")
            else:
                print(f"  {jid}: {err_type} (redacted)")
        # Checkpoint save every 50
        if (idx + 1) % 50 == 0:
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
            print(f"  progress {idx+1}/{len(ids_to_fetch)} fetched {fetched} failed {failed}")
            with open(progress_log, "a") as pf:
                pf.write(f"{datetime.now(timezone.utc).isoformat()} details progress {idx+1}/{len(ids_to_fetch)} fetched={fetched} failed={failed}\n")
        time.sleep(SLEEP_DETAILS)
        # Save checkpoint periodically
        if (idx + 1) % 100 == 0:
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
    # Final save
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
    print(f"Details fetching complete: fetched {fetched}, failed {failed}, checkpoint {len(checkpoint)}")

    # ---------- Normalize and filter ----------
    print("Normalizing and filtering candidates...")
    # Load all details from checkpoint and raw
    # We need to collect details that succeeded
    # Instead of re-reading raw, we can iterate checkpoint and fetch details from raw file if needed
    # Simpler: read details_raw file and build dict
    details_by_id = {}
    if details_raw_path.exists():
        with open(details_raw_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    jid = obj.get("jobId")
                    det = obj.get("details")
                    if jid and det:
                        details_by_id[str(jid)] = det
                except:
                    continue
    # Also need to handle details that were fetched but not in raw due to previous runs? Use checkpoint's success IDs
    candidates = []
    for jid_str, det in details_by_id.items():
        # Normalize fields
        job_title = det.get("jobTitle") or det.get("job_title") or ""
        employer = det.get("employerName") or det.get("employer_name") or ""
        location = det.get("locationName") or det.get("location_name") or ""
        posted = det.get("date") or det.get("datePosted") or det.get("postedDate") or det.get("posted_date") or det.get("date_posted") or ""
        exp = det.get("expirationDate") or det.get("expiration_date") or ""
        desc_html = det.get("jobDescription") or det.get("job_description") or det.get("description") or ""
        job_summary = strip_html(desc_html)
        # Filters
        if not job_summary or len(job_summary) < 200:
            continue
        if not is_analyst_title(job_title):
            continue
        if not is_uk_location(location):
            continue
        if not is_newer_than_cutoff(posted):
            continue
        # Build normalized record
        # Find query_origin from hits
        q_origin = ""
        for h in all_hits:
            if str(h.get("jobId")) == jid_str:
                q_origin = h.get("_query_origin", "")
                break
        rec = {
            "source": "reed",
            "source_posting_id": jid_str,
            "job_title": job_title,
            "employer_name": employer,
            "location_name": location,
            "posted_date": posted,
            "expiration_date": exp,
            "minimum_salary": det.get("minimumSalary"),
            "maximum_salary": det.get("maximumSalary"),
            "yearly_minimum_salary": det.get("yearlyMinimumSalary"),
            "yearly_maximum_salary": det.get("yearlyMaximumSalary"),
            "currency": det.get("currency"),
            "salary_type": det.get("salaryType"),
            "contract_type": det.get("contractType"),
            "job_type": det.get("jobType"),
            "external_url": det.get("jobUrl") or det.get("job_url") or f"https://www.reed.co.uk/jobs/view/{jid_str}",
            "reed_url": f"https://www.reed.co.uk/jobs/view/{jid_str}",
            "job_summary": job_summary,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "query_origin": q_origin,
            "role_family_sampling_stratum": assign_role_family(job_title),
        }
        candidates.append(rec)
    print(f"After filtering: {len(candidates)} eligible candidates (from {len(details_by_id)} detailed)")
    if candidates:
        df_cand = pd.DataFrame(candidates)
        # Save private
        df_cand.to_csv(candidates_private_path, index=False)
        print(f"Saved candidates to {candidates_private_path} (private, gitignored)")
        # Also save date distribution
        print("Date range candidates:", df_cand["posted_date"].min(), "to", df_cand["posted_date"].max())
        print("Role family distribution:", df_cand["role_family_sampling_stratum"].value_counts().to_dict())
    else:
        print("No eligible candidates after filtering!")
        return 1
    # Log final
    with open(progress_log, "a") as pf:
        pf.write(f"{datetime.now(timezone.utc).isoformat()} candidates eligible={len(candidates)}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
