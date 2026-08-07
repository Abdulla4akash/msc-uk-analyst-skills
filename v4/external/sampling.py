"""
Sampling for external sets E1 (natural, N=200) and E2 (challenge, N=100).

E1 — NATURAL EXTERNAL SET
  N=200, primary external generalisation estimate.
  Selection MUST NOT use model predictions or model confidence.
  Allowed: job_title, UK location, source, posting date, role-family inclusion, duplicate info.
  Forbidden: A1/S1/S2/S3/hybrid scores or future labels.
  Deterministic stratified sampling with fixed seed, documented quotas.

E2 — CHALLENGE SET (post-hoc, non-natural, N=100)
  Strata:
   C1 lexical-low-coverage / semantic-disagreement (~40)
   C2 lexical ambiguity / homonym (~30)
   C3 role/terminology edge (~30)
  Uses frozen MODEL OUTPUTS to identify difficult unlabelled cases (explicitly stress test).
  Pre-registered strata, deterministic rules, not hand-picked to favour a method.
"""

import hashlib
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()

# ----- Role family assignment (reuse development concept) -----
ROLE_FAMILY_KEYWORDS = {
    "business analyst": [r"business analyst", r"business\s+analy"],
    "data analyst": [r"data analyst", r"data\s+analy"],
    "finance analyst": [r"finance analyst", r"financial analyst", r"fp&a", r"finance\s+analy"],
    "data scientist": [r"data scientist", r"machine learning"],
    "marketing analyst": [r"marketing analyst", r"marketing\s+analy"],
    "analytics engineer": [r"analytics engineer", r"analytics\s+engineer"],
}

def assign_role_family(job_title: str) -> str:
    if not job_title:
        return "other"
    low = job_title.lower()
    for fam, pats in ROLE_FAMILY_KEYWORDS.items():
        for pat in pats:
            if re.search(pat, low):
                return fam
    # fallback: if contains analyst generic
    if "analyst" in low:
        return "other_analyst"
    return "other"

# ----- E1 Natural Sampling (no model scores) -----
def sample_E1(candidate_df: pd.DataFrame, n_target=200, seed=42, min_per_family=2):
    """
    Deterministic stratified sampling for E1 natural set.

    candidate_df must contain at least:
      - external_id (unique)
      - job_title
      - role_family (or will be derived)
      - source
      - posting_date / published_at
      - job_location

    MUST NOT contain model score columns. This function asserts that.

    Returns selected DataFrame with sampling metadata.
    """
    # Assert no model score columns leaked into E1 sampling
    forbidden_cols = [c for c in candidate_df.columns if any(k in c.lower() for k in ["a1_score", "s1_score", "s2_score", "s3_score", "semantic", "hybrid", "prediction", "confidence", "threshold"])]
    # Also check explicit names
    for col in ["A1", "S1", "S2", "S3", "A1_score", "S1_score", "S2_score", "S3_score"]:
        if col in candidate_df.columns:
            raise ValueError(f"E1 sampling must not use model predictions: column {col} present")
    if forbidden_cols:
        raise ValueError(f"E1 sampling must not use model scores: forbidden columns {forbidden_cols}")

    if "role_family" not in candidate_df.columns:
        candidate_df = candidate_df.copy()
        candidate_df["role_family"] = candidate_df["job_title"].apply(assign_role_family)

    # Measure candidate pool distribution
    pool_dist = candidate_df["role_family"].value_counts().to_dict()
    # Define quotas: proportional but with minimum coverage
    total = len(candidate_df)
    # Compute proportional allocation
    quotas = {}
    remaining = n_target
    # First ensure minimum per family where possible
    families = list(pool_dist.keys())
    for fam in families:
        avail = pool_dist[fam]
        # minimum is min(min_per_family, avail, remaining)
        q = min(min_per_family, avail, remaining)
        quotas[fam] = q
        remaining -= q
    # Distribute remaining proportionally to pool share
    if remaining > 0:
        remaining_pool = {fam: pool_dist[fam] - quotas[fam] for fam in families}
        total_remaining_pool = sum(remaining_pool.values())
        for fam in families:
            if total_remaining_pool > 0:
                share = remaining_pool[fam] / total_remaining_pool
                add = int(round(share * remaining))
                # cap by availability
                add = min(add, remaining_pool[fam])
                quotas[fam] += add
        # Adjust to exactly n_target due to rounding
        allocated = sum(quotas.values())
        # Simple adjustment: add/subtract from largest families
        diff = n_target - allocated
        sorted_fams = sorted(families, key=lambda f: pool_dist[f], reverse=True)
        i=0
        while diff != 0 and i < len(sorted_fams)*2:
            fam = sorted_fams[i % len(sorted_fams)]
            if diff > 0 and quotas[fam] < pool_dist[fam]:
                quotas[fam] +=1
                diff -=1
            elif diff < 0 and quotas[fam] > min_per_family:
                quotas[fam] -=1
                diff +=1
            i+=1

    # Deterministic sampling per stratum
    rng = np.random.default_rng(seed)
    selected_rows = []
    for fam, q in quotas.items():
        sub = candidate_df[candidate_df["role_family"]==fam]
        if len(sub) < q:
            # take all
            selected_rows.append(sub)
        else:
            # deterministic shuffle via rng
            idx = rng.choice(len(sub), size=q, replace=False)
            # Use iloc with sorted idx for reproducibility? Keep rng order but sort for stable manifest?
            selected_rows.append(sub.iloc[np.sort(idx)])
    # If still less than target due to pool shortage, fill from remaining pool
    selected = pd.concat(selected_rows) if selected_rows else pd.DataFrame()
    if len(selected) < n_target:
        # fill randomly from remaining candidates not yet selected
        remaining_pool_df = candidate_df[~candidate_df["external_id"].isin(selected["external_id"])]
        need = n_target - len(selected)
        if len(remaining_pool_df) >= need:
            idx = rng.choice(len(remaining_pool_df), size=need, replace=False)
            selected = pd.concat([selected, remaining_pool_df.iloc[np.sort(idx)]])
        else:
            selected = pd.concat([selected, remaining_pool_df])
    # Shuffle final selected deterministically for manifest order? Keep sorted by external_id for stability
    selected = selected.sort_values("external_id").reset_index(drop=True)
    metadata = {
        "sampling_seed": seed,
        "sampling_frame_size": int(len(candidate_df)),
        "role_family_quotas": quotas,
        "pool_distribution": pool_dist,
        "selected_n": int(len(selected)),
        "role_distribution_selected": selected["role_family"].value_counts().to_dict(),
        "selection_rule": "deterministic stratified sampling on role_family, no model scores, seed 42",
        "challenge_or_natural": "natural (E1), primary external generalisation set"
    }
    return selected, metadata

# ----- E2 Challenge Sampling (post-hoc, uses frozen model outputs) -----
# This is intentionally non-natural and must be reported separately.

# C2 ambiguity frozen text rules (from ablation)
C2_AMBIGUITY_RULES = {
    "excel_collision": [r"\bexcellent\b", r"\bexcellence\b", r"\bexcelled\b"],
    "reporting_reports_to": [r"reports?\s+to\b", r"direct reports?\b"],
    "pipeline_ambiguity": [r"demand pipeline", r"(?:drug|product|sales|candidate)\s+pipeline"],
    "clinical_programme": [r"clinical coding", r"wellbeing programme", r"\bprogramme\b"],
    "cloud_ambiguity": [r"sales cloud", r"service cloud", r"oracle cloud services"],
    "oracle_hyperion": [r"oracle\s+(?:fusion|epm|epbcs|pbcs)", r"hyperion"],
}

def matches_ambiguity(text: str) -> dict:
    low = text.lower()
    hits = {}
    for name, pats in C2_AMBIGUITY_RULES.items():
        for pat in pats:
            if re.search(pat, low, flags=re.IGNORECASE):
                hits[name] = True
                break
    return hits

def sample_E2_challenge(candidate_df: pd.DataFrame,
                        texts: list,
                        ids: list,
                        model_scores: dict,  # dict with keys "A1_pred", "S1_scores", "S2_scores", "S3_scores" etc, frozen
                        n_target=100,
                        seed=42):
    """
    Challenge set sampling using frozen model outputs (explicitly post-hoc).

    candidate_df must align with texts/ids.

    model_scores: dict containing frozen outputs for challenge identification only.
      Example: {"A1_pred": (n,13), "S2_scores": (n,13), "S3_scores": (n,13), ...}
      These are allowed here because challenge set is stress test and labelled separately.

    Pre-registered strata:
      C1 ~40 lexical-low-coverage / semantic-disagreement
      C2 ~30 lexical ambiguity / homonym (frozen text rules)
      C3 ~30 role/terminology edge (underrepresented families, low coverage, novel companies)

    Returns selected DataFrame with challenge_stratum column, and metadata.
    """
    n = len(candidate_df)
    # Ensure deterministic
    rng = np.random.default_rng(seed)
    # Prepare hits
    # C1: lexical low coverage AND/OR A1 negative while multiple semantic positive
    # Need frozen thresholds for semantic to decide positive; we use tuned thresholds from manifest if available,
    # otherwise simple 0.5 placeholder? For pre-registration, we define rule as:
    #  A1 coverage = (# categories where A1_pred==1)/13 ; low if <0.15 (i.e., <2 categories)
    #  OR semantic disagreement: A1_pred==0 but S2>=thr_S2 and S3>=thr_S3 for same category (>=2 semantics agree)
    # For manifest generation we will use manifest thresholds if available, else 0.5.

    # Load thresholds if manifest exists (for reproducibility)
    try:
        import json
        from pathlib import Path
        mani = json.load(open(Path(__file__).resolve().parents[1] / "EXTERNAL_FREEZE_MANIFEST.json"))
        thr_S2 = np.array(mani["S2"]["thresholds"])
        thr_S3 = np.array(mani["S3"]["thresholds"])
        thr_S1 = np.array(mani["S1"]["thresholds"])
    except Exception:
        thr_S2 = np.full(13, 0.5)
        thr_S3 = np.full(13, 0.5)
        thr_S1 = np.full(13, 0.5)

    A1_pred = model_scores.get("A1_pred")
    S1_scores = model_scores.get("S1_scores")
    S2_scores = model_scores.get("S2_scores")
    S3_scores = model_scores.get("S3_scores")

    c1_flags = []
    for i in range(n):
        flag=False
        if A1_pred is not None:
            cov = float((A1_pred[i]==1).sum())/13.0
            if cov < 0.15:
                flag=True
            # semantic disagreement: A1 0 but >=2 semantics positive
            if S2_scores is not None and S3_scores is not None:
                # count per category where A1 0 but S2>=thr and S3>=thr
                for cat in range(13):
                    if A1_pred[i,cat]==0:
                        s2pos = int(S2_scores[i,cat] >= thr_S2[cat])
                        s3pos = int(S3_scores[i,cat] >= thr_S3[cat])
                        s1pos = int(S1_scores[i,cat] >= thr_S1[cat]) if S1_scores is not None else 0
                        if (s2pos + s3pos + s1pos) >=2:
                            flag=True
                            break
        c1_flags.append(flag)

    c2_flags = [bool(matches_ambiguity(t)) for t in texts]
    # C3: role terminology edge: underrepresented families or low A1 coverage or novel
    # Use candidate_df role_family if present else assign
    if "role_family" not in candidate_df.columns:
        candidate_df = candidate_df.copy()
        candidate_df["role_family"] = candidate_df["job_title"].apply(assign_role_family)
    rare_families = {"marketing analyst", "analytics engineering", "data scientist", "other", "other_analyst"}
    c3_flags = []
    for i, fam in enumerate(candidate_df["role_family"]):
        flag = False
        if fam in rare_families:
            flag=True
        if A1_pred is not None:
            if float((A1_pred[i]==1).sum()) < 2:
                flag=True
        c3_flags.append(flag)

    # Build strata assignments (allow overlap, but assign priority C1>C2>C3 for quota)
    strata = []
    for i in range(n):
        if c1_flags[i]:
            strata.append("C1_lexical_low_coverage_semantic_disagreement")
        elif c2_flags[i]:
            strata.append("C2_lexical_ambiguity_homonym")
        elif c3_flags[i]:
            strata.append("C3_role_terminology_edge")
        else:
            strata.append("unassigned")

    candidate_df = candidate_df.copy()
    candidate_df["challenge_stratum_candidate"] = strata

    # Now select quotas: 40 C1, 30 C2, 30 C3
    quotas = {"C1_lexical_low_coverage_semantic_disagreement": 40,
              "C2_lexical_ambiguity_homonym": 30,
              "C3_role_terminology_edge": 30}
    selected_parts = []
    for stratum, q in quotas.items():
        pool = candidate_df[candidate_df["challenge_stratum_candidate"]==stratum]
        if len(pool) ==0:
            continue
        if len(pool) <= q:
            selected_parts.append(pool)
        else:
            idx = rng.choice(len(pool), size=q, replace=False)
            selected_parts.append(pool.iloc[np.sort(idx)])
    selected = pd.concat(selected_parts) if selected_parts else pd.DataFrame()
    # If still less than 100, fill from remaining unassigned or other strata
    if len(selected) < n_target:
        remaining = candidate_df[~candidate_df["external_id"].isin(selected["external_id"])]
        need = n_target - len(selected)
        if len(remaining) >= need:
            idx = rng.choice(len(remaining), size=need, replace=False)
            selected = pd.concat([selected, remaining.iloc[np.sort(idx)]])
        else:
            selected = pd.concat([selected, remaining])
    selected = selected.sort_values("external_id").reset_index(drop=True)
    # Assign final challenge_stratum (the one we sampled for, or fallback)
    # Ensure column exists
    if "challenge_stratum" not in selected.columns:
        selected["challenge_stratum"] = selected["challenge_stratum_candidate"]
    metadata = {
        "sampling_seed": seed,
        "quotas_target": quotas,
        "selected_counts": selected["challenge_stratum"].value_counts().to_dict() if "challenge_stratum" in selected.columns else {},
        "total_selected": int(len(selected)),
        "selection_rule": "challenge set uses frozen model outputs to identify difficult unlabelled cases, explicitly post-hoc, reported separately, not mixed with E1",
        "challenge_or_natural": "challenge (E2), post-hoc stress test, not natural population",
        "c1_definition": "low A1 coverage <0.15 or A1 negative while >=2 semantics positive",
        "c2_definition": "frozen ambiguity text rules: excellent/Excel, reports to, pipeline, clinical coding/programme, Sales/Service Cloud, Oracle/Hyperion",
        "c3_definition": "underrepresented role families or low lexical coverage or novel companies"
    }
    return selected, metadata
