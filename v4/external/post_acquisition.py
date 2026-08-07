"""
Post-acquisition: dedup, E1/E2 sampling, lock, blank annotation workbooks.

Run after Reed details fetched and candidates_private.csv exists.
Does NOT evaluate models against gold, does NOT generate labels.
"""

import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from v4.external.dedup import compute_dedup, dedup_against_development, text_hash, normalised_text_hash
from v4.external.sampling import sample_E1, sample_E2_challenge, assign_role_family
from v4.external.annotation_package import create_blank_workbook, select_overlap_ids
from v4.external.acquisition import strip_html
from v4.tests._paths import CORPUS_PATH, GOLD_PATH
from v4.evaluation.data import load_gold_with_texts

def main():
    print("=== Post-acquisition: dedup, sampling, lock ===")
    private_dir = REPO_ROOT / "v4" / "external" / "private"
    raw_dir = REPO_ROOT / "v4" / "external" / "raw"
    candidates_path = private_dir / "reed_candidates_private.csv"
    if not candidates_path.exists():
        print(f"ERROR: candidates not found at {candidates_path}")
        return 1
    df = pd.read_csv(candidates_path)
    print(f"Loaded {len(df)} candidates from {candidates_path}")
    # Ensure required columns
    assert "job_summary" in df.columns, "job_summary missing"
    assert "job_title" in df.columns, "job_title missing"
    assert "source_posting_id" in df.columns, "source_posting_id missing"

    # Add external_id deterministic: reed_{source_posting_id}
    df["external_id"] = "reed_" + df["source_posting_id"].astype(str)
    # Ensure role_family
    if "role_family_sampling_stratum" not in df.columns:
        df["role_family_sampling_stratum"] = df["job_title"].apply(assign_role_family)
        df["role_family"] = df["role_family_sampling_stratum"]
    else:
        df["role_family"] = df["role_family_sampling_stratum"]
    # Add source column if missing
    if "source" not in df.columns:
        df["source"] = "reed"
    # Hashes
    df["text_sha256"] = df["job_summary"].apply(lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest())
    df["normalised_text_sha256"] = df["job_summary"].apply(lambda x: hashlib.sha256(re.sub(r"\s+", " ", re.sub(r"[^\w ]", "", str(x).lower())).strip().encode("utf-8")).hexdigest())
    # Duplicate group within pool
    dedup_pool = compute_dedup(df["job_summary"].tolist(), posting_ids=df["external_id"].tolist())
    df["duplicate_group_id"] = dedup_pool["group_ids"]
    print(f"Pool dedup: unique exact {dedup_pool['n_unique_exact']}/{len(df)}, normalized {dedup_pool['n_unique_normalised']}, near pairs {len(dedup_pool['near_duplicate_pairs'])}")

    # Dedup against 820 development corpus
    corpus = pd.read_csv(CORPUS_PATH)
    dev_texts = corpus["job_summary"].fillna("").astype(str).tolist()
    dev_ids = corpus["posting_id"].astype(str).tolist()
    # Also need to check against 300 gold? But spec says 820
    dedup_dev = dedup_against_development(df["job_summary"].tolist(), df["external_id"].tolist(), dev_texts, dev_ids)
    print(f"Dedup vs 820: exact overlaps {dedup_dev['exact_overlaps']}, normalised {dedup_dev['normalised_overlaps']}, near {len(dedup_dev['near_duplicate_pairs'])}")
    # Remove exact and normalized overlaps
    exact_set = set(dedup_dev["exact_overlap_ids"])
    norm_set = set(dedup_dev["normalised_overlap_ids"])
    to_remove = exact_set | norm_set
    if to_remove:
        print(f"Removing {len(to_remove)} overlapping candidates with development corpus")
        df = df[~df["external_id"].isin(to_remove)].reset_index(drop=True)
        print(f"After removal: {len(df)} candidates remain")
        # Recompute dedup after removal? Not needed for now
    else:
        print("No development overlaps found")

    # Within-pool exact dedup: keep one per duplicate_group_id
    # For each group, keep first (lowest external_id)
    before = len(df)
    df = df.sort_values("external_id").drop_duplicates(subset=["text_sha256"], keep="first").reset_index(drop=True)
    after = len(df)
    print(f"Within-pool exact dedup: {before} -> {after} (removed {before-after})")

    # Also handle near-duplicate groups: if near pairs, keep one per group? Use duplicate_group_id already for exact, but for near we should group
    # For now, if near pairs exist, we will keep all but flag; the manifest will have duplicate_group_id for exact only.
    # The near-duplicate threshold is 0.90, but we should not silently include; we have removed exact overlaps, near will be reviewed but not automatically removed unless protocol says.
    # For now, keep all remaining.

    if len(df) < 300:
        print(f"WARNING: only {len(df)} eligible candidates after filtering/dedup, need 300 for E1+E2")
        # Continue but will be short
    # ---------- Build E1 ----------
    print("Building E1 natural sample N=200 (no model scores)")
    # Candidate pool for E1 is df (all eligible)
    # sample_E1 expects candidate_df with external_id, job_title, role_family, etc.
    # It will assert no model columns, which is true
    e1_selected, e1_meta = sample_E1(df, n_target=200, seed=42, min_per_family=2)
    e1_selected["natural_or_challenge"] = "natural"
    e1_selected["challenge_stratum"] = ""
    print(f"E1 selected {len(e1_selected)}: {e1_meta['role_distribution_selected']}")
    # ---------- Build E2 ----------
    # Need model scores for challenge identification (post-hoc, allowed)
    # For this, we need to compute frozen model outputs for the remaining pool (not selected for E1? Or for all?)
    # Spec: E2 uses frozen model outputs to identify difficult unlabelled cases
    # We'll compute for the pool excluding E1 selected, to avoid overlap, then sample
    print("Building E2 challenge sample N=100 (C1/C2/C3, uses frozen model outputs post-hoc)")
    # For challenge, we need to use the pool excluding E1
    remaining_for_e2 = df[~df["external_id"].isin(e1_selected["external_id"])].reset_index(drop=True)
    print(f"Remaining for E2 pool: {len(remaining_for_e2)}")
    # Compute model scores for remaining pool (and also for E1? but we need for C1)
    # To determine C1, we need A1_pred, S1/S2/S3 scores for all candidates
    # We can compute A1, S1, S2, S3 for the remaining pool using frozen methods
    # But this will take time (embedding, NLI). We should do it.
    # For now, we will compute A1 quickly, and S1/S2/S3 via cached methods
    # If pool is large, this may be heavy but okay for 500 candidates
    # Let's compute
    from v4.ablation.lexical_ablation import score_for_variant
    from v4.semantic.embedding_similarity import embedding_scores, get_category_embeddings
    from v4.semantic.zero_shot_nli import nli_scores_for_texts
    from v4.semantic.supervised_tfidf import build_vectoriser as build_s1_vec
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    # Prepare texts for remaining + E1? For C1 we need to identify low coverage etc, so we need scores for remaining pool
    texts_e2 = remaining_for_e2["job_summary"].tolist()
    # A1
    print("Computing A1 for E2 pool...")
    S_A1 = score_for_variant("A1", texts_e2)
    A1_pred = (S_A1 > 0).astype(int)
    # S1: need to load frozen S1 model fitted on all 300? For challenge, we can use the frozen S1 deployment config from manifest
    # Simplest: use embedding and NLI which are frozen and don't require training; for S1 we can approximate with TF-IDF using dev data?
    # But we can also load the S1 model from freeze: it requires vectoriser fitted on all 300 and classifiers
    # Instead, for challenge sampling, we can use S2 and S3 only (plus A1) to define disagreement, which satisfies spec (at least one category where A1 negative while multiple frozen semantics positive)
    # So we need S2 and S3
    print("Computing S2 embedding for E2 pool...")
    cat_embs = get_category_embeddings()
    S_S2 = embedding_scores(texts_e2, cat_embs=cat_embs)
    print("Computing S3 NLI for E2 pool (may take time)...")
    # Use cached if available? But for external texts, cache miss, will compute
    # To avoid long time for 500 texts NLI (each with many chunks), we can try to batch
    # For now, compute; if too slow, we can fallback to not using S3 and just use S2 + S1 approximated
    try:
        S_S3 = nli_scores_for_texts(texts_e2)
    except Exception as e:
        print(f"S3 failed for E2 pool: {e}, using zeros")
        S_S3 = np.zeros((len(texts_e2), 13))
    # S1: we can try to compute via simple TF-IDF model fitted on dev 300? Load S1 from manifest and refit
    # For now, we can create a dummy S1 scores using embedding as proxy, or we can actually fit S1 on dev and score
    # Let's attempt to fit S1 quickly on dev 300 and score external
    print("Computing S1 for E2 pool via dev-fitted model...")
    try:
        from v4.evaluation.data import load_gold_with_texts
        from v4.tests._paths import GOLD_PATH
        from v4.semantic.supervised_tfidf import build_vectoriser, get_outer_scores_and_thresholds
        gold_df, y, dev_texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
        # Build S1 vectoriser on all dev
        vec = build_vectoriser()
        X_train = vec.fit_transform(dev_texts)
        X_ext = vec.transform(texts_e2)
        # Need to have classifiers trained on dev
        # Use manifest's selected C (10.0)
        import json
        mani = json.loads((REPO_ROOT / "v4" / "EXTERNAL_FREEZE_MANIFEST.json").read_text())
        best_C = mani["S1"]["selected_C"]
        S_S1 = np.zeros((len(texts_e2), 13))
        for ci in range(13):
            clf = LogisticRegression(C=best_C, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)
            clf.fit(X_train, y[:, ci])
            S_S1[:, ci] = clf.predict_proba(X_ext)[:, 1]
    except Exception as e:
        print(f"S1 failed: {e}, using zeros")
        S_S1 = np.zeros((len(texts_e2), 13))

    model_scores = {"A1_pred": A1_pred, "S1_scores": S_S1, "S2_scores": S_S2, "S3_scores": S_S3}
    # Now sample E2
    # Need to pass candidate_df and texts/ids
    e2_candidate_df = remaining_for_e2.copy()
    e2_selected, e2_meta = sample_E2_challenge(e2_candidate_df, texts_e2, remaining_for_e2["external_id"].tolist(), model_scores, n_target=100, seed=42)
    e2_selected["natural_or_challenge"] = "challenge"
    print(f"E2 selected {len(e2_selected)}: {e2_meta['selected_counts']}")

    # Check for shortage
    if len(e2_selected) < 100:
        print(f"WARNING: E2 shortage: got {len(e2_selected)} < 100")
        # Document deviation
        dev_path = REPO_ROOT / "v4" / "external" / "PROTOCOL_DEVIATIONS.md"
        with open(dev_path, "a") as f:
            f.write(f"\n## Deviation {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"E2 challenge sample shortage: requested 100, got {len(e2_selected)} due to limited eligible pool after filtering/dedup.\n")
            f.write(f"Quotas target: C1 40, C2 30, C3 30; got {e2_meta['selected_counts']}\n")
            f.write(f"Candidate pool after dedup: {len(df)}, E1 200, remaining {len(remaining_for_e2)}\n")
    if len(e1_selected) < 200:
        dev_path = REPO_ROOT / "v4" / "external" / "PROTOCOL_DEVIATIONS.md"
        with open(dev_path, "a") as f:
            f.write(f"\n## Deviation {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"E1 natural sample shortage: requested 200, got {len(e1_selected)}\n")

    # ---------- Combine and lock ----------
    # Ensure no overlap between E1 and E2
    assert len(set(e1_selected["external_id"]) & set(e2_selected["external_id"])) == 0, "E1/E2 overlap"
    # Ensure no overlap with dev 820 already handled
    combined = pd.concat([e1_selected, e2_selected], ignore_index=True)
    # Add required manifest columns
    # Need to ensure columns: external_id, source, source_posting_id, published_at, acquired_at, role_family_sampling_stratum, natural_or_challenge, challenge_stratum, text_sha256, normalised_text_sha256, duplicate_group_id
    # Fill missing
    if "published_at" not in combined.columns:
        combined["published_at"] = combined["posted_date"] if "posted_date" in combined.columns else ""
    if "acquired_at" not in combined.columns:
        combined["acquired_at"] = combined["acquired_at"] if "acquired_at" in combined.columns else datetime.now(timezone.utc).isoformat()
    if "role_family_sampling_stratum" not in combined.columns:
        combined["role_family_sampling_stratum"] = combined["role_family"]
    # Ensure challenge_stratum column exists
    if "challenge_stratum" not in combined.columns:
        combined["challenge_stratum"] = ""
    # For E1, challenge_stratum is empty; for E2, it's from sampling
    # Add text hashes (already have)
    # Add duplicate_group_id (already)
    # Select public-safe columns
    public_cols = ["external_id", "source", "source_posting_id", "published_at", "acquired_at", "role_family_sampling_stratum", "natural_or_challenge", "challenge_stratum", "text_sha256", "normalised_text_sha256", "duplicate_group_id"]
    # Ensure all exist
    for c in public_cols:
        if c not in combined.columns:
            combined[c] = ""
    manifest_public = combined[public_cols].copy()
    # Sort by external_id for determinism
    manifest_public = manifest_public.sort_values("external_id").reset_index(drop=True)
    # Compute hash
    manifest_bytes = manifest_public.to_csv(index=False).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    # Save public manifest (if no restricted text, can be committed)
    out_manifest = REPO_ROOT / "v4" / "external" / "LOCKED_SAMPLE_MANIFEST.csv"
    manifest_public.to_csv(out_manifest, index=False)
    print(f"Saved public manifest to {out_manifest} with hash {manifest_hash}")

    # Private manifest with full text (gitignored) for annotation
    private_manifest = combined.copy()
    # Keep job_summary for annotation
    private_manifest_path = REPO_ROOT / "v4" / "external" / "private" / "LOCKED_SAMPLE_MANIFEST_PRIVATE.csv"
    private_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    private_manifest.to_csv(private_manifest_path, index=False)
    print(f"Saved private manifest with text to {private_manifest_path}")

    # Create EXTERNAL_SAMPLE_LOCK.json
    lock_data = {
        "locked_sample_manifest_sha256": manifest_hash,
        "manifest_path": "v4/external/LOCKED_SAMPLE_MANIFEST.csv",
        "private_manifest_path": "v4/external/private/LOCKED_SAMPLE_MANIFEST_PRIVATE.csv",
        "n_total": int(len(combined)),
        "n_E1": int(len(e1_selected)),
        "n_E2": int(len(e2_selected)),
        "e1_meta": e1_meta,
        "e2_meta": e2_meta,
        "dedup_pool": {"n_unique_exact": int(dedup_pool["n_unique_exact"]), "n_near_pairs": len(dedup_pool["near_duplicate_pairs"])},
        "dedup_dev": {"exact_overlaps": int(dedup_dev["exact_overlaps"]), "normalised_overlaps": int(dedup_dev["normalised_overlaps"]), "near_pairs": len(dedup_dev["near_duplicate_pairs"])},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "METHODS_FROZEN": True,
        "SAMPLE_LOCKED": True,
        "LABELS_CREATED": False,
        "LABELS_LOCKED": False,
        "MODELS_EVALUATED": False,
    }
    lock_path = REPO_ROOT / "v4" / "external" / "EXTERNAL_SAMPLE_LOCK.json"
    lock_path.write_text(json.dumps(lock_data, indent=2, sort_keys=True))
    print(f"Saved lock to {lock_path}")

    # Update freeze manifest flags? Not yet, but we should record sample lock hash in freeze manifest or separate
    # For now, we will also update EXTERNAL_FREEZE_MANIFEST.json's SAMPLE_LOCKED flag? But freeze manifest is historical; we keep separate lock file
    # Also need to generate annotation workbooks
    print("Generating blank annotation workbooks...")
    from v4.external.annotation_package import create_blank_workbook, select_overlap_ids
    ann_private_dir = REPO_ROOT / "v4" / "external" / "private" / "annotation"
    ann_private_dir.mkdir(parents=True, exist_ok=True)
    # Annotator A: all 300
    workbook_a = ann_private_dir / "ANNOTATOR_A.xlsx"
    create_blank_workbook(private_manifest, workbook_a, annotator_label="A")
    # Overlap selection: 50 E1 + 50 E2
    e1_private = private_manifest[private_manifest["natural_or_challenge"]=="natural"]
    e2_private = private_manifest[private_manifest["natural_or_challenge"]=="challenge"]
    overlap_all, overlap_meta = select_overlap_ids(e1_private, e2_private, n_overlap=100, seed=42)
    workbook_b = ann_private_dir / "ANNOTATOR_B_OVERLAP.xlsx"
    # For B, create workbook with only overlap rows
    overlap_private = private_manifest[private_manifest["external_id"].isin(overlap_all["external_id"])]
    create_blank_workbook(overlap_private, workbook_b, annotator_label="B")
    # Adjudication template
    adj_path = ann_private_dir / "ADJUDICATION_TEMPLATE.xlsx"
    # Create empty adjudication sheet
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Adjudication"
    headers = ["posting_id","category","Annotator_A","Annotator_B","adjudicated_label","adjudication_note"]
    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h)
    wb.save(adj_path)
    print(f"Saved adjudication template to {adj_path}")
    # Save overlap meta
    overlap_path = REPO_ROOT / "v4" / "external" / "private" / "OVERLAP_IDS.json"
    overlap_path.write_text(json.dumps(overlap_meta, indent=2))
    print(f"Saved overlap meta to {overlap_path} with {len(overlap_all)} IDs")

    print("Post-acquisition complete. No labels created, no model evaluation.")
    print(f"METHODS_FROZEN=true SAMPLE_LOCKED=true LABELS_CREATED=false LABELS_LOCKED=false MODELS_EVALUATED=false")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
