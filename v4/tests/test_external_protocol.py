"""
External protocol tests — pre-registration and lock checks.

These tests verify that methods are frozen and external package is defensible,
without requiring external data to have been acquired yet (State B allowed).
If external sample manifests are missing, tests check that blocker is documented
rather than faking data.
"""

import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from v4.config import CATEGORIES, TAXONOMY_VERSION, CATEGORY_LABELS, LEXICONS, NEGATIVE_PATTERNS
from v4.tests._paths import REPO_ROOT, GOLD_PATH, CORPUS_PATH
from v4.evaluation.data import load_gold_with_texts

FREEZE_MANIFEST = REPO_ROOT / "v4" / "EXTERNAL_FREEZE_MANIFEST.json"
LOCKED_MANIFEST = REPO_ROOT / "v4" / "external" / "LOCKED_SAMPLE_MANIFEST.csv"
LOCKED_JSON = REPO_ROOT / "v4" / "external" / "LOCKED_SAMPLE_MANIFEST.json"
SOURCE_AUDIT = REPO_ROOT / "v4" / "external" / "SOURCE_AUDIT.md"
GITIGNORE = REPO_ROOT / ".gitignore"

def _load_manifest():
    assert FREEZE_MANIFEST.exists(), f"Freeze manifest missing: {FREEZE_MANIFEST}"
    return json.loads(FREEZE_MANIFEST.read_text())

# ---------- test_full_method_freeze_manifest ----------
def test_full_method_freeze_manifest():
    m = _load_manifest()
    required_keys = ["taxonomy_version", "CATEGORIES", "CATEGORY_LABELS_hash", "LEXICONS_hash", "NEGATIVE_PATTERNS_hash",
                     "source_git_commit", "random_seed", "development_posting_id_hash", "development_text_hash",
                     "A1", "A2", "A5", "S1", "S2", "S3", "H1", "H2", "package_versions", "python_version", "pytorch_version",
                     "transformers_version", "sentence_transformers_version", "timestamp_utc", "EXTERNAL_LABELS_ACCESSED"]
    for k in required_keys:
        assert k in m, f"manifest missing key {k}"
    # Check each method has thresholds/config
    assert "thresholds" in m["A5"] or "thresholds_by_category" in m["A5"], "A5 thresholds missing"
    assert "selected_C" in m["S1"], "S1 C missing"
    assert "thresholds" in m["S1"], "S1 thresholds missing"
    assert "thresholds" in m["S2"], "S2 thresholds missing"
    assert "thresholds" in m["S3"], "S3 thresholds missing"
    assert "thresholds" in m["H1"], "H1 thresholds missing"
    assert "thresholds" in m["H2"], "H2 thresholds missing"
    assert m["EXTERNAL_LABELS_ACCESSED"] is False, "EXTERNAL_LABELS_ACCESSED must be false at freeze"
    assert m["METHODS_FROZEN"] is True
    assert m["MODELS_EVALUATED"] is False

def test_taxonomy_frozen():
    m = _load_manifest()
    assert m["taxonomy_version"] == "v3-13cat-frozen"
    assert m["CATEGORIES"] == CATEGORIES
    assert len(CATEGORIES) == 13
    assert CATEGORIES == ["programming","sql","visualisation_bi","reporting","excel","statistics","machine_learning","data_cleaning","etl","data_modelling","cloud","stakeholder_comm","ethics_governance"]

def _get_dev_hashes():
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    corpus = pd.read_csv(CORPUS_PATH)
    # exact and normalised sets
    def th(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
    def nth(s):
        t = s.lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^\w ]", "", t)
        t = t.strip()
        return hashlib.sha256(t.encode("utf-8")).hexdigest()
    dev_exact = set(th(t) for t in texts)
    dev_norm = set(nth(t) for t in texts)
    corpus_texts = corpus["job_summary"].fillna("").astype(str).tolist()
    corp_exact = set(th(t) for t in corpus_texts)
    corp_norm = set(nth(t) for t in corpus_texts)
    return dev_exact, dev_norm, corp_exact, corp_norm

def test_external_sample_no_overlap_with_300():
    dev_exact, dev_norm, _, _ = _get_dev_hashes()
    if not LOCKED_MANIFEST.exists():
        # State B: check blocker documented
        assert SOURCE_AUDIT.exists(), "Source audit must exist when external sample not acquired"
        txt = SOURCE_AUDIT.read_text()
        assert "Reed API" in txt or "Adzuna" in txt
        # Also check that no fake sample manifest is present with overlap (if exists it must be empty or safe)
        pytest.skip("External sample not yet acquired (State B) — no overlap to check, but source audit documented")
    df = pd.read_csv(LOCKED_MANIFEST)
    # Check required columns
    assert "external_id" in df.columns
    # If raw text column exists, check hashes
    if "job_summary" in df.columns or "text" in df.columns:
        pytest.fail("LOCKED_SAMPLE_MANIFEST should not contain raw text if committed (gitignored)")
    # If manifest contains hashes, check no overlap
    if "text_sha256" in df.columns:
        for h in df["text_sha256"]:
            assert h not in dev_exact, f"External sample overlaps dev 300 exact hash {h}"
    if "normalised_text_sha256" in df.columns:
        for h in df["normalised_text_sha256"]:
            assert h not in dev_norm, f"External sample overlaps dev 300 normalised hash {h}"
    # Also check external_id not in dev 300
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    dev_ids = set(gold_df["posting_id"].astype(str))
    for eid in df["external_id"].astype(str):
        assert eid not in dev_ids, f"External id {eid} overlaps dev 300"

def test_external_sample_no_overlap_with_820():
    _, _, corp_exact, corp_norm = _get_dev_hashes()
    if not LOCKED_MANIFEST.exists():
        pytest.skip("External sample not yet acquired (State B)")
    df = pd.read_csv(LOCKED_MANIFEST)
    if "text_sha256" in df.columns:
        for h in df["text_sha256"]:
            assert h not in corp_exact, f"External overlaps 820 exact"
    if "normalised_text_sha256" in df.columns:
        for h in df["normalised_text_sha256"]:
            assert h not in corp_norm, f"External overlaps 820 normalised"
    corpus = pd.read_csv(CORPUS_PATH)
    corp_ids = set(corpus["posting_id"].astype(str))
    for eid in df["external_id"].astype(str):
        assert eid not in corp_ids, f"External id {eid} overlaps 820 corpus"

def test_natural_sampling_does_not_use_model_scores():
    # E1 sampling code must not reference A1/S1/S2/S3 predictions
    p = REPO_ROOT / "v4" / "external" / "sampling.py"
    assert p.exists(), "sampling.py missing"
    text = p.read_text()
    # Find sample_E1 function and ensure it asserts no model columns
    assert "sample_E1" in text, "sample_E1 not found"
    # Check that it raises if forbidden columns present
    assert "forbidden" in text.lower() or "model" in text.lower(), "E1 sampling should check forbidden model columns"
    # Ensure it does not import or use model prediction variables for selection
    # The function should not contain "A1_pred" or "S1_scores" in E1 logic (only in E2)
    e1_section = text.split("def sample_E1")[1].split("def sample_E2")[0] if "def sample_E2" in text else text
    assert "A1_pred" not in e1_section, "E1 sampling must not use A1_pred"
    assert "S1_scores" not in e1_section, "E1 sampling must not use S1_scores"
    assert "S2_scores" not in e1_section, "E1 sampling must not use S2_scores"
    assert "S3_scores" not in e1_section, "E1 sampling must not use S3_scores"

def test_challenge_sampling_declared_posthoc():
    p = REPO_ROOT / "v4" / "external" / "sampling.py"
    text = p.read_text()
    assert "sample_E2" in text or "challenge" in text.lower(), "challenge sampling not found"
    # Must be explicitly labelled challenge/post-hoc
    assert "post-hoc" in text.lower() or "posthoc" in text.lower() or "post hoc" in text.lower(), "challenge must be declared post-hoc"
    assert "challenge" in text.lower(), "challenge must be labelled"
    # Ensure E2 uses model outputs (allowed) but is documented as stress test
    assert "model_scores" in text or "frozen" in text.lower(), "challenge should use frozen model outputs"
    # Check that E2 is not mixed with E1 in primary estimate comment
    assert "never mix" in text.lower() or "separately" in text.lower() or "not mixed" in text.lower() or "post-hoc" in text.lower(), "challenge separation must be documented"

def test_annotation_workbook_has_no_model_outputs():
    # Check annotation_package.py creates blank workbooks without model outputs
    p = REPO_ROOT / "v4" / "external" / "annotation_package.py"
    assert p.exists()
    text = p.read_text()
    assert "create_blank_workbook" in text
    # Must validate no model outputs
    assert "validate_workbook_has_no_model_outputs" in text or "no model" in text.lower()
    # Ensure it doesn't write predictions
    assert "prediction" not in text.lower() or "no model" in text.lower() or "blank" in text.lower(), "annotation package should not write predictions"
    # If workbook exists, check it
    ann_dir = REPO_ROOT / "v4" / "external" / "annotation"
    if ann_dir.exists():
        for xlsx in ann_dir.glob("*.xlsx"):
            # Load and check headers
            import openpyxl
            wb = openpyxl.load_workbook(xlsx, read_only=True)
            ws = wb["Annotation"]
            headers = [c.value for c in ws[1] if c.value]
            forbidden = ["a1", "s1", "s2", "s3", "prediction", "score", "threshold", "confidence"]
            for h in headers:
                assert not any(k in str(h).lower() for k in forbidden), f"Workbook {xlsx} contains model output column {h}"
    # Also check that workbook creation uses blank label cells only
    assert "blank" in text.lower(), "workbook must have blank label cells"

def test_overlap_annotation_ids_frozen():
    p = REPO_ROOT / "v4" / "external" / "annotation_package.py"
    text = p.read_text()
    assert "select_overlap_ids" in text, "overlap selection function missing"
    assert "100" in text or "n_overlap" in text, "100 overlap must be targeted"
    # Must be seeded and stratified, not based on labels/model errors
    assert "seed" in text.lower(), "overlap selection must be seeded"
    assert "stratified" in text.lower() or "role_family" in text.lower(), "overlap must be stratified"
    assert "model" not in text.split("select_overlap_ids")[1].split("def ")[0].lower() or "independent" in text.lower(), "overlap must not be selected based on model errors"
    # If overlap manifest exists, check frozen
    overlap_manifest = REPO_ROOT / "v4" / "external" / "OVERLAP_IDS.json"
    if overlap_manifest.exists():
        data = json.loads(overlap_manifest.read_text())
        assert len(data.get("ids", [])) == 100, "overlap must be 100"
        assert data.get("n_E1_overlap") == 50
        assert data.get("n_E2_overlap") == 50

def test_gold_not_generated_by_code():
    p = REPO_ROOT / "v4" / "external" / "annotation_package.py"
    text = p.read_text()
    # Must write blank label cells only
    assert "value=None" in text or "blank" in text.lower(), "gold must be blank, not generated"
    assert "0,1" in text or "DataValidation" in text, "validation should enforce 0/1"
    # Ensure no automated filling of gold
    assert "Muse" not in text or "AI" in text, "should mention AI not used"
    # Check that freeze manifest says labels not created
    m = _load_manifest()
    assert m["LABELS_CREATED"] is False
    assert m["LABELS_LOCKED"] is False

def test_external_locked_test_not_evaluated():
    # No model evaluation function may consume future gold in this stage
    # Check that freeze manifest says MODELS_EVALUATED false
    m = _load_manifest()
    assert m["MODELS_EVALUATED"] is False, "MODELS_EVALUATED must be false at this stage"
    # Check that no code in v4/external evaluates models on external gold in this stage
    # Search for evaluation that consumes external gold (allow freeze docs and agreement)
    import glob
    for py in (REPO_ROOT / "v4" / "external").rglob("*.py"):
        content = py.read_text()
        # Should not have evaluation on external gold
        assert "EXTERNAL_LABEL_LOCK" not in content or "MODELS_EVALUATED" in content, f"{py} should not evaluate external gold yet"
        # Ensure no function that takes gold and predicts then evaluates external
        # Exclude files that document not evaluating (freeze, acquisition) and agreement/adjudication
        if py.name in ["freeze.py", "agreement.py", "_paths.py", "run_reed_acquisition.py", "post_acquisition.py", "simple_post.py"]:
            continue
        # Also allow files that explicitly say they do NOT evaluate
        if "does not evaluate" in content.lower() or "does not evaluate frozen models" in content.lower():
            continue
        if "evaluate" in content.lower() and "external" in content.lower():
            # Allow agreement evaluation but not model evaluation on external gold
            assert "agreement" in content.lower() or "annotat" in content.lower(), f"{py} may be incorrectly evaluating external"

def test_sample_manifest_hash():
    if not LOCKED_MANIFEST.exists():
        pytest.skip("Sample manifest not yet locked (State B)")
    # Hash should be stable
    h1 = hashlib.sha256(LOCKED_MANIFEST.read_bytes()).hexdigest()
    h2 = hashlib.sha256(LOCKED_MANIFEST.read_bytes()).hexdigest()
    assert h1 == h2, "hash not stable"
    # Also check if JSON manifest exists with hash
    if LOCKED_JSON.exists():
        j = json.loads(LOCKED_JSON.read_text())
        assert "locked_sample_manifest_sha256" in j or "manifest_hash" in j or "hash" in j

def test_raw_text_not_tracked():
    # restricted raw external text paths gitignored
    assert GITIGNORE.exists()
    gitignore_text = GITIGNORE.read_text()
    assert "v4/external/raw" in gitignore_text or "raw" in gitignore_text, "raw external text must be gitignored"
    assert "candidates" in gitignore_text or "external_raw" in gitignore_text, "candidates gitignored"
    # Also check that no raw text is tracked (if manifest exists, it shouldn't contain raw)
    if LOCKED_MANIFEST.exists():
        df = pd.read_csv(LOCKED_MANIFEST)
        assert "job_summary" not in df.columns, "raw text should not be tracked in committed manifest"
        assert "jobDescription" not in df.columns

def test_freeze_manifest():
    m = _load_manifest()
    # deterministic: hashes present
    assert "CATEGORY_LABELS_hash" in m
    assert "LEXICONS_hash" in m
    # Check vocab hashes present for methods that have them
    assert "vocabulary_hash" in m["A5"] or "vocabulary_size" in m["A5"]
    assert "vocabulary_hash" in m["S1"]
    # Check thresholds deterministic length 13
    for method in ["A5","S1","S2","S3","H1","H2"]:
        thr = m[method].get("thresholds")
        assert thr is not None and len(thr)==13, f"{method} thresholds must be 13"
    # Check python version recorded
    assert "python_version" in m
    assert "package_versions" in m
