"""
Freeze deployment configurations for external locked evaluation.

Derives ONE locked configuration per method using ONLY the 300 development postings.
Does NOT access external labels. The 300 are development data; external set remains unseen.

IMPORTANT: Performance of the final fitted deployment configuration on the same 300
is NOT an unbiased estimate. The unbiased-ish internal estimate remains nested CV.
The newly fitted all-300 configuration exists ONLY for external deployment.

Usage:
    PYTHONPATH=. python3 msc-uk-analyst-skills/v4/external/freeze.py
    or
    PYTHONPATH=. python3 -m v4.external.freeze

Outputs:
    v4/EXTERNAL_FREEZE_MANIFEST.json
"""

import hashlib
import json
import subprocess
import sys
import platform
from pathlib import Path
from datetime import datetime, timezone
import re

import numpy as np
import pandas as pd

# Portable repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
# When invoked as msc-uk-analyst-skills/v4/external/freeze.py, parents[2] is still repo root
# Fallback: if not found, try cwd
if not (REPO_ROOT / "v4" / "config.py").exists():
    REPO_ROOT = Path.cwd() / "msc-uk-analyst-skills"
    if not (REPO_ROOT / "v4" / "config.py").exists():
        REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

from v4.config import CATEGORIES, CATEGORY_LABELS, LEXICONS, NEGATIVE_PATTERNS, TAXONOMY_VERSION, RANDOM_SEED
from v4.evaluation.data import load_gold_with_texts
from v4.evaluation.splits import make_cv_splits
from v4.tests._paths import GOLD_PATH, CORPUS_PATH

from v4.semantic.model_config import (
    S1_VECTORISER_CONFIG, S1_C_GRID, S1_CLASS_WEIGHT,
    S2_MODEL_ID, S2_REVISION, S2_CHUNK_TOKENS,
    S3_MODEL_ID, S3_REVISION, S3_CHUNK_TOKENS, S3_AGGREGATION, NLI_HYPOTHESES_LIST
)
from v4.methods.lexical_baseline import tune_thresholds, VECTORISER_CONFIG, fit_tfidf_vectoriser, weighted_lexical_scores_with_vec
from v4.semantic.supervised_tfidf import build_vectoriser as build_s1_vectoriser, select_hyperparameters_inner_cv, get_outer_scores_and_thresholds
from v4.semantic.embedding_similarity import embedding_scores, get_category_embeddings
from v4.semantic.zero_shot_nli import nli_scores_for_texts
from v4.hybrid.selective_fallback import GRID_WITH_OFF, OFF_THRESHOLD, apply_hybrid_thresholds, tune_hybrid_thresholds, tune_hybrid_thresholds_inner_cv, lexical_A1_predictions
from v4.ablation.lexical_ablation import score_for_variant

MANIFEST_PATH = REPO_ROOT / "v4" / "EXTERNAL_FREEZE_MANIFEST.json"

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalised_text_hash(s: str) -> str:
    t = s.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w ]", "", t)
    t = t.strip()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def get_git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        return out
    except Exception:
        return "unknown"

def get_package_versions():
    versions = {}
    for pkg in ["scikit-learn", "pandas", "numpy", "torch", "transformers", "sentence-transformers", "openpyxl"]:
        try:
            if pkg == "scikit-learn":
                import sklearn
                versions[pkg] = sklearn.__version__
            elif pkg == "sentence-transformers":
                import sentence_transformers
                versions[pkg] = sentence_transformers.__version__
            elif pkg == "pandas":
                import pandas
                versions[pkg] = pandas.__version__
            elif pkg == "numpy":
                import numpy
                versions[pkg] = numpy.__version__
            elif pkg == "torch":
                import torch
                versions[pkg] = torch.__version__
            elif pkg == "transformers":
                import transformers
                versions[pkg] = transformers.__version__
            elif pkg == "openpyxl":
                import openpyxl
                versions[pkg] = openpyxl.__version__
            else:
                m = __import__(pkg.replace("-", "_"))
                versions[pkg] = getattr(m, "__version__", "unknown")
        except Exception as e:
            versions[pkg] = f"error:{e}"
    # torch etc already
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:
        pass
    return versions

def main():
    print("=== v4 External Freeze: deriving locked deployment configs from 300 development postings ===")
    gold_df, y, texts = load_gold_with_texts(str(GOLD_PATH), str(CORPUS_PATH))
    n = len(gold_df)
    assert n == 300, f"expected 300 gold, got {n}"
    print(f"Loaded {n} gold postings")

    # ---- Hashes ----
    cat_labels_hash = hash_json(CATEGORY_LABELS)
    lexicons_hash = hash_json(LEXICONS)
    neg_patterns_hash = hash_json(NEGATIVE_PATTERNS)
    # posting id hash sorted
    posting_ids_sorted = sorted(gold_df["posting_id"].astype(str).tolist())
    posting_ids_hash = sha256_hex(",".join(posting_ids_sorted))
    # dev text hash: join raw texts (order by posting_id sorted for determinism)
    # Use posting_id order as in gold_df (which is workbook order) but also compute sorted
    texts_by_id = dict(zip(gold_df["posting_id"].astype(str), texts))
    sorted_texts = [texts_by_id[pid] for pid in posting_ids_sorted]
    dev_text_hash = sha256_hex("".join(sorted_texts))
    # Also per-posting hashes for dedup
    dev_text_hashes = [text_hash(t) for t in texts]
    dev_norm_hashes = [normalised_text_hash(t) for t in texts]
    # corpus 820 hashes
    corpus_df = pd.read_csv(CORPUS_PATH)
    corpus_ids_hash = sha256_hex(",".join(sorted(corpus_df["posting_id"].astype(str).tolist())))
    corpus_text_hash = sha256_hex("".join(corpus_df["job_summary"].fillna("").astype(str).tolist()))

    # File hashes for definitions
    a1_file = REPO_ROOT / "v4" / "ablation" / "lexical_ablation.py"
    lex_baseline_file = REPO_ROOT / "v4" / "methods" / "lexical_baseline.py"
    a1_hash = file_sha256(a1_file) if a1_file.exists() else "missing"
    lex_baseline_hash = file_sha256(lex_baseline_file) if lex_baseline_file.exists() else "missing"
    # Config file hash
    config_hash = file_sha256(REPO_ROOT / "v4" / "config.py")

    # ---- A1 ----
    # No training, exact scorer/version/hash. A1 is whole-word any-hit, frozen lexicon, no negative suppression
    # Score matrix for reference
    S_A1 = score_for_variant("A1", texts)
    # A1 has no thresholds (any-hit >0)
    a1_config = {
        "method": "A1 lexical",
        "description": "Whole-word / phrase-safe matching, final frozen lexicon, no negative suppression, fixed any-hit (>0)",
        "scorer": "v4.ablation.lexical_ablation.score_for_variant('A1') -> any_hit",
        "lexicons_hash": lexicons_hash,
        "config_hash": config_hash,
        " scorer_file_hash": a1_hash,
        "thresholds": "fixed any-hit (>0), no tuning",
        "note": "No training. Deterministic."
    }

    # ---- A2 ----
    S_A2 = score_for_variant("A2", texts)
    a2_config = {
        "method": "A2 lexical",
        "description": "Whole-word matching + frozen NEGATIVE_PATTERNS suppression, fixed any-hit",
        "scorer": "v4.ablation.lexical_ablation.score_for_variant('A2') -> any_hit",
        "lexicons_hash": lexicons_hash,
        "negative_patterns_hash": neg_patterns_hash,
        "config_hash": config_hash,
        "scorer_file_hash": a1_hash,
        "thresholds": "fixed any-hit (>0)",
        "note": "Keep separately because external shift may change value of suppression"
    }

    # ---- A5 ----
    # Weighted lexical IDF: inductive TF-IDF fitted on train only. For deployment, we generate OOF then fit final vectoriser on all 300.
    print("Deriving A5 thresholds via OOF...")
    # OOF via 3-fold CV (same as nested)
    outer_splits, meta = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=RANDOM_SEED)
    n_cats = len(CATEGORIES)
    oof_A5 = np.zeros((n, n_cats), dtype=float)
    for train_idx, val_idx in outer_splits:
        vec = fit_tfidf_vectoriser([texts[i] for i in train_idx], config=VECTORISER_CONFIG)
        S_val = weighted_lexical_scores_with_vec(vec, [texts[i] for i in val_idx])
        oof_A5[val_idx] = S_val
    # Tune thresholds on OOF (genuine, not fitting on full data)
    grid_51 = np.linspace(0.0, 1.0, 51)
    thresholds_A5 = tune_thresholds(oof_A5, y, grid=grid_51)
    # Fit final vectoriser on all 300
    vec_A5_final = fit_tfidf_vectoriser(texts, config=VECTORISER_CONFIG)
    vocab_A5 = vec_A5_final.vocabulary_
    vocab_hash_A5 = hash_json(sorted(vocab_A5.keys()))
    a5_config = {
        "method": "A5 IDF lexical",
        "description": "A4 + inductive IDF weighting, continuous score + nested thresholds (historical weighted reference)",
        "vectoriser_config": VECTORISER_CONFIG,
        "thresholds": thresholds_A5.tolist(),
        "thresholds_by_category": {cat: float(thresholds_A5[i]) for i, cat in enumerate(CATEGORIES)},
        "vocabulary_size": len(vocab_A5),
        "vocabulary_hash": vocab_hash_A5,
        "oof_method": "3-fold CV OOF (Stratified on role_family, seed 42), thresholds tuned on OOF via tune_thresholds grid 0..1 51pts",
        "final_fit": "vectoriser fitted on all 300 development texts",
        "note": "Do NOT report re-tuned development performance as unbiased; internal nested estimate remains 0.9342"
    }

    # ---- S1 ----
    print("Deriving S1 final C and thresholds...")
    # Use 2-fold inner CV over all 300 to select best C
    from sklearn.model_selection import StratifiedKFold
    roles = gold_df["role_family"].values
    try:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
        inner_splits_all = list(skf.split(np.zeros(n), roles))
    except ValueError:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
        inner_splits_all = list(kf.split(np.zeros(n)))
    best_C, best_thr_S1, best_macro, details = select_hyperparameters_inner_cv(texts, y, inner_splits_all)
    # best_thr_S1 is from OOF tuning
    # Fit final vectoriser and classifiers on all 300 with best_C
    vec_S1_final = build_s1_vectoriser()
    X_all = vec_S1_final.fit_transform(texts)
    # Fit classifiers per category on all 300
    from sklearn.linear_model import LogisticRegression
    clfs_S1 = []
    coefs_shape = []
    for j in range(n_cats):
        clf = LogisticRegression(C=best_C, class_weight=S1_CLASS_WEIGHT, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_all, y[:, j])
        clfs_S1.append(clf)
        coefs_shape.append(list(clf.coef_.shape))
    vocab_S1 = vec_S1_final.vocabulary_
    vocab_hash_S1 = hash_json(sorted(vocab_S1.keys()))
    # Also record thresholds dict vs ndarray handling
    # best_thr_S1 may be dict or ndarray; unify to list
    if isinstance(best_thr_S1, dict):
        thr_list_S1 = [float(best_thr_S1[cat]) for cat in CATEGORIES]
        thr_dict_S1 = {cat: float(best_thr_S1[cat]) for cat in CATEGORIES}
    else:
        thr_list_S1 = [float(x) for x in best_thr_S1]
        thr_dict_S1 = {cat: float(thr_list_S1[i]) for i, cat in enumerate(CATEGORIES)}
    s1_config = {
        "method": "S1 supervised TF-IDF LR",
        "description": "Exact Experiment-2 definition: TF-IDF + LogisticRegression one-vs-rest",
        "vectoriser_config": S1_VECTORISER_CONFIG,
        "C_grid": S1_C_GRID,
        "class_weight": S1_CLASS_WEIGHT,
        "selected_C": best_C,
        "selection_rule": "inner 2-fold CV OOF on all 300, thresholds tuned on OOF, macro-F1, first-max wins",
        "inner_details": details,
        "thresholds": thr_list_S1,
        "thresholds_by_category": thr_dict_S1,
        "vocabulary_size": len(vocab_S1),
        "vocabulary_hash": vocab_hash_S1,
        "coefficient_shapes": coefs_shape,
        "fitting_population": "all 300 development postings",
        "fitting_population_hash": posting_ids_hash,
        "note": "Do NOT report performance of final fitted config on same 300 as unbiased; nested estimate remains 0.624"
    }

    # ---- S2 ----
    print("Deriving S2 thresholds (frozen embedding)...")
    # Embedding scores are frozen, compute for all 300
    # Use cache-aware: get_category_embeddings + embedding_scores
    cat_embs = get_category_embeddings()
    S_S2 = embedding_scores(texts, cat_embs=cat_embs)
    # Tune thresholds on all 300 (development only)
    thresholds_S2 = tune_thresholds(S_S2, y, grid=np.linspace(0.0, 1.0, 51))
    # Huggingface revision actual SHA
    try:
        from huggingface_hub import model_info
        # Not calling network; just record configured revision
        s2_revision_actual = S2_REVISION
    except Exception:
        s2_revision_actual = S2_REVISION
    s2_config = {
        "method": "S2 frozen embedding similarity",
        "description": "Exact Experiment-2 MiniLM definition, frozen, mean-pool chunking",
        "model_id": S2_MODEL_ID,
        "model_revision": S2_REVISION,
        "model_revision_actual": s2_revision_actual,
        "licence": "Apache-2.0",
        "params": "~22.7M",
        "max_tokens": 256,
        "chunk_tokens": S2_CHUNK_TOKENS,
        "chunk_overlap": 0,
        "pooling": "mean",
        "category_source": "CATEGORY_LABELS only",
        "category_labels_hash": cat_labels_hash,
        "thresholds": thresholds_S2.tolist(),
        "thresholds_by_category": {cat: float(thresholds_S2[i]) for i, cat in enumerate(CATEGORIES)},
        "note": "Embedding model frozen; thresholds development-only"
    }

    # ---- S3 ----
    print("Deriving S3 thresholds (frozen NLI)...")
    # Use cached NLI scores if available for speed, but ensure provenance matches
    from v4.evaluation.hybrid_nested import _get_nli_scores_cached, _provenance_hash
    # Try cache first
    try:
        S_S3, provenance_S3, was_cached = _get_nli_scores_cached(texts)
        print(f"  S3 NLI cache: was_cached={was_cached}")
    except Exception as e:
        print(f"  S3 cache failed ({e}), computing fresh...")
        S_S3 = nli_scores_for_texts(texts)
        provenance_S3 = {
            "model_id": S3_MODEL_ID,
            "chunk_tokens": S3_CHUNK_TOKENS,
            "n_texts": n,
            "texts_hash": _provenance_hash(texts, S3_MODEL_ID, NLI_HYPOTHESES_LIST, S3_CHUNK_TOKENS),
            "hypotheses": NLI_HYPOTHESES_LIST
        }
        was_cached = False
    thresholds_S3 = tune_thresholds(S_S3, y, grid=np.linspace(0.0, 1.0, 51))
    hyp_hash = hash_json(NLI_HYPOTHESES_LIST)
    s3_config = {
        "method": "S3 zero-shot NLI",
        "description": "Exact frozen DistilBERT NLI, 13 hypotheses, MAX aggregation",
        "model_id": S3_MODEL_ID,
        "model_revision": S3_REVISION,
        "hypotheses": NLI_HYPOTHESES_LIST,
        "hypotheses_hash": hyp_hash,
        "hypotheses_by_category": {cat: NLI_HYPOTHESES_LIST[i] for i, cat in enumerate(CATEGORIES)},
        "chunk_tokens": S3_CHUNK_TOKENS,
        "aggregation": S3_AGGREGATION,
        "thresholds": thresholds_S3.tolist(),
        "thresholds_by_category": {cat: float(thresholds_S3[i]) for i, cat in enumerate(CATEGORIES)},
        "nli_provenance": provenance_S3,
        "nli_was_cached": was_cached,
        "note": "Carry as secondary because highest lexically-unseen recall despite poor aggregate 0.387; high cost"
    }

    # ---- H1 / H2 ----
    print("Deriving H1/H2 hybrid thresholds...")
    # Lexical A1 predictions for all 300
    lex_A1_all = lexical_A1_predictions(texts)  # (300,13)
    # For H1 we need OOF semantic S1 scores (not final all-300 scores) to tune hybrid correctly
    # Generate OOF S1 via same inner splits used for S1 selection but with best_C
    # Use 3-fold outer OOF for S1 like semantic_nested does? For final hybrid we use OOF via 3-fold CV over all 300 with best_C
    # Simplify: generate OOF S1 via 3-fold CV with best_C
    oof_S1 = np.zeros((n, n_cats), dtype=float)
    outer_splits_s1, _ = make_cv_splits(gold_df, texts=texts, n_splits=3, seed=RANDOM_SEED)
    # Need to fit vectoriser per fold with best_C
    for train_idx, val_idx in outer_splits_s1:
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        vec = build_s1_vectoriser()
        X_train = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)
        for ci in range(n_cats):
            clf = LogisticRegression(C=best_C, class_weight=S1_CLASS_WEIGHT, solver="lbfgs", max_iter=1000, random_state=42)
            clf.fit(X_train, y[train_idx, ci])
            oof_S1[val_idx, ci] = clf.predict_proba(X_val)[:, 1]
    # Now tune hybrid thresholds for H1 using OOF
    # Use inner CV or direct tune? Use tune_hybrid_thresholds directly on OOF (since OOF already CV)
    # For strict nested, use tune_hybrid_thresholds_inner_cv with inner splits, but we have OOF already; simpler direct tune
    # We'll use direct tune_hybrid_thresholds on (lex_A1_all, oof_S1, y) with conservative OFF
    thresholds_H1 = tune_hybrid_thresholds(lex_A1_all, oof_S1, y, grid=GRID_WITH_OFF)
    # For H2, use NLI scores (frozen, no OOF needed) directly
    thresholds_H2 = tune_hybrid_thresholds(lex_A1_all, S_S3, y, grid=GRID_WITH_OFF)
    # Determine H2 status
    h2_all_off = bool(np.all(thresholds_H2 >= 1.5))
    h2_status = "equivalent_to_A1" if h2_all_off else "distinct"
    print(f"  H1 thresholds: {thresholds_H1}")
    print(f"  H2 thresholds: {thresholds_H2} -> {h2_status}")

    h1_config = {
        "method": "H1 A1+S1 fallback",
        "description": "A1 gate + S1 fallback, asymmetric OFF, thresholds tuned on FULL HYBRID F1, conservative OFF-first",
        "lexical_gate": "A1 any-hit",
        "semantic_model": "S1 (best_C from above)",
        "grid": "51 points 0..1 step 0.02 + OFF=2.0, descending tie-break",
        "thresholds": thresholds_H1.tolist(),
        "thresholds_by_category": {cat: float(thresholds_H1[i]) for i, cat in enumerate(CATEGORIES)},
        "oof_method": "3-fold CV OOF S1 with best_C, hybrid tuned on OOF",
        "status": "secondary POST-HOC hypothesis, not primary",
        "note": "Do not call primary model"
    }
    h2_config = {
        "method": "H2 A1+S3 fallback",
        "description": "A1 gate + S3 fallback, asymmetric OFF, tuned on hybrid F1",
        "lexical_gate": "A1 any-hit",
        "semantic_model": "S3 DistilBERT NLI",
        "grid": "51 points 0..1 step 0.02 + OFF=2.0, descending tie-break",
        "thresholds": thresholds_H2.tolist(),
        "thresholds_by_category": {cat: float(thresholds_H2[i]) for i, cat in enumerate(CATEGORIES)},
        "oof_method": "Direct hybrid tune on (lex_A1, S3 scores) development OOF (NLI frozen, no CV needed)",
        " H2_EXTERNAL_STATUS": h2_status,
        "distinct_from_A1": not h2_all_off,
        "note": "If OFF for all 13, do NOT waste external NLI computation pretending distinct"
    }

    # ---- Package versions and environment ----
    versions = get_package_versions()
    python_version = sys.version
    platform_str = platform.platform()
    # Get torch, transformers versions specifically
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except Exception:
        st_version = "unknown"

    timestamp = datetime.now(timezone.utc).isoformat()

    # ---- Manifest ----
    manifest = {
        "taxonomy_version": TAXONOMY_VERSION,
        "CATEGORIES": CATEGORIES,
        "CATEGORIES_order": CATEGORIES,
        "CATEGORY_LABELS": CATEGORY_LABELS,
        "CATEGORY_LABELS_hash": cat_labels_hash,
        "LEXICONS_hash": lexicons_hash,
        "NEGATIVE_PATTERNS_hash": neg_patterns_hash,
        "LEXICONS": LEXICONS,
        "NEGATIVE_PATTERNS": NEGATIVE_PATTERNS,
        "source_git_commit": get_git_commit(),
        "random_seed": RANDOM_SEED,
        "development_posting_id_hash": posting_ids_hash,
        "development_posting_ids": posting_ids_sorted,
        "development_text_hash": dev_text_hash,
        "development_text_hashes": dev_text_hashes,
        "development_normalised_hashes": dev_norm_hashes,
        "corpus_820_posting_id_hash": corpus_ids_hash,
        "corpus_820_text_hash": corpus_text_hash,
        "corpus_820_n": int(len(corpus_df)),
        "A1": a1_config,
        "A2": a2_config,
        "A5": a5_config,
        "S1": s1_config,
        "S2": s2_config,
        "S3": s3_config,
        "H1": h1_config,
        "H2": h2_config,
        "H2_EXTERNAL_STATUS": h2_status,
        "package_versions": versions,
        "python_version": python_version,
        "platform": platform_str,
        "pytorch_version": torch_version,
        "transformers_version": transformers_version,
        "sentence_transformers_version": st_version,
        "timestamp_utc": timestamp,
        "EXTERNAL_LABELS_ACCESSED": False,
        "METHODS_FROZEN": True,
        "SAMPLE_LOCKED": False,
        "LABELS_CREATED": False,
        "LABELS_LOCKED": False,
        "MODELS_EVALUATED": False,
        "freeze_script": "v4/external/freeze.py",
        "config_file_hash": config_hash,
        "manifest_version": "1.0"
    }

    # Sanitize for JSON (convert numpy types)
    def _sanitize(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(x) for x in o]
        return o
    manifest = _sanitize(manifest)
    # Write manifest deterministically
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote manifest to {MANIFEST_PATH}")
    print(f"  taxonomy: {TAXONOMY_VERSION}")
    print(f"  git commit: {manifest['source_git_commit']}")
    print(f"  A5 thresholds mean: {np.mean(thresholds_A5):.3f}")
    print(f"  S1 C: {best_C}, thresholds mean: {np.mean(thr_list_S1):.3f}")
    print(f"  S2 thresholds mean: {np.mean(thresholds_S2):.3f}")
    print(f"  S3 thresholds mean: {np.mean(thresholds_S3):.3f}")
    print(f"  H1 thresholds: {thresholds_H1.tolist()}")
    print(f"  H2 status: {h2_status}")
    print("Done. IMPORTANT: Do NOT report performance of final fitted configs on same 300 as unbiased.")

if __name__ == "__main__":
    main()
