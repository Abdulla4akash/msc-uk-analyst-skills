"""
Semantic model configuration — frozen before results (see SEMANTIC_MODEL_SELECTION.md).
Do not change model IDs/thresholds after seeing results (model shopping forbidden).
"""

CATEGORIES = None  # imported from v4.config at runtime to avoid duplication

# ---- S1 ----
S1_VECTORISER_CONFIG = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
}

S1_C_GRID = [0.1, 1.0, 10.0]
S1_CLASS_WEIGHT = "balanced"  # single pre-registered policy; grid only over C
S1_THRESHOLD_GRID = None  # uses v4.methods.lexical_baseline.tune_thresholds grid 0..1 step 0.02 (51 points)

# ---- S2 ----
S2_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
S2_REVISION = "main"  # pinned to main at 2026-08-07; actual SHA logged at runtime (8b3219a...)
S2_LICENCE = "Apache-2.0"
S2_PARAMS = "~22.7M"
S2_MAX_TOKENS = 256
S2_POOLING = "mean"
S2_CHUNK_TOKENS = 256
S2_CHUNK_OVERLAP = 0

# ---- S3 ---- (revised 2026-08-07 before results due to BART-large infeasible runtime — see SEMANTIC_MODEL_SELECTION.md)
S3_MODEL_ID = "typeform/distilbert-base-uncased-mnli"
S3_REVISION = "main"  # pinned to main at 2026-08-07; actual SHA logged at runtime (0558d89...)
S3_LICENCE = "Apache-2.0"
S3_PARAMS = "~66M"
S3_MAX_TOKENS = 512
S3_CHUNK_TOKENS = 400  # per-premise chunk before hypothesis tokens (512 max, leave room for hyp)
S3_AGGREGATION = "max"

# Frozen hypotheses derived deterministically from CATEGORY_LABELS
# Template: f"This job requires {CATEGORY_LABELS[cat]}."
# Stored here in CATEGORIES order; do not edit after seeing results.
from v4.config import CATEGORY_LABELS, CATEGORIES as _CATS

NLI_HYPOTHESES = {cat: f"This job requires {CATEGORY_LABELS[cat]}." for cat in _CATS}
# Explicit list in order for scorer
NLI_HYPOTHESES_LIST = [NLI_HYPOTHESES[cat] for cat in _CATS]
