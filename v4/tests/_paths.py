"""Shared portable paths for v4 tests.

Derives repository root from this file's location so tests work
regardless of checkout name or directory.
"""

from pathlib import Path

# v4/tests/_paths.py -> v4/tests -> v4 -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

GOLD_PATH = REPO_ROOT / "v3" / "manual_work" / "gold_standard_annotation_workbook_v2.xlsx"
CORPUS_PATH = REPO_ROOT / "v3" / "manual_work" / "uk_analyst_corpus_v4_clean.csv"
