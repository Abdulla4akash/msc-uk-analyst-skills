"""Regression test: no committed v4 Python file contains machine-specific absolute paths."""

from pathlib import Path

import pytest

from v4.tests._paths import REPO_ROOT


def test_no_absolute_user_paths_in_v4_python():
    """Ensure no v4/*.py (except this test's own string literals) contains machine-specific paths."""
    v4_root = REPO_ROOT / "v4"
    offenders = []
    for py_path in v4_root.rglob("*.py"):
        text = py_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for line in lines:
            # skip this file's own assertion string that necessarily mentions "/Users/"
            if py_path.name == "test_portability.py" and '"/Users/"' in line:
                continue
            if "/Users/" in line:
                offenders.append(str(py_path.relative_to(REPO_ROOT)))
                break
            if py_path.name != "test_portability.py" and "akashx" in line:
                offenders.append(str(py_path.relative_to(REPO_ROOT)) + " (akashx)")
                break
    assert not offenders, f"Machine-specific absolute paths found in v4 Python files: {offenders}"


def test_portable_paths_resolve():
    """Portable paths helper actually points to existing files."""
    from v4.tests._paths import GOLD_PATH, CORPUS_PATH
    assert GOLD_PATH.exists(), f"GOLD_PATH does not exist: {GOLD_PATH}"
    assert CORPUS_PATH.exists(), f"CORPUS_PATH does not exist: {CORPUS_PATH}"
