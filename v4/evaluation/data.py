"""
Data loading for v4.

- load_gold(): reads annotation workbook, returns (gold_df, y, texts)
- load_corpus(): reads cleaned corpus csv

gold_df carries posting_id, role_family, and label columns.
texts alignment is enforced via posting_id join to corpus.
"""

from pathlib import Path

import pandas as pd
import openpyxl

from v4.config import CATEGORIES


def load_gold(workbook_path, corpus_path=None):
    """
    Load gold standard from annotation workbook.

    Returns (gold_df, y) where gold_df is the filtered annotation dataframe
    and y is ndarray shape (n, 13) in CATEGORIES order.
    """
    g = pd.read_excel(workbook_path, sheet_name="Annotation", engine="openpyxl")
    # Keep only rows where all 13 label columns are non-null (matches v3 behaviour)
    g = g[g[CATEGORIES].notna().all(axis=1)].reset_index(drop=True)
    y = g[CATEGORIES].astype(int).values
    return g, y


def load_gold_with_texts(workbook_path, corpus_path):
    """
    Load gold and join texts from corpus via posting_id.

    Returns (gold_df, y, texts) where texts is list[str] aligned to gold_df.
    Also loads the Posting_Texts sheet and checks consistency if corpus_path given.
    """
    gold_df, y = load_gold(workbook_path)
    corpus = pd.read_csv(corpus_path)
    # corpus must contain posting_id and job_summary
    if "posting_id" not in corpus.columns or "job_summary" not in corpus.columns:
        raise ValueError("corpus must contain posting_id and job_summary columns")
    corpus_map = dict(zip(corpus["posting_id"].astype(str), corpus["job_summary"].fillna("").astype(str)))
    # Ensure every gold posting_id exists in corpus
    missing = [pid for pid in gold_df["posting_id"].astype(str) if pid not in corpus_map]
    if missing:
        raise ValueError(f"{len(missing)} gold posting_ids not found in corpus: {missing[:5]}")
    texts = [corpus_map[str(pid)] for pid in gold_df["posting_id"].astype(str)]
    return gold_df, y, texts


def load_corpus(corpus_path):
    """Load corpus csv as DataFrame."""
    df = pd.read_csv(corpus_path)
    return df
