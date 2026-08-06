"""
Method 1 of 4: TF-IDF baseline.

Adapts the mapping approach of Attwood & Williams (2023), who represented both job
listings and CyBOK knowledge areas as TF-IDF vectors and assigned listings to areas
by cosine similarity. Here the target label space is the 13-category skill taxonomy
rather than CyBOK, and each category is represented as a pseudo-document built from
its Tier 2 lexicon.

Two variants are produced:

  A. cosine       - TF-IDF cosine similarity between posting and category
                    pseudo-document, thresholded. This is the Attwood & Williams
                    method transferred to this taxonomy.
  B. weighted-hit - sum of TF-IDF weights of the category's lexicon terms present in
                    the posting, normalised. A stricter lexical variant that also
                    applies the negative-pattern suppression documented in annotation
                    guidelines v2.0 (homonyms such as "excellent" for Excel).

Thresholds are tuned per category on the dev split only and applied unchanged to the
test split, so reported test scores are not fitted to the evaluation data.

Usage:  python tfidf_baseline.py --corpus <csv> --gold <xlsx> --outdir <dir>
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CATEGORIES, LEXICONS, NEGATIVE_PATTERNS
from evaluate import load_gold, make_split, tune_thresholds, evaluate, format_report


def build_category_documents():
    """One pseudo-document per category, from its lexicon terms."""
    return [" ".join(LEXICONS[c]) for c in CATEGORIES]


def mask_negative_patterns(text, category):
    """Blank out known false-positive spans before lexical matching."""
    for pat in NEGATIVE_PATTERNS.get(category, []):
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


def cosine_scores(corpus_texts, target_texts):
    """Variant A: fit TF-IDF on the corpus, score postings against category docs."""
    vec = TfidfVectorizer(
        sublinear_tf=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_df=0.9, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#/.\-]+\b",
    )
    vec.fit(corpus_texts)
    X = vec.transform(target_texts)
    C = vec.transform(build_category_documents())
    S = cosine_similarity(X, C)
    # scale each category's scores to [0,1] so one threshold grid fits all
    denom = S.max(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return S / denom, vec


def weighted_hit_scores(corpus_texts, target_texts, vec):
    """Variant B: TF-IDF-weighted lexicon hits, with negative-pattern suppression."""
    vocab = vec.vocabulary_
    idf = vec.idf_
    S = np.zeros((len(target_texts), len(CATEGORIES)))
    for j, cat in enumerate(CATEGORIES):
        terms = [t.lower() for t in LEXICONS[cat]]
        weights = np.array([idf[vocab[t]] if t in vocab else 1.0 for t in terms])
        weights = weights / weights.sum()
        patterns = [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), w)
                    for t, w in zip(terms, weights)]
        for i, text in enumerate(target_texts):
            cleaned = mask_negative_patterns(text, cat)
            S[i, j] = sum(w for pat, w in patterns if pat.search(cleaned))
    denom = S.max(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return S / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(args.corpus)
    gold_df, y = load_gold(args.gold)
    gold_texts = (
        corpus.set_index("posting_id")
        .loc[gold_df["posting_id"], "job_summary"]
        .fillna("")
        .tolist()
    )
    corpus_texts = corpus["job_summary"].fillna("").tolist()

    dev, test = make_split(gold_df)
    print(f"corpus {len(corpus)} postings | gold {len(gold_df)} "
          f"(dev {dev.sum()}, test {test.sum()})")

    t0 = time.time()
    S_cos, vec = cosine_scores(corpus_texts, gold_texts)
    t_cos = time.time() - t0

    t0 = time.time()
    S_hit = weighted_hit_scores(corpus_texts, gold_texts, vec)
    t_hit = time.time() - t0

    results, timings, preds = {}, {"cosine_sec": t_cos, "weighted_hit_sec": t_hit}, {}
    for name, S in (("A_cosine", S_cos), ("B_weighted_hit", S_hit)):
        thr = tune_thresholds(S[dev], y[dev])
        pred = (S >= thr).astype(int)
        rep_test = evaluate(y[test], pred[test])
        rep_dev = evaluate(y[dev], pred[dev])
        print(format_report(rep_test, f"TF-IDF {name} — TEST split (n={test.sum()})"))
        rep_test.to_csv(outdir / f"tfidf_{name}_test.csv", index=False)
        rep_dev.to_csv(outdir / f"tfidf_{name}_dev.csv", index=False)
        results[name] = {
            "thresholds": dict(zip(CATEGORIES, thr.round(4).tolist())),
            "test_macro_f1": float(rep_test.loc[rep_test.category == "MACRO AVG", "f1"].iloc[0]),
            "test_micro_f1": float(rep_test.loc[rep_test.category == "MICRO AVG", "f1"].iloc[0]),
            "dev_macro_f1": float(rep_dev.loc[rep_dev.category == "MACRO AVG", "f1"].iloc[0]),
        }
        preds[name] = pred

    # predictions for the full corpus, using the better variant, for later comparison
    best = max(results, key=lambda k: results[k]["test_macro_f1"])
    S_full_cos, vec_full = cosine_scores(corpus_texts, corpus_texts)
    S_full = (S_full_cos if best == "A_cosine"
              else weighted_hit_scores(corpus_texts, corpus_texts, vec_full))
    thr_best = np.array([results[best]["thresholds"][c] for c in CATEGORIES])
    full_pred = pd.DataFrame((S_full >= thr_best).astype(int), columns=CATEGORIES)
    full_pred.insert(0, "posting_id", corpus["posting_id"].values)
    full_pred.to_csv(outdir / "tfidf_predictions_corpus.csv", index=False)

    gold_pred = pd.DataFrame(preds[best], columns=CATEGORIES)
    gold_pred.insert(0, "posting_id", gold_df["posting_id"].values)
    gold_pred.insert(1, "split", np.where(dev, "dev", "test"))
    gold_pred.to_csv(outdir / "tfidf_predictions_gold.csv", index=False)

    summary = {"method": "tfidf", "best_variant": best, "results": results,
               "timings": timings, "n_corpus": len(corpus), "n_gold": len(gold_df),
               "n_dev": int(dev.sum()), "n_test": int(test.sum())}
    (outdir / "tfidf_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest variant: {best} | "
          f"test macro-F1 {results[best]['test_macro_f1']:.3f} | "
          f"runtime {t_cos + t_hit:.1f}s")


if __name__ == "__main__":
    main()
