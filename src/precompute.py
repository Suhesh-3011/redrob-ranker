"""
precompute.py

OFFLINE step. Extracts structured features for every candidate and fits a
purely local semantic embedding space (TF-IDF -> TruncatedSVD / LSA) over
the whole candidate pool's profile text.

Why TF-IDF+SVD instead of a downloaded transformer model: it needs zero
network access at any point (no huggingface.co dependency, no risk of the
Stage-3 sandboxed reproduction environment failing because it can't reach
an external model hub), it's deterministic, fast, and the semantic signal
is only 35% of the final score anyway (see scoring.py) -- the structured,
rule-based score does the heavy lifting against this dataset's traps. This
trades a bit of semantic nuance for materially higher reproducibility,
which the spec explicitly weights heavily (Stage 3: "if your submission
cannot be reproduced within these limits, it is disqualified regardless of
your composite score").

Usage:
    python precompute.py --candidates ../data/candidates.jsonl.gz --out ../artifacts

Produces in --out:
    features.jsonl        one structured-feature JSON object per candidate
    embeddings.npy          (N, n_components) float32 matrix, same row order as features.jsonl
    candidate_ids.json      list of candidate_ids, same order as the above two
    vectorizer.joblib       fitted TfidfVectorizer (needed by rank.py to embed the JD text consistently)
    svd.joblib               fitted TruncatedSVD
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import feature_extraction as fe


def load_candidates(path: str):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_json_array(path: str):
    """Fallback for the sample_candidates.json format (a pretty-printed JSON array)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for c in data:
        yield c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl / .jsonl.gz / sample_candidates.json")
    ap.add_argument("--out", default="../artifacts", help="Output directory for artifacts")
    ap.add_argument("--n-components", type=int, default=200, help="SVD (LSA) dimensionality")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import joblib

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = Path(args.candidates)
    if cand_path.suffix == ".json":
        loader = load_candidates_json_array(args.candidates)
    else:
        loader = load_candidates(args.candidates)

    print(f"Loading candidates from {args.candidates} ...")
    t0 = time.time()
    candidates = list(loader)
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")

    print("Extracting structured features (rule-based, no ML)...")
    t0 = time.time()
    features = []
    for c in candidates:
        try:
            features.append(fe.extract_features(c))
        except Exception as e:
            print(f"  WARNING: failed to extract features for {c.get('candidate_id', '???')}: {e}")
    print(f"Extracted {len(features)} feature records in {time.time()-t0:.1f}s")

    print("Fitting local TF-IDF + SVD (LSA) embedding space over the candidate pool "
          "(no network, no model download)...")
    t0 = time.time()
    texts = [f["embedding_text"] for f in features]

    # include the JD reference text in the fit corpus so the shared
    # vocabulary/IDF weighting reflects JD-relevant terms too, and so the
    # fitted space generalizes to the JD vector at rank time.
    import jd_spec as jd
    fit_texts = texts + [jd.IDEAL_CANDIDATE_TEXT]

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(fit_texts)

    n_components = min(args.n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf)

    embeddings = reduced[:-1].astype(np.float32)  # drop the JD row we appended for fitting
    print(f"Fitted + transformed {len(features)} candidates in {time.time()-t0:.1f}s "
          f"(dim={n_components}, explained_variance={svd.explained_variance_ratio_.sum():.3f})")

    candidate_ids = [f["candidate_id"] for f in features]

    # Drop the raw embedding_text from the saved features (not needed after
    # embeddings are computed; reasoning.py never reads it)
    for f in features:
        f.pop("embedding_text", None)

    print(f"Writing artifacts to {out_dir} ...")
    with open(out_dir / "features.jsonl", "w", encoding="utf-8") as fout:
        for f in features:
            fout.write(json.dumps(f) + "\n")

    np.save(out_dir / "embeddings.npy", embeddings)

    with open(out_dir / "candidate_ids.json", "w", encoding="utf-8") as fout:
        json.dump(candidate_ids, fout)

    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(svd, out_dir / "svd.joblib")

    print(f"Done. {len(candidate_ids)} candidates precomputed.")
    print(f"  {out_dir / 'features.jsonl'}")
    print(f"  {out_dir / 'embeddings.npy'}  shape={embeddings.shape}")
    print(f"  {out_dir / 'candidate_ids.json'}")
    print(f"  {out_dir / 'vectorizer.joblib'}, {out_dir / 'svd.joblib'}")


if __name__ == "__main__":
    main()
