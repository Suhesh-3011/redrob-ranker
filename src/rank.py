"""
rank.py

THE RANKING STEP. This is the single command reproduced at Stage 3 in a
sandboxed container with: <=5min wall-clock, <=16GB RAM, CPU only, NO
network access. This script honors that fully:

  - No LLM API calls of any kind.
  - No GPU usage, no network calls of any kind -- the "embedding model" is
    a locally-fitted TF-IDF + TruncatedSVD pipeline saved by precompute.py
    (scikit-learn, no external model hub dependency at all). Everything
    else is precomputed feature parsing + vectorized numpy math.

Usage:
    python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv

Requires that precompute.py has already been run against the same
candidates file to produce ./artifacts/{features.jsonl,embeddings.npy,candidate_ids.json}.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import jd_spec as jd
import scoring
import reasoning as rs

TOP_N = 100


def load_artifacts(artifacts_dir: Path):
    with open(artifacts_dir / "features.jsonl", "r", encoding="utf-8") as f:
        features = [json.loads(line) for line in f]
    embeddings = np.load(artifacts_dir / "embeddings.npy")
    with open(artifacts_dir / "candidate_ids.json", "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
    assert len(features) == embeddings.shape[0] == len(candidate_ids), "artifact row-count mismatch"
    return features, embeddings, candidate_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=False, default=None,
                     help="Path to candidates.jsonl (unused directly here, kept for spec-compatible CLI; "
                          "all candidate data is read from --artifacts, produced by precompute.py)")
    ap.add_argument("--artifacts", default="../artifacts")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--top-n", type=int, default=TOP_N,
                     help="How many ranked rows to output. Default 100 (the official "
                          "submission requirement). Use a smaller value when demoing "
                          "against a small sample (e.g. sample_candidates.json only has "
                          "50 records) -- the official validate_submission.py still "
                          "requires exactly 100 rows for the real submission.")
    args = ap.parse_args()
    top_n = args.top_n

    t_start = time.time()
    artifacts_dir = Path(args.artifacts)

    if top_n != TOP_N:
        print(f"NOTE: --top-n={top_n} (official submission requires exactly {TOP_N} rows; "
              f"only use a smaller value for small-sample sandbox demos).")

    print("Loading precomputed artifacts (no network, no GPU)...")
    features, embeddings, candidate_ids = load_artifacts(artifacts_dir)
    print(f"Loaded {len(features)} candidates.")

    print("Encoding JD reference text with the precomputed local TF-IDF+SVD pipeline "
          "(no network, no model download)...")
    import joblib
    vectorizer = joblib.load(artifacts_dir / "vectorizer.joblib")
    svd = joblib.load(artifacts_dir / "svd.joblib")
    jd_tfidf = vectorizer.transform([jd.IDEAL_CANDIDATE_TEXT])
    jd_embedding = svd.transform(jd_tfidf)[0].astype(np.float32)

    print("Computing semantic similarity for the full pool...")
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(jd_embedding)
    norms[norms == 0] = 1e-8
    cos_sims = (embeddings @ jd_embedding) / norms
    semantic_pct = scoring.semantic_percentile(cos_sims)

    print("Scoring all candidates (structured features + semantic blend)...")
    scored = []
    n_honeypot_flagged = 0
    n_disqualified = 0
    n_kw_stuffing = 0
    for i, f in enumerate(features):
        s = scoring.final_score(f, float(semantic_pct[i]))
        s_rounded = round(s, 4)
        scored.append((f["candidate_id"], s_rounded, f))
        if f["is_honeypot_suspect"]:
            n_honeypot_flagged += 1
        if f["disqualified"]:
            n_disqualified += 1
        if f["keyword_stuffing_suspect"]:
            n_kw_stuffing += 1

    print(f"  disqualified: {n_disqualified} | honeypot-suspect: {n_honeypot_flagged} | keyword-stuffing-suspect: {n_kw_stuffing}")

    # Sort on the SAME rounded value that gets written to the CSV, so
    # candidates that round to an identical score are true ties and get
    # broken by candidate_id ascending (validator requirement) -- sorting
    # on the unrounded float first can silently violate this once two
    # close-but-distinct floats round to the same displayed value.
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:top_n]

    max_score = top[0][1] if top else 1.0
    honeypots_in_top100 = sum(1 for _, _, f in top if f["is_honeypot_suspect"])
    print(f"Honeypot rate in top {top_n}: {honeypots_in_top100}/{top_n} ({honeypots_in_top100/top_n:.1%})")
    if honeypots_in_top100 / top_n > 0.10:
        print("  !! WARNING: honeypot rate exceeds the 10% Stage-3 disqualification threshold !!")

    rows = []
    n = len(top)
    for rank, (cid, score, f) in enumerate(top, start=1):
        # Tone is keyed to RANK POSITION within the final top-100, not
        # score-relative-to-max: this dataset's genuine top-100 candidates
        # cluster tightly in absolute score (deep, narrow-fit pool), so a
        # max-relative percentile made literally every row read as
        # "confident" tone -- which fails the Stage-4 "does reasoning tone
        # match rank" check even though every individual claim was true.
        # Rank-based tone keeps a rank-95 pick reading differently from a
        # rank-3 pick, which is what a reviewer sampling rows expects.
        rank_pct = 1.0 - (rank - 1) / max(1, n - 1)
        reasoning_text = rs.generate_reasoning(f, rank_pct)
        rows.append([cid, rank, score, reasoning_text])

    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        writer.writerows(rows)

    elapsed = time.time() - t_start
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Total ranking-step time: {elapsed:.1f}s")
    if elapsed > 300:
        print("  !! WARNING: exceeded the 5-minute compute budget !!")


if __name__ == "__main__":
    main()
