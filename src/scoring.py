"""
scoring.py

Combines the structured, rule-based features (feature_extraction.py) with
a semantic-similarity signal (from precomputed embeddings) into a single
final score. Two-signal design on purpose:

  - Structured score: transparent, traceable to specific JD requirements.
    This is what does the heavy lifting against the dataset's traps
    (keyword stuffing, title mismatches, honeypots) because it reasons
    over *evidence*, not just vector distance.
  - Semantic score: catches genuine fits that don't use the JD's exact
    vocabulary (a "Tier 5 candidate [who] may not use the words 'RAG' or
    'Pinecone'... but built a recommendation system at a product company"),
    and gives us a continuous signal for candidates who are close variants
    of each other.

The structured score dominates the blend (65/35) deliberately -- this
dataset is specifically designed to punish systems that lean on semantic
similarity alone.
"""

from __future__ import annotations
import numpy as np


STRUCTURED_WEIGHT = 0.65
SEMANTIC_WEIGHT = 0.35

# Weighting within the structured score. Reflects the JD's own framing:
# must-have skills + shipped production systems + eval-framework experience
# are called out as non-negotiable; experience band, location, and
# trajectory are secondary shaping signals.
W_MUST_HAVE = 0.34
W_EXPERIENCE_FIT = 0.14
W_APPLIED_ML_RATIO = 0.16
W_LOCATION = 0.10
W_NICE_TO_HAVE = 0.08
W_TRAJECTORY = 0.08
W_PRODUCT_COMPANY_YEARS = 0.10  # rewards more raw years of relevant experience, not just ratio

DISQUALIFIER_PENALTY = 0.04       # multiplicative -- doesn't fully zero, so tie-breaks stay stable
KEYWORD_STUFFING_PENALTY = 0.35   # applied on top of the must-have score's own internal discount
HONEYPOT_PENALTY = 0.015          # near-zero, but nonzero so scores stay strictly ordered


def must_have_composite(must_have: dict) -> float:
    return (
        0.25 * must_have["embeddings_retrieval"]
        + 0.20 * must_have["vector_db_hybrid_search"]
        + 0.10 * must_have["python"]
        + 0.25 * must_have["eval_framework"]
        + 0.20 * must_have["production_ranking_system"]
    )


def structured_score(features: dict) -> float:
    mh = must_have_composite(features["must_have"])
    trajectory_score = 0.3 if features["trajectory"]["title_chaser"] else 1.0
    years_ml_component = min(1.0, features["years_applied_ml_product"] / 5.0)

    score = (
        W_MUST_HAVE * mh
        + W_EXPERIENCE_FIT * features["experience_fit"]
        + W_APPLIED_ML_RATIO * features["applied_ml_product_ratio"]
        + W_LOCATION * features["location_score"]
        + W_NICE_TO_HAVE * features["nice_to_have_score"]
        + W_TRAJECTORY * trajectory_score
        + W_PRODUCT_COMPANY_YEARS * years_ml_component
    )

    if features["keyword_stuffing_suspect"]:
        score *= KEYWORD_STUFFING_PENALTY

    return max(0.0, min(1.0, score))


def semantic_percentile(cos_sims: np.ndarray) -> np.ndarray:
    """
    Convert raw cosine similarities (which cluster in a narrow band for
    sentence-transformer models on similar-domain text) into a 0-1
    percentile rank across the whole candidate pool. More robust than
    using the raw cosine value directly.
    """
    order = np.argsort(cos_sims)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(cos_sims))
    return ranks / max(1, len(cos_sims) - 1)


def final_score(features: dict, semantic_pct: float) -> float:
    s = STRUCTURED_WEIGHT * structured_score(features) + SEMANTIC_WEIGHT * semantic_pct

    if features["disqualified"]:
        s *= DISQUALIFIER_PENALTY
    if features["is_honeypot_suspect"]:
        s *= HONEYPOT_PENALTY

    s *= features["behavioral_multiplier"]
    return max(0.0, min(1.0, s))
