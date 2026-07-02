"""
reasoning.py

Generates the 1-2 sentence `reasoning` column entirely from facts already
present in `features` (feature_extraction.extract_features output). No LLM
call -- this is deliberate:

  1. The compute constraints ban network calls during ranking.
  2. Grounding every sentence in an actual extracted fact is the only way
     to reliably pass the Stage-4 "no hallucination" check -- an LLM
     paraphrasing a profile can still invent or misattribute a detail;
     a template reading directly from the feature dict cannot.

To avoid the "all-identical / templated with just the name swapped" penalty,
sentence structure is chosen based on which facts are actually present and
strongest for that candidate (not a fixed slot-fill), and tone is tied to
the candidate's own score tier -- so a rank-95 pick reads differently from
a rank-3 pick, not just with different nouns.
"""

from __future__ import annotations


def _fmt_pct(x) -> str:
    if x is None:
        return "unknown"
    return f"{x:.0%}"


def _strengths(f: dict) -> list[str]:
    out = []
    mh = f["must_have"]
    if mh["embeddings_corroborated"]:
        out.append(f"hands-on embeddings/retrieval work ({', '.join(mh['embeddings_corroborated'][:2])})")
    if mh["vector_db_corroborated"]:
        out.append(f"production vector-search experience ({', '.join(mh['vector_db_corroborated'][:2])})")
    if mh["eval_corroborated"]:
        out.append("real evaluation-framework experience (A/B testing / offline-online correlation)")
    if mh["production_system_phrases"]:
        out.append("has shipped a production ranking/recommendation/search system")
    if f["years_applied_ml_product"] >= 3:
        out.append(f"{f['years_applied_ml_product']:.1f} years in applied ML/AI at product companies")
    if f["nice_to_have_hits"]:
        out.append(f"bonus exposure to {', '.join(f['nice_to_have_hits'].keys())}")
    return out


def _concerns(f: dict) -> list[str]:
    out = []
    if f["disqualified"]:
        out.extend(f["disqualifier_reasons"])
    if f["keyword_stuffing_suspect"]:
        out.append(
            f"skills list claims {', '.join(f['claimed_only_terms'][:3])} but none of it is corroborated "
            f"in the actual career history -- likely keyword stuffing"
        )
    if f["is_honeypot_suspect"]:
        out.append(f"profile has internal inconsistencies ({'; '.join(f['honeypot_flags'][:2])})")
    if f["trajectory"]["title_chaser"]:
        out.append(f"trajectory shows {f['trajectory']['n_jobs']} jobs averaging {f['trajectory']['avg_tenure_months']:.0f} months each -- title-hopping risk")
    if f["location_score"] < 0.5:
        out.append(f"location concern: {f['location_note']}")
    if f["behavioral_notes"]:
        out.extend(f["behavioral_notes"])
    if f["must_have"]["embeddings_retrieval"] < 0.3 and f["must_have"]["vector_db_hybrid_search"] < 0.3:
        out.append("limited direct evidence of embeddings/vector-search production experience")
    years = f["years_of_experience"]
    if not (5.0 <= years <= 9.0):
        out.append(f"{years:.1f} years of experience is outside the JD's stated 5-9y band")
    return out


def _sentence_case(s: str) -> str:
    """Uppercase only the first character -- unlike str.capitalize(), this
    doesn't lowercase the rest of the string (which was turning 'India'
    into 'india' mid-sentence)."""
    return s[0].upper() + s[1:] if s else s


def generate_reasoning(f: dict, score_percentile: float) -> str:
    """
    score_percentile: this candidate's score as a fraction of the max score
    in the top-100 (1.0 = best pick). Used only to set tone, not content.
    """
    strengths = _strengths(f)
    concerns = _concerns(f)
    title = f["current_title"] or "candidate"
    company = f["current_company"] or "current company"
    years = f["years_of_experience"]

    lead = f"{title} at {company} with {years:.1f} years experience"

    if f["disqualified"] or f["is_honeypot_suspect"] or f["keyword_stuffing_suspect"]:
        # Should essentially never land in top 100 given the score gates,
        # but if it does, the reasoning must say so plainly.
        concern_text = concerns[0] if concerns else "profile does not match core JD requirements"
        return f"{lead}; {concern_text}."

    if score_percentile >= 0.75:
        # confident tone, lead with strength, one honest caveat if present
        if strengths:
            body = f"{lead}: {strengths[0]}"
            if len(strengths) > 1:
                body += f", and {strengths[1]}"
            if concerns:
                body += f". Only concern: {concerns[0]}."
            else:
                body += ". Strong match against the JD's core requirements."
            return body
        else:
            return f"{lead}, strong on semantic fit to the role though specific must-have evidence is thinner than top picks."

    elif score_percentile >= 0.4:
        # balanced tone
        if strengths and concerns:
            return f"{lead}: {strengths[0]}, but {concerns[0]}."
        elif strengths:
            return f"{lead}: {strengths[0]}. No major red flags, but not a standout match either."
        elif concerns:
            return f"{lead}. Main gap: {concerns[0]}."
        else:
            return f"{lead}; moderate overall fit against the JD, no single decisive strength or gap."

    else:
        # hedged tone -- these are filler-to-100, be upfront about it
        biggest_concern = concerns[0] if concerns else "fit against the JD's specific must-haves is weak"
        if strengths:
            return f"{lead}. {_sentence_case(biggest_concern)}, though {strengths[0]} keeps this above the cutoff for the final 100."
        return f"{lead}. {_sentence_case(biggest_concern)} -- included as lower-confidence filler for the top 100, not a strong recommend."
