"""
feature_extraction.py

Turns one raw candidate record (matching candidate_schema.json) into a
structured, explainable feature dict. This is the "candidate understanding"
layer: career trajectory, evidenced skills (not just claimed), disqualifiers,
honeypot signals, location fit, and behavioral signals.

Design principle: every feature here should be traceable back to specific
fields in the candidate's profile, so the reasoning generator (reasoning.py)
never has to invent anything -- it just narrates these features. This is
what keeps Stage-4 "no hallucination" checks clean.

No ML calls happen here -- this is pure, fast, deterministic Python. Only
the free-text embedding (built separately) touches a model.
"""

from __future__ import annotations
import re
from datetime import date, datetime
from typing import Any

import jd_spec as jd

TODAY = date(2026, 7, 1)


# ---------------------------------------------------------------------------
# small utils
# ---------------------------------------------------------------------------
def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _text_contains_any(text: str, terms: list[str]) -> list[str]:
    text = text.lower()
    return [t for t in terms if t in text]


def _career_text_blob(candidate: dict) -> str:
    """All free text describing what the candidate has actually done."""
    parts = [candidate.get("profile", {}).get("summary", "")]
    for h in candidate.get("career_history", []):
        parts.append(f"{h.get('title','')} {h.get('description','')}")
    return " \n".join(parts).lower()


def _title_text(candidate: dict) -> str:
    return candidate.get("profile", {}).get("current_title", "").lower()


# ---------------------------------------------------------------------------
# Skill evidence: a skill only "counts" if the career_history PROSE
# corroborates it, not because the self-reported skill list says so.
#
# This is the direct fix for the JD's own instruction: "ignore keywords
# without context ... UNLESS you see concrete evidence of them being used
# in production in the Experience section." A skill's own duration_months
# field is itself a self-report, not corroboration -- a keyword-stuffed
# profile can claim "Embeddings, 18 months" while every job description is
# about brand design. We only trust a skill claim once the career_history
# text independently shows something in the same territory.
# ---------------------------------------------------------------------------
def _history_text_blob(candidate: dict) -> str:
    """Career-history titles + descriptions ONLY (the most trustworthy
    layer -- narrative claims about what was actually done job by job)."""
    parts = []
    for h in candidate.get("career_history", []):
        parts.append(f"{h.get('title','')} {h.get('description','')}")
    return " \n".join(parts).lower()


def _production_evidence_text(candidate: dict) -> str:
    """
    Summary + career_history text, MINUS any sentence that hedges itself as
    coursework/hobby/side-project (see jd.HEDGE_TERMS). This lets a genuine
    claim like "led the team that migrated to embedding-based retrieval"
    count as evidence, while filtering out "taking online courses on RAG
    and vector databases, experimenting with LangChain for side projects" --
    which is exactly the JD's own named disqualifier pattern, even though
    it name-drops the right keywords.
    """
    raw = _career_text_blob(candidate)
    # naive sentence split
    sentences = re.split(r"[.\n;]", raw)
    kept = [s for s in sentences if not any(h in s for h in jd.HEDGE_TERMS)]
    return " . ".join(kept)


def _corroborated_skills(candidate: dict, evidence_text: str, min_months: int = 3) -> dict[str, dict]:
    """
    A skill is 'corroborated' only if BOTH:
      (a) it's self-reported with meaningful duration, AND
      (b) that skill name (or a close variant) actually appears in the
          filtered production-evidence text -- not just the skills array,
          and not inside a hedge-word sentence about courses/hobby projects.
    Skills claimed but never mentioned in real production narrative are
    flagged as 'claimed_only' and get sharply discounted credit.
    """
    out = {}
    for s in candidate.get("skills", []):
        name = (s.get("name") or "").strip()
        if not name:
            continue
        months = s.get("duration_months", 0) or 0
        name_lower = name.lower()
        corroborated = months >= min_months and name_lower in evidence_text
        out[name_lower] = {
            "name": name,
            "proficiency": s.get("proficiency", ""),
            "endorsements": s.get("endorsements", 0),
            "duration_months": months,
            "corroborated": corroborated,
            "claimed_only": months >= min_months and not corroborated,
        }
    return out


def _skill_match_score(skill_evidence: dict, term_list: list[str]) -> tuple[float, list[str], list[str]]:
    """
    Returns (0-1 score, corroborated matches, claimed-only matches) for how
    well the candidate covers a set of JD terms. Corroborated matches earn
    full credit; claimed-only (skills-array-but-never-mentioned-in-a-job)
    matches earn a small fraction, since a self-report alone is weak but
    not zero evidence.
    """
    corroborated, claimed_only = [], []
    for skill_name, info in skill_evidence.items():
        if any(term in skill_name for term in term_list):
            if info["corroborated"]:
                corroborated.append(info["name"])
            elif info["claimed_only"]:
                claimed_only.append(info["name"])
    score = min(1.0, len(corroborated) / 2.0) + min(0.25, len(claimed_only) * 0.08)
    return (min(1.0, score), corroborated, claimed_only)


# ---------------------------------------------------------------------------
# Career trajectory
# ---------------------------------------------------------------------------
def _trajectory_features(candidate: dict) -> dict:
    hist = candidate.get("career_history", [])
    hist_sorted = sorted(hist, key=lambda h: h.get("start_date") or "")
    tenures = [h.get("duration_months", 0) or 0 for h in hist]
    avg_tenure_months = sum(tenures) / len(tenures) if tenures else 0
    n_jobs = len(hist)
    # title-chasing signal: many short (<18mo) stints with escalating titles
    short_stints = sum(1 for t in tenures if t < 18)
    title_chaser = n_jobs >= 3 and short_stints >= max(2, n_jobs - 1) and avg_tenure_months < 18

    total_career_months = sum(tenures)
    computed_years = round(total_career_months / 12, 1)

    return {
        "n_jobs": n_jobs,
        "avg_tenure_months": round(avg_tenure_months, 1),
        "title_chaser": title_chaser,
        "computed_years_from_history": computed_years,
    }


# ---------------------------------------------------------------------------
# Applied-ML-at-product-company ratio
# ---------------------------------------------------------------------------
def _applied_ml_product_ratio(candidate: dict) -> tuple[float, float]:
    hist = candidate.get("career_history", [])
    total_months = sum(h.get("duration_months", 0) or 0 for h in hist) or 1
    ml_product_months = 0
    for h in hist:
        text = f"{h.get('title','')} {h.get('description','')} {h.get('industry','')}".lower()
        company = (h.get("company") or "").lower()
        is_consulting = any(f in company for f in jd.CONSULTING_FIRMS) or "consulting" in (h.get("industry") or "").lower() or "it services" in (h.get("industry") or "").lower()
        is_ml_relevant = any(t in text for t in jd.NLP_IR_TERMS) or any(t in text for t in jd.MUST_HAVE_PRODUCTION_SYSTEM)
        if is_ml_relevant and not is_consulting:
            ml_product_months += h.get("duration_months", 0) or 0
    years_applied_ml = round(ml_product_months / 12, 1)
    ratio = min(1.0, ml_product_months / total_months)
    return years_applied_ml, ratio


# ---------------------------------------------------------------------------
# Disqualifier gate
# ---------------------------------------------------------------------------
def _disqualifiers(candidate: dict, career_text: str, title_text: str) -> dict:
    reasons = []

    if any(bad in title_text for bad in jd.BAD_CORE_TITLES):
        reasons.append(f"current title '{candidate['profile'].get('current_title')}' is an unrelated core function")

    hist = candidate.get("career_history", [])
    companies = [(h.get("company") or "").lower() for h in hist]
    if companies and all(any(f in c for f in jd.CONSULTING_FIRMS) for c in companies):
        reasons.append("entire career history is at consulting firms with no product-company experience")

    industries = " ".join((h.get("industry") or "") for h in hist).lower()
    if any(t in industries for t in jd.PURE_RESEARCH_INDUSTRY_TERMS) and not any(t in career_text for t in jd.PRODUCTION_EVIDENCE_TERMS):
        reasons.append("career history is pure research/academic with no production deployment evidence")

    cv_hits = _text_contains_any(career_text, jd.CV_ROBOTICS_SPEECH_TERMS)
    nlp_hits = _text_contains_any(career_text, jd.NLP_IR_TERMS)
    if cv_hits and not nlp_hits:
        reasons.append("expertise is computer vision/robotics/speech with no NLP/IR exposure")

    if any(t in title_text for t in jd.NON_CODING_SENIOR_TITLES):
        traj = _trajectory_features(candidate)
        # only a disqualifier if they've been in that kind of role a while
        current = next((h for h in hist if h.get("is_current")), None)
        if current and (current.get("duration_months", 0) or 0) >= 18:
            reasons.append("has been in a non-coding leadership/architecture role for 18+ months")

    return {"disqualified": len(reasons) > 0, "reasons": reasons}


# ---------------------------------------------------------------------------
# Honeypot detection -- "subtly impossible profiles"
# ---------------------------------------------------------------------------
def _honeypot_flags(candidate: dict) -> dict:
    flags = []
    profile = candidate.get("profile", {})
    hist = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    edu = candidate.get("education", [])

    # 1. stated years_of_experience vs sum of career_history durations
    stated_years = profile.get("years_of_experience", 0) or 0
    hist_years = sum(h.get("duration_months", 0) or 0 for h in hist) / 12
    if stated_years > 0 and abs(stated_years - hist_years) > 3.0:
        flags.append(f"stated experience ({stated_years}y) vs career-history total ({hist_years:.1f}y) mismatch")

    # 2. "expert" proficiency with near-zero duration
    for s in skills:
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) < 3:
            flags.append(f"claims 'expert' in {s.get('name')} with <3 months usage")

    # 3. implausibly many "expert" skills
    n_expert = sum(1 for s in skills if s.get("proficiency") == "expert")
    if n_expert >= 8:
        flags.append(f"{n_expert} skills marked 'expert' simultaneously")

    # 4. overlapping "is_current" jobs
    n_current = sum(1 for h in hist if h.get("is_current"))
    if n_current > 1:
        flags.append(f"{n_current} concurrent 'is_current' roles")

    # 5. overlapping date ranges between distinct jobs (excluding legitimate short gaps)
    parsed = []
    for h in hist:
        sd = _parse_date(h.get("start_date"))
        ed = _parse_date(h.get("end_date")) or TODAY
        if sd:
            parsed.append((sd, ed))
    parsed.sort()
    for i in range(len(parsed) - 1):
        if parsed[i][1] and parsed[i + 1][0] and parsed[i][1] > parsed[i + 1][0]:
            overlap_days = (parsed[i][1] - parsed[i + 1][0]).days
            if overlap_days > 60:
                flags.append("career_history has jobs overlapping by more than 2 months")
                break

    # 6. start_date/end_date inconsistent with duration_months
    for h in hist:
        sd, ed = _parse_date(h.get("start_date")), _parse_date(h.get("end_date"))
        if sd and ed:
            actual_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            claimed = h.get("duration_months", 0) or 0
            if abs(actual_months - claimed) > 6:
                flags.append(f"duration_months ({claimed}) inconsistent with start/end dates (~{actual_months}mo) at {h.get('company')}")

    # 7. education timeline impossible given experience (graduated after career supposedly started)
    if edu and hist:
        earliest_job = min((_parse_date(h.get("start_date")) for h in hist if _parse_date(h.get("start_date"))), default=None)
        latest_grad = max((e.get("end_year", 0) for e in edu), default=0)
        if earliest_job and latest_grad and earliest_job.year < latest_grad - 1:
            flags.append("career history begins before stated graduation year")

    # 8. skill_assessment_scores wildly contradicting self-reported proficiency
    assess = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    skill_by_name = {s.get("name"): s for s in skills}
    contradictions = 0
    for skill_name, score in assess.items():
        s = skill_by_name.get(skill_name)
        if s and s.get("proficiency") in ("expert", "advanced") and score < 25:
            contradictions += 1
    if contradictions >= 2:
        flags.append(f"{contradictions} skills self-rated expert/advanced but platform assessment score <25")

    return {"is_honeypot_suspect": len(flags) >= 2, "honeypot_flags": flags, "n_flags": len(flags)}


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------
def _location_score(candidate: dict) -> tuple[float, str]:
    profile = candidate.get("profile", {})
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    if country != "india":
        # JD: outside India is case-by-case, no visa sponsorship -- treat as
        # a strong penalty rather than a hard filter, since "case-by-case"
        # leaves room for exceptional fits.
        return (0.15, "outside India (no visa sponsorship)")

    if any(c in location for c in jd.PREFERRED_CITIES):
        return (1.0, "in a preferred location (Noida/Pune)")
    if any(c in location for c in jd.WELCOME_CITIES):
        return (0.8, "in an explicitly welcome India location")
    if relocate:
        return (0.6, "in India, elsewhere, but willing to relocate")
    return (0.4, "in India but not near preferred cities and not flagged as willing to relocate")


# ---------------------------------------------------------------------------
# Behavioral multiplier
# ---------------------------------------------------------------------------
def _behavioral_multiplier(signals: dict) -> tuple[float, list[str]]:
    notes = []
    mult = 1.0

    last_active = _parse_date(signals.get("last_active_date"))
    days_inactive = (TODAY - last_active).days if last_active else 9999
    if days_inactive > 180:
        mult *= 0.55
        notes.append(f"inactive for {days_inactive} days")
    elif days_inactive > 60:
        mult *= 0.85
        notes.append(f"last active {days_inactive} days ago")

    resp = signals.get("recruiter_response_rate", 0) or 0
    if resp < 0.15:
        mult *= 0.65
        notes.append(f"very low recruiter response rate ({resp:.0%})")
    elif resp < 0.35:
        mult *= 0.9

    if not signals.get("open_to_work_flag", False):
        mult *= 0.85
        notes.append("not flagged open to work")

    notice = signals.get("notice_period_days", 60) or 60
    if notice > 60:
        mult *= 0.85
        notes.append(f"long notice period ({notice} days)")
    elif notice <= jd.IDEAL_NOTICE_DAYS:
        mult *= 1.05

    interview_rate = signals.get("interview_completion_rate", 1.0)
    if interview_rate is not None and interview_rate < 0.4:
        mult *= 0.85
        notes.append(f"low interview completion rate ({interview_rate:.0%})")

    offer_accept = signals.get("offer_acceptance_rate", -1)
    if offer_accept is not None and 0 <= offer_accept < 0.2:
        mult *= 0.9
        notes.append(f"low historical offer acceptance ({offer_accept:.0%})")

    if not signals.get("verified_email", True):
        mult *= 0.97
    if not signals.get("verified_phone", True):
        mult *= 0.97

    return max(0.05, round(mult, 3)), notes


def _case_insensitive_dedup(terms: list[str]) -> list[str]:
    """Dedupe terms ignoring case, keeping first-seen casing (e.g. avoids
    'Elasticsearch, elasticsearch' showing up as two separate items when
    one came from the skills array and one from free-text matching)."""
    seen = set()
    out = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _keyword_stuffing_flag(candidate: dict, must_have: dict, title_text: str) -> dict:
    """
    Explicit detector for the JD's own named trap: an AI-keyword-heavy
    skills list sitting on top of a career history that has nothing to do
    with it. This is distinct from an "honeypot" (impossible profile) --
    it's a plausible profile that's just misleading about relevance.
    """
    claimed_terms = (
        must_have["embeddings_claimed_only"] + must_have["vector_db_claimed_only"] + must_have["eval_claimed_only"]
    )
    corroborated_terms = (
        must_have["embeddings_corroborated"] + must_have["vector_db_corroborated"] + must_have["eval_corroborated"]
    )
    is_core_tech_title = any(
        t in title_text
        for t in ["engineer", "scientist", "developer", "architect", "researcher", "ml", "ai ", "data"]
    )
    suspect = len(claimed_terms) >= 3 and len(corroborated_terms) == 0 and not is_core_tech_title
    return {
        "keyword_stuffing_suspect": suspect,
        "claimed_only_terms": claimed_terms,
    }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def extract_features(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career_text = _career_text_blob(candidate)          # summary + history (broadest, for embeddings)
    evidence_text = _production_evidence_text(candidate)  # hedge-filtered, for corroboration
    title_text = _title_text(candidate)
    skill_evidence = _corroborated_skills(candidate, evidence_text)

    emb_score, emb_corrob, emb_claimed = _skill_match_score(skill_evidence, jd.MUST_HAVE_EMBEDDINGS)
    vdb_score, vdb_corrob, vdb_claimed = _skill_match_score(skill_evidence, jd.MUST_HAVE_VECTOR_DB)
    py_score, py_corrob, py_claimed = _skill_match_score(skill_evidence, jd.MUST_HAVE_PYTHON)
    eval_score, eval_corrob, eval_claimed = _skill_match_score(skill_evidence, jd.MUST_HAVE_EVAL_FRAMEWORK)

    # evidence text itself directly describing must-haves in prose (candidates
    # don't always list a formal "skill" for what they built)
    emb_text_hits = _text_contains_any(evidence_text, jd.MUST_HAVE_EMBEDDINGS)
    vdb_text_hits = _text_contains_any(evidence_text, jd.MUST_HAVE_VECTOR_DB)
    eval_text_hits = _text_contains_any(evidence_text, jd.MUST_HAVE_EVAL_FRAMEWORK)
    prod_system_hits = _text_contains_any(evidence_text, jd.MUST_HAVE_PRODUCTION_SYSTEM)

    emb_score = max(emb_score, 0.7 if emb_text_hits else 0.0)
    vdb_score = max(vdb_score, 0.7 if vdb_text_hits else 0.0)
    eval_score = max(eval_score, 0.7 if eval_text_hits else 0.0)
    py_score = max(py_score, 0.5 if "python" in evidence_text else 0.0)
    prod_system_score = min(1.0, len(prod_system_hits) / 2.0)

    must_have = {
        "embeddings_retrieval": round(emb_score, 2),
        "vector_db_hybrid_search": round(vdb_score, 2),
        "python": round(py_score, 2),
        "eval_framework": round(eval_score, 2),
        "production_ranking_system": round(prod_system_score, 2),
        "embeddings_corroborated": _case_insensitive_dedup(emb_corrob + emb_text_hits),
        "vector_db_corroborated": _case_insensitive_dedup(vdb_corrob + vdb_text_hits),
        "eval_corroborated": _case_insensitive_dedup(eval_corrob + eval_text_hits),
        "embeddings_claimed_only": _case_insensitive_dedup([t for t in emb_claimed if t.lower() not in {x.lower() for x in emb_text_hits}]),
        "vector_db_claimed_only": _case_insensitive_dedup([t for t in vdb_claimed if t.lower() not in {x.lower() for x in vdb_text_hits}]),
        "eval_claimed_only": _case_insensitive_dedup([t for t in eval_claimed if t.lower() not in {x.lower() for x in eval_text_hits}]),
        "production_system_phrases": _case_insensitive_dedup(prod_system_hits),
    }

    kw_stuff = _keyword_stuffing_flag(candidate, must_have, title_text)
    if kw_stuff["keyword_stuffing_suspect"]:
        # heavy discount on the must-have scores that were driven only by
        # unverified skill claims
        must_have["embeddings_retrieval"] = min(must_have["embeddings_retrieval"], 0.15)
        must_have["vector_db_hybrid_search"] = min(must_have["vector_db_hybrid_search"], 0.15)
        must_have["eval_framework"] = min(must_have["eval_framework"], 0.15)

    nice_to_have_hits = {k: _text_contains_any(career_text, v) for k, v in jd.NICE_TO_HAVE.items()}
    nice_to_have_score = min(1.0, sum(1 for v in nice_to_have_hits.values() if v) / 3.0)

    traj = _trajectory_features(candidate)
    years_applied_ml, applied_ml_ratio = _applied_ml_product_ratio(candidate)
    disq = _disqualifiers(candidate, career_text, title_text)
    honeypot = _honeypot_flags(candidate)
    loc_score, loc_note = _location_score(candidate)
    behavior_mult, behavior_notes = _behavioral_multiplier(signals)

    years = profile.get("years_of_experience", 0) or 0
    if jd.IDEAL_EXPERIENCE_MIN <= years <= jd.IDEAL_EXPERIENCE_MAX:
        exp_fit = 1.0
    elif jd.JD_EXPERIENCE_MIN <= years <= jd.JD_EXPERIENCE_MAX:
        exp_fit = 0.75
    else:
        dist = min(abs(years - jd.JD_EXPERIENCE_MIN), abs(years - jd.JD_EXPERIENCE_MAX))
        exp_fit = max(0.15, 1 - dist / 8)

    return {
        "candidate_id": candidate["candidate_id"],
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "years_of_experience": years,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "must_have": must_have,
        "keyword_stuffing_suspect": kw_stuff["keyword_stuffing_suspect"],
        "claimed_only_terms": kw_stuff["claimed_only_terms"],
        "nice_to_have_score": round(nice_to_have_score, 2),
        "nice_to_have_hits": {k: v for k, v in nice_to_have_hits.items() if v},
        "trajectory": traj,
        "years_applied_ml_product": years_applied_ml,
        "applied_ml_product_ratio": round(applied_ml_ratio, 2),
        "experience_fit": round(exp_fit, 2),
        "disqualified": disq["disqualified"],
        "disqualifier_reasons": disq["reasons"],
        "is_honeypot_suspect": honeypot["is_honeypot_suspect"],
        "honeypot_flags": honeypot["honeypot_flags"],
        "location_score": loc_score,
        "location_note": loc_note,
        "behavioral_multiplier": behavior_mult,
        "behavioral_notes": behavior_notes,
        "notice_period_days": signals.get("notice_period_days"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "last_active_date": signals.get("last_active_date"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "embedding_text": _embedding_text(candidate),
    }


def _embedding_text(candidate: dict) -> str:
    """Text used for the semantic-similarity side of the score."""
    profile = candidate.get("profile", {})
    history_text = " ".join(
        f"{h.get('title','')} at {h.get('company','')} ({h.get('industry','')}): {h.get('description','')}"
        for h in candidate.get("career_history", [])
    )
    skills_text = ", ".join(
        s.get("name", "") for s in candidate.get("skills", []) if (s.get("duration_months", 0) or 0) >= 3
    )
    return f"{profile.get('headline','')}. {profile.get('summary','')} Experience: {history_text} Evidenced skills: {skills_text}"
