from __future__ import annotations

import random
from typing import Dict, Any


def _pct(v: float) -> str:
    return f"{int(round(100 * max(0.0, min(1.0, float(v)))))}%"


def _years_str(v: float) -> str:
    if v <= 0:
        return "< 1 year"
    if v == int(v):
        return f"{int(v)} years"
    return f"{v:.1f} years"


def build_reasoning(row: Dict[str, Any]) -> str:
    """
    Generate a 1-2 sentence, candidate-specific reasoning string.
    Pulls actual facts: name, title, company, location, years, specific signals.
    Avoids generic boilerplate across all candidates.
    """
    # Profile facts
    name = str(row.get("name", "")).strip()
    title = str(row.get("current_title", "")).strip()
    company = str(row.get("current_company", "")).strip()
    location = str(row.get("location_str", "")).strip()
    country = str(row.get("country", "")).strip()

    # Score signals
    years = float(row.get("years_total", 0.0))
    retrieval = float(row.get("retrieval_prod_score", 0.0))
    vector_db = float(row.get("vector_db_score", 0.0))
    python_ev = float(row.get("python_evidence_score", 0.0))
    ranking_eval = float(row.get("ranking_eval_score", 0.0))
    location_sc = float(row.get("location_match_score", 0.0))
    notice = float(row.get("notice_period_score", 0.0))
    behavior = float(row.get("behavioral_multiplier", 1.0))
    title_rel = float(row.get("title_relevance_score", 0.0))
    corroboration = float(row.get("career_corroboration_score", 0.0))
    is_services = float(row.get("services_only_flag", 0.0)) > 0.5
    is_research = float(row.get("research_only_flag", 0.0)) > 0.5

    # --- Build subject phrase ---
    if title and company:
        subject = f"{title} at {company}"
    elif title:
        subject = title
    elif company:
        subject = f"Engineer at {company}"
    else:
        subject = "Candidate"

    # --- Build location phrase ---
    if location_sc >= 0.95:
        loc_phrase = f"based in {location} (preferred Pune/Noida location)"
    elif location_sc >= 0.86:
        loc_phrase = f"based in {location} (accepted India location)"
    elif location_sc >= 0.72:
        loc_phrase = f"based in India"
    elif country and country.lower() != "india":
        loc_phrase = f"based in {country} (international, possible relocation risk)"
    else:
        loc_phrase = f"location: {location or 'unspecified'}"

    # --- Build skill evidence phrase ---
    strengths = []
    if retrieval >= 0.5:
        strengths.append("production embedding/retrieval experience in career history")
    if vector_db >= 0.5:
        strengths.append("hands-on vector database or hybrid search work")
    if ranking_eval >= 0.5:
        strengths.append("ranking evaluation exposure (NDCG/MRR/MAP)")
    if python_ev >= 0.6:
        strengths.append("strong Python evidence across career roles")
    if corroboration >= 0.7:
        strengths.append("skills well-corroborated by career descriptions")

    if not strengths:
        if retrieval > 0 or vector_db > 0:
            strengths.append("some retrieval/search exposure in career descriptions")
        else:
            strengths.append("limited direct AI/ML retrieval signals in career history")

    # --- Build availability phrase ---
    avail_parts = []
    if notice >= 1.0:
        avail_parts.append("notice period <=30 days (JD preferred)")
    elif notice >= 0.75:
        avail_parts.append("notice period 31-60 days (acceptable)")
    else:
        avail_parts.append("notice period >60 days (availability concern)")

    if behavior >= 1.0:
        avail_parts.append("high recruiter engagement signals")
    elif behavior >= 0.85:
        avail_parts.append("moderate recruiter engagement")
    else:
        avail_parts.append("lower engagement/availability signals")

    # --- Build penalty phrase if applicable ---
    penalty_phrase = ""
    if is_services:
        penalty_phrase = " Career concentrated at IT services firms (no product-company exposure) reduces fit score."
    elif is_research:
        penalty_phrase = " Academic/research-only background without production deployment evidence reduces fit score."

    # --- Assemble final string ---
    # Sentence 1: who they are + years + main strength
    years_str = _years_str(years)
    strength_text = strengths[0]
    if len(strengths) > 1:
        strength_text += f" and {strengths[1]}"

    sentence1 = f"{subject} with {years_str} total experience; {strength_text}; {loc_phrase}."
    sentence2 = f"Availability: {'; '.join(avail_parts)}.{penalty_phrase}"

    return (sentence1 + " " + sentence2).strip()
