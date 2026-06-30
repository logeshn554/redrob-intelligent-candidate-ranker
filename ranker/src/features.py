from __future__ import annotations

import gzip
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from dateutil import parser as dt_parser

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "niit technologies", "patni",
    "zensar", "mastech", "infotech", "syntel", "virtusa",
}

ACADEMIC_COMPANIES = {
    "university", "iit", "iim", "nit", "iiser", "research lab", "research center",
    "research institute", "lab", "postdoc", "academia", "college", "institute of",
    "national laboratory", "foundation", "iisc",
}

# Titles that clearly indicate irrelevance to AI/ML/Search engineering
IRRELEVANT_TITLE_TOKENS = {
    "hr", "human resources", "accountant", "accounting",
    "civil engineer", "civil", "mechanical engineer", "mechanical",
    "graphic designer", "graphic", "content writer",
    "sales executive", "sales", "marketing manager", "marketing",
    "customer support", "customer service", "operations manager",
    "operations", "project manager", "business analyst",
    "legal", "finance", "procurement", "supply chain",
    "receptionist", "administrator", "administrative",
}

# Titles that indicate strong alignment
RELEVANT_TITLE_TOKENS = {
    "ml engineer", "machine learning", "ai engineer", "artificial intelligence",
    "nlp engineer", "search engineer", "ranking engineer",
    "data scientist", "applied scientist", "research engineer",
    "data engineer", "senior data engineer", "lead engineer",
    "software engineer", "backend engineer", "platform engineer",
    "retrieval", "recommendations", "recsys", "information retrieval",
    "analytics engineer",
}

PROD_DEPLOYMENT_KEYWORDS = {
    "production", "deployed", "real users", "million", "billion",
    "scale", "latency", "uptime", "sla", "serving", "api",
    "microservice", "pipeline", "platform",
}
RETRIEVAL_KEYWORDS = {
    "retrieval", "rag", "embedding", "semantic search", "dense retrieval",
    "rerank", "reranking", "approximate nearest neighbor", "ann",
    "vector search", "similarity search", "hybrid search",
    "recommendation", "recommendations", "search ranking",
    "information retrieval",
}
VECTOR_DB_KEYWORDS = {
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "vector db", "vector database",
    "ann index", "hnsw", "ivf",
}
EVAL_KEYWORDS = {
    "ndcg", "mrr", "map", "precision@", "recall@",
    "ab test", "a/b test", "a/b", "ranking quality",
    "offline eval", "online eval", "evaluation metric",
    "click-through rate", "ctr improvement",
}
PYTHON_KEYWORDS = {
    "python", "fastapi", "flask", "pandas", "numpy",
    "pytorch", "tensorflow", "scikit-learn", "transformers",
    "huggingface", "langchain",
}
PRODUCT_KEYWORDS = {
    "product company", "startup", "saas", "b2b", "b2c",
    "consumer", "platform", "app", "marketplace",
}
RESEARCH_KEYWORDS = {
    "academic", "lab", "phd", "university", "research lab",
    "publication", "paper", "arxiv", "journal", "conference paper",
    "thesis", "dissertation", "postdoc",
}


@dataclass
class CandidateFeatures:
    candidate_id: str
    # Profile-level fields (for reasoning)
    name: str
    current_title: str
    current_company: str
    location_str: str
    country: str
    # Core fit signals
    years_total: float
    years_relevant: float
    title_relevance_score: float        # alignment of current_title to JD
    retrieval_prod_score: float         # evidence of production retrieval/embedding
    vector_db_score: float              # vector DB hands-on evidence
    python_evidence_score: float        # Python in career history
    ranking_eval_score: float           # NDCG/MRR style evaluation
    product_experience_score: float     # product company work evidence
    career_corroboration_score: float   # skills corroborated by career descriptions
    # Penalty flags
    services_only_flag: float
    research_only_flag: float
    keyword_stuffing_flag: float        # title/career vs skills mismatch
    # Location & availability
    location_match_score: float
    notice_period_score: float
    redrob_behavioral_multiplier: float
    # Consistency & honeypot
    consistency_penalty: float
    is_honeypot: float


def _profile(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract nested profile sub-object, falling back to top-level for compat."""
    p = candidate.get("profile")
    if isinstance(p, dict):
        return p
    return {}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _safe_lower(value: Any) -> str:
    return _to_text(value).lower()


def _parse_date(value: Any) -> Optional[datetime]:
    text = _to_text(value).strip()
    if not text:
        return None
    if text.lower() in {"present", "current", "now", "ongoing"}:
        return datetime.now(timezone.utc)
    try:
        parsed = dt_parser.parse(text, default=datetime(2000, 1, 1, tzinfo=timezone.utc))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _months_between(start: Optional[datetime], end: Optional[datetime]) -> float:
    if not start or not end:
        return 0.0
    if end < start:
        return 0.0
    return (end.year - start.year) * 12 + (end.month - start.month) + max(0.0, (end.day - start.day) / 30.0)


def _iter_career_entries(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = candidate.get("career_history") or []
    if isinstance(entries, dict):
        return [entries]
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def _iter_skills(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    skills = candidate.get("skills") or []
    if isinstance(skills, dict):
        return [skills]
    if isinstance(skills, list):
        return [s for s in skills if isinstance(s, dict)]
    return []


def _match_count(text: str, terms: set) -> int:
    return sum(1 for t in terms if t in text)


def _normalize(value: float, scale: float = 5.0) -> float:
    return max(0.0, min(1.0, value / scale))


def _years_soft_band_score(years_total: float, lo: float = 5.0, hi: float = 9.0) -> float:
    if lo <= years_total <= hi:
        return 1.0
    center = (lo + hi) / 2.0
    spread = 2.5
    return max(0.0, math.exp(-((years_total - center) ** 2) / (2 * spread ** 2)))


def _title_relevance_score(title: str, career_entries: List[Dict[str, Any]]) -> float:
    """Score how relevant the candidate's title is for a Senior AI Engineer role."""
    title_lower = _safe_lower(title)
    
    # Check for clearly irrelevant titles
    for irr in IRRELEVANT_TITLE_TOKENS:
        if irr in title_lower:
            # Hard check: is the career evidence clearly not AI-related?
            career_text = " ".join(_safe_lower(e.get("description", "")) for e in career_entries)
            ai_hits = _match_count(career_text, RETRIEVAL_KEYWORDS | VECTOR_DB_KEYWORDS | PYTHON_KEYWORDS)
            if ai_hits < 3:
                return 0.02  # Irrelevant title with no AI career evidence
            else:
                return 0.25  # Irrelevant title but some AI career mentions (title mismatch case)
    
    # Check for relevant title signals
    relevance = 0.0
    for rel in RELEVANT_TITLE_TOKENS:
        if rel in title_lower:
            relevance += 0.3
    
    # Partial signals
    if any(t in title_lower for t in ["engineer", "developer", "scientist", "analyst"]):
        relevance += 0.1
    if any(t in title_lower for t in ["senior", "lead", "staff", "principal", "architect"]):
        relevance += 0.1
    if any(t in title_lower for t in ["ml", "ai", "nlp", "search", "data"]):
        relevance += 0.25
    
    return min(1.0, relevance)


def _company_type_score(
    company: str,
    industry: str,
    career_entries: List[Dict[str, Any]],
) -> tuple[float, float, float]:
    """Return (product_score, services_only_flag, research_only_flag)."""
    company_lower = _safe_lower(company)
    industry_lower = _safe_lower(industry)

    # Check current company against known services companies
    current_is_services = any(svc in company_lower for svc in SERVICES_COMPANIES)
    industry_is_services = "it services" in industry_lower or "consulting" in industry_lower

    # Check if current company is academic/research
    current_is_academic = any(ac in company_lower for ac in ACADEMIC_COMPANIES)

    # Analyse career history for product vs services vs academic evidence
    all_career_companies = [_safe_lower(e.get("company", "")) for e in career_entries]
    career_services_count = sum(
        1 for co in all_career_companies
        if any(svc in co for svc in SERVICES_COMPANIES)
    )
    career_academic_count = sum(
        1 for co in all_career_companies
        if any(ac in co for ac in ACADEMIC_COMPANIES)
    )
    # Product = not services AND not academic
    career_product_count = len(all_career_companies) - career_services_count - career_academic_count

    # Research detection from career descriptions
    all_career_text = " ".join(_safe_lower(e.get("description", "")) for e in career_entries)
    all_titles_text = " ".join(_safe_lower(e.get("title", "")) for e in career_entries)
    research_hits = _match_count(all_career_text, RESEARCH_KEYWORDS) + _match_count(all_titles_text, {"researcher", "postdoc", "phd student", "doctoral"})
    # Specifically check for production deployment signals (not just mention of retrieval)
    production_hits = _match_count(all_career_text, PROD_DEPLOYMENT_KEYWORDS)
    retrieval_hits = _match_count(all_career_text, RETRIEVAL_KEYWORDS | PROD_DEPLOYMENT_KEYWORDS)

    # Product experience score
    product_kw_hits = _match_count(all_career_text, PRODUCT_KEYWORDS)
    product_score = _normalize(product_kw_hits + career_product_count, 6.0)

    # Services-only: currently at services AND career mostly services AND minimal product work
    n_entries = max(1, len(career_entries))
    services_fraction = career_services_count / n_entries
    services_only_flag = 1.0 if (
        (current_is_services or industry_is_services)
        and services_fraction >= 0.6
        and career_product_count == 0
        and retrieval_hits < 2
    ) else 0.0

    # Research-only: academic career with no *production* deployment evidence
    # Key distinction: retrieval mentions in academic papers != production deployment
    academic_fraction = career_academic_count / n_entries
    research_only_flag = 1.0 if (
        (research_hits >= 2 or current_is_academic or academic_fraction >= 0.5)
        and production_hits < 2       # No production deployment signals
        and career_product_count == 0
        and services_fraction < 0.3
    ) else 0.0


    return product_score, services_only_flag, research_only_flag



def _keyword_stuffing_flag(
    skills: List[Dict[str, Any]],
    title: str,
    career_entries: List[Dict[str, Any]],
) -> float:
    """Detect title/career vs skills mismatch (JD's designed-in trap)."""
    title_lower = _safe_lower(title)
    
    # Is the current title clearly non-technical?
    is_irrelevant_title = any(irr in title_lower for irr in IRRELEVANT_TITLE_TOKENS)
    
    if not is_irrelevant_title:
        # Also check if claims expert AI skills with zero duration (honeypot precursor)
        high_claim_zero = sum(
            1 for s in skills
            if _safe_lower(s.get("proficiency")) in {"expert", "advanced"}
            and float(s.get("duration_months") or 0) <= 0
        )
        all_text = " ".join(_safe_lower(e.get("description", "")) for e in career_entries)
        dense_kw = _match_count(all_text, RETRIEVAL_KEYWORDS | VECTOR_DB_KEYWORDS | EVAL_KEYWORDS)
        if high_claim_zero >= 2 and dense_kw >= 6:
            return 0.8  # Not honeypot but suspicious
        return 0.0
    
    # Irrelevant title — check if skills list is AI-keyword-stuffed
    ai_skill_count = sum(
        1 for s in skills
        if any(
            kw in _safe_lower(s.get("name", ""))
            for kw in ["embedding", "rag", "vector", "llm", "bert", "transformer",
                       "pinecone", "qdrant", "ndcg", "ranking", "retrieval"]
        )
    )
    if ai_skill_count >= 2:
        return 1.0  # Classic keyword-stuffer trap: irrelevant title + AI skill claims
    
    return 0.0


def _career_corroboration_score(
    skills: List[Dict[str, Any]],
    career_entries: List[Dict[str, Any]],
) -> float:
    """Check how well the skills list is corroborated by career descriptions."""
    if not skills:
        return 0.5  # neutral if no skills listed
    
    all_career_text = " ".join(
        _safe_lower(e.get("description", "")) + " " + _safe_lower(e.get("title", ""))
        for e in career_entries
    )
    
    corroborated = 0
    total_checked = 0
    for s in skills:
        skill_name = _safe_lower(s.get("name", ""))
        if not skill_name or len(skill_name) < 3:
            continue
        total_checked += 1
        if skill_name in all_career_text:
            corroborated += 1
    
    if total_checked == 0:
        return 0.5
    
    return corroborated / total_checked


def _behavioral_multiplier(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals") or {}
    if not isinstance(signals, dict):
        signals = {}

    open_to_work = 1.0 if bool(signals.get("open_to_work_flag", False)) else 0.7

    def rate(name: str, neutral: float = 0.5) -> float:
        raw = signals.get(name, -1)
        try:
            v = float(raw)
        except Exception:
            return neutral
        if v < 0:
            return neutral
        if v > 1.0:
            v /= 100.0
        return max(0.0, min(1.0, v))

    recruiter_rate = rate("recruiter_response_rate")
    interview_rate = rate("interview_completion_rate")
    offer_rate = rate("offer_acceptance_rate")

    notice_days_raw = signals.get("notice_period_days", 30)
    try:
        notice_days = float(notice_days_raw)
    except Exception:
        notice_days = 30.0

    if notice_days <= 0:
        notice_component = 1.0
    elif notice_days <= 30:
        notice_component = 1.0
    elif notice_days <= 60:
        notice_component = 0.8
    elif notice_days <= 90:
        notice_component = 0.6
    else:
        notice_component = 0.45

    last_active_component = 0.8
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active is not None:
        age_days = (datetime.now(timezone.utc) - last_active).days
        if age_days <= 14:
            last_active_component = 1.0
        elif age_days <= 45:
            last_active_component = 0.9
        elif age_days <= 90:
            last_active_component = 0.75
        else:
            last_active_component = 0.6

    trust = 0.92
    for flag_name in ["verified_email", "verified_phone", "linkedin_connected"]:
        if bool(signals.get(flag_name, False)):
            trust += 0.03
    trust = min(1.0, trust)

    composite = (
        0.22 * open_to_work
        + 0.16 * recruiter_rate
        + 0.16 * interview_rate
        + 0.10 * offer_rate
        + 0.20 * notice_component
        + 0.10 * last_active_component
        + 0.06 * trust
    )

    return max(0.45, min(1.10, composite))


def _location_match_score(candidate: Dict[str, Any], profile: Dict[str, Any]) -> float:
    preferred = {"pune", "noida"}
    accepted = {"hyderabad", "pune", "mumbai", "delhi ncr", "delhi", "gurgaon", "noida",
                "bengaluru", "bangalore", "india"}

    # Pull location from profile fields (primary source)
    location_parts = [
        _safe_lower(profile.get("location", "")),
        _safe_lower(profile.get("country", "")),
        # Also check top-level fallback fields for compatibility
        _safe_lower(candidate.get("location", "")),
        _safe_lower(candidate.get("current_location", "")),
    ]
    city_text = " ".join(location_parts)
    country = _safe_lower(profile.get("country", ""))

    if any(p in city_text for p in preferred):
        base = 1.0
    elif any(a in city_text for a in accepted):
        base = 0.86
    elif country == "india" or "india" in city_text:
        base = 0.72
    else:
        base = 0.55  # International — no visa sponsorship, meaningful down-weight

    signals = candidate.get("redrob_signals") or {}
    willing_relocate = bool(signals.get("willing_to_relocate", False))
    preferred_mode = _safe_lower(signals.get("preferred_work_mode", ""))
    hybrid_ok = preferred_mode in {"hybrid", "any", "onsite"} or preferred_mode == ""

    # Down-weight international candidates who won't relocate
    if base < 0.7 and not willing_relocate:
        base *= 0.7

    if not hybrid_ok:
        base *= 0.82

    return max(0.3, min(1.0, base))


def _notice_period_score(candidate: Dict[str, Any]) -> float:
    signals = candidate.get("redrob_signals") or {}
    try:
        days = float(signals.get("notice_period_days", 30))
    except Exception:
        days = 30.0
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.75
    return 0.5


def _consistency_and_honeypot(
    years_total: float,
    career_entries: List[Dict[str, Any]],
    skills: List[Dict[str, Any]],
) -> tuple[float, float]:
    issues = 0
    now = datetime.now(timezone.utc)

    summed_months = 0.0
    for e in career_entries:
        start = _parse_date(e.get("start_date"))
        end = _parse_date(e.get("end_date") or "present")
        stated_months_raw = e.get("duration_months")
        inferred = _months_between(start, end)

        # Future start dates are impossible
        if start and start > now:
            issues += 2

        # More than 35 years at one company is near-impossible
        if inferred > 420:
            issues += 2
        elif inferred > 300:
            issues += 1

        summed_months += inferred

        if stated_months_raw is not None:
            try:
                stated = float(stated_months_raw)
                # Stated vs inferred diverges by >60% (and >6 months)
                if inferred > 12 and abs(stated - inferred) > max(6.0, inferred * 0.6):
                    issues += 1
            except Exception:
                pass

    # Total career months wildly exceeds claimed years
    if years_total > 0 and summed_months > 0 and years_total * 12 > summed_months * 1.8:
        issues += 1

    # Skills with "expert" / "advanced" proficiency but zero duration_months
    for s in skills:
        proficiency = _safe_lower(s.get("proficiency"))
        try:
            months = float(s.get("duration_months") or 0)
        except Exception:
            months = 0.0
        if proficiency in {"expert", "advanced"} and months <= 0:
            issues += 1

    is_honeypot = 1.0 if issues >= 3 else 0.0
    consistency_penalty = 1.0 if issues == 0 else max(0.05, 1.0 - 0.30 * issues)
    return consistency_penalty, is_honeypot


def extract_features(candidate: Dict[str, Any]) -> CandidateFeatures:
    candidate_id = _to_text(candidate.get("candidate_id"))
    profile = _profile(candidate)

    # ------------------------------------------------------------------
    # Profile-level fields (all from nested profile sub-object)
    # ------------------------------------------------------------------
    name = _to_text(profile.get("anonymized_name") or profile.get("name") or candidate.get("name") or "")
    current_title = _to_text(
        profile.get("current_title") or profile.get("headline") or candidate.get("current_title") or ""
    )
    current_company = _to_text(
        profile.get("current_company") or candidate.get("current_company") or ""
    )
    current_industry = _to_text(
        profile.get("current_industry") or candidate.get("current_industry") or ""
    )
    location_str = _to_text(profile.get("location") or candidate.get("location") or "")
    country = _to_text(profile.get("country") or candidate.get("country") or "")

    # Years of experience — prefer explicit profile field
    years_raw = profile.get("years_of_experience") or candidate.get("total_experience_years") or candidate.get("experience_years")
    try:
        years_total = float(years_raw)
    except Exception:
        # Fall back to summing career dates
        career_months = 0.0
        for e in _iter_career_entries(candidate):
            career_months += _months_between(_parse_date(e.get("start_date")), _parse_date(e.get("end_date") or "present"))
        years_total = career_months / 12.0

    # ------------------------------------------------------------------
    # Text extraction for scoring
    # ------------------------------------------------------------------
    career_entries = _iter_career_entries(candidate)
    skills = _iter_skills(candidate)

    summary_text = _safe_lower(
        profile.get("summary") or candidate.get("summary") or ""
    )
    headline_text = _safe_lower(
        profile.get("headline") or candidate.get("headline") or ""
    )

    all_career_text = "\n".join(
        _safe_lower(e.get("title", "")) + " " +
        _safe_lower(e.get("company", "")) + " " +
        _safe_lower(e.get("description", ""))
        for e in career_entries
    )
    skills_text = "\n".join(
        _safe_lower(s.get("name", "")) + " " + _safe_lower(s.get("proficiency", ""))
        for s in skills
    )
    all_text = "\n".join([summary_text, headline_text, all_career_text, skills_text])

    # ------------------------------------------------------------------
    # Title relevance
    # ------------------------------------------------------------------
    title_relevance_score = _title_relevance_score(current_title, career_entries)

    # ------------------------------------------------------------------
    # Company type classification
    # ------------------------------------------------------------------
    product_experience_score, services_only_flag, research_only_flag = _company_type_score(
        current_company, current_industry, career_entries
    )

    # ------------------------------------------------------------------
    # Career signal extraction
    # ------------------------------------------------------------------
    retrieval_raw = _match_count(all_career_text, RETRIEVAL_KEYWORDS) + _match_count(all_career_text, PROD_DEPLOYMENT_KEYWORDS)
    vector_raw = _match_count(all_career_text, VECTOR_DB_KEYWORDS)
    python_raw = _match_count(all_career_text + " " + skills_text, PYTHON_KEYWORDS)
    eval_raw = _match_count(all_career_text, EVAL_KEYWORDS)

    retrieval_prod_score = _normalize(retrieval_raw, 8.0)
    vector_db_score = _normalize(vector_raw, 4.0)
    python_evidence_score = _normalize(python_raw + (1 if "python" in all_career_text else 0), 5.0)
    ranking_eval_score = _normalize(eval_raw, 4.0)

    # Relevant years = proportion of career spent on relevant tech
    relevant_hits = _match_count(all_career_text, RETRIEVAL_KEYWORDS | VECTOR_DB_KEYWORDS | PYTHON_KEYWORDS)
    years_relevant = min(years_total, years_total * _normalize(relevant_hits, 8.0))

    # ------------------------------------------------------------------
    # Skill corroboration (skills vs career description evidence)
    # ------------------------------------------------------------------
    career_corroboration_score = _career_corroboration_score(skills, career_entries)

    # ------------------------------------------------------------------
    # Keyword stuffing detection (title-skills mismatch trap)
    # ------------------------------------------------------------------
    keyword_stuffing_flag = _keyword_stuffing_flag(skills, current_title, career_entries)

    # ------------------------------------------------------------------
    # Location, notice period, behavioral multiplier
    # ------------------------------------------------------------------
    location_match_score = _location_match_score(candidate, profile)
    notice_period_score = _notice_period_score(candidate)
    behavioral = _behavioral_multiplier(candidate)

    # ------------------------------------------------------------------
    # Consistency & honeypot checks
    # ------------------------------------------------------------------
    consistency_penalty, is_honeypot = _consistency_and_honeypot(years_total, career_entries, skills)

    return CandidateFeatures(
        candidate_id=candidate_id,
        name=name,
        current_title=current_title,
        current_company=current_company,
        location_str=location_str,
        country=country,
        years_total=years_total,
        years_relevant=years_relevant,
        title_relevance_score=title_relevance_score,
        retrieval_prod_score=retrieval_prod_score,
        vector_db_score=vector_db_score,
        python_evidence_score=python_evidence_score,
        ranking_eval_score=ranking_eval_score,
        product_experience_score=product_experience_score,
        career_corroboration_score=career_corroboration_score,
        services_only_flag=services_only_flag,
        research_only_flag=research_only_flag,
        keyword_stuffing_flag=keyword_stuffing_flag,
        location_match_score=location_match_score,
        notice_period_score=notice_period_score,
        redrob_behavioral_multiplier=behavioral,
        consistency_penalty=consistency_penalty,
        is_honeypot=is_honeypot,
    )


def candidate_text_for_embedding(candidate: Dict[str, Any]) -> str:
    """Build the text blob used for semantic embedding — use ALL relevant profile text."""
    profile = _profile(candidate)
    career_entries = _iter_career_entries(candidate)
    skills = _iter_skills(candidate)

    parts = [
        _to_text(profile.get("summary") or candidate.get("summary") or ""),
        _to_text(profile.get("headline") or candidate.get("headline") or ""),
        _to_text(profile.get("current_title") or ""),
        _to_text(profile.get("current_company") or ""),
        _to_text(profile.get("current_industry") or ""),
    ]
    for e in career_entries:
        parts.append(_to_text(e.get("title", "")))
        parts.append(_to_text(e.get("company", "")))
        parts.append(_to_text(e.get("description", "")))
    for s in skills:
        parts.append(_to_text(s.get("name", "")))
    return "\n".join(p for p in parts if p)


def read_candidates(candidates_path: Path) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if candidates_path.suffix == ".gz" else open
    mode = "rt"
    with opener(candidates_path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def features_to_dict(features: CandidateFeatures) -> Dict[str, Any]:
    out = asdict(features)
    out["years_soft_band_score"] = _years_soft_band_score(features.years_total)
    return out
