from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import extract_features, features_to_dict
from src.scoring import combine_scores


def _candidate(
    base_id: str,
    title: str = "ML Engineer",
    company: str = "Product Labs",
    industry: str = "Software",
    country: str = "India",
    location: str = "Pune",
    summary: str = "",
    career_desc: str = "Deployed embedding retrieval and qdrant hybrid search to production users. Ran NDCG and MRR evaluation.",
    skill_duration: int = 48,
    proficiency: str = "advanced",
    skill_name: str = "Python",
):
    return {
        "candidate_id": base_id,
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": f"{title} | AI",
            "summary": summary or f"Experienced {title} with focus on ML.",
            "location": location,
            "country": country,
            "years_of_experience": 7,
            "current_title": title,
            "current_company": company,
            "current_industry": industry,
        },
        "career_history": [
            {
                "title": title,
                "company": company,
                "start_date": "2019-01-01",
                "end_date": "2024-12-01",
                "description": career_desc,
            }
        ],
        "skills": [
            {
                "name": skill_name,
                "proficiency": proficiency,
                "duration_months": skill_duration,
            }
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.9,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": -1,
            "notice_period_days": 30,
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "last_active_date": "2026-06-25",
        },
    }


def test_strong_candidate_scores_high():
    """A genuine ML engineer with production deployment should score well."""
    strong = _candidate(
        "CAND_0000001",
        title="Senior ML Engineer",
        company="Swiggy",
        industry="Food Delivery",
        career_desc=(
            "Built production RAG pipeline using Qdrant vector database and dense retrieval "
            "embeddings deployed to 5M daily users. Ran NDCG and MRR evaluation plus A/B testing. "
            "Python, sentence-transformers, Faiss ANN search."
        ),
        skill_name="Python",
        skill_duration=48,
    )
    feats = features_to_dict(extract_features(strong))
    assert feats["title_relevance_score"] > 0.3, "Strong ML engineer title should be relevant"
    assert feats["retrieval_prod_score"] > 0.5, "Production retrieval should be detected"
    assert feats["is_honeypot"] == 0.0, "Should not be flagged as honeypot"
    assert feats["keyword_stuffing_flag"] == 0.0, "Should not be flagged as keyword stuffer"


def test_keyword_stuffer_is_penalized():
    """Marketing manager claiming AI expertise without career evidence should be penalized."""
    stuffer = _candidate(
        "CAND_0000002",
        title="Marketing Manager",
        company="Brand Co",
        industry="Consumer Goods",
        summary="Marketing professional curious about AI!",
        career_desc="Led digital marketing campaigns and brand strategy. Content planning.",
        skill_duration=0,
        proficiency="expert",
        skill_name="RAG",
    )
    # Add more AI skills to make it clearly stuffed
    stuffer["skills"] += [
        {"name": "Pinecone", "proficiency": "expert", "duration_months": 0},
        {"name": "LLM", "proficiency": "expert", "duration_months": 0},
    ]
    feats = extract_features(stuffer)
    assert feats.keyword_stuffing_flag > 0.5, "Keyword stuffer should be flagged"
    assert feats.title_relevance_score < 0.1, "Marketing Manager title should have low relevance"


def test_services_only_is_flagged():
    """A TCS/Infosys-only candidate with no product company exposure should be flagged."""
    services = _candidate(
        "CAND_0000003",
        title="Software Engineer",
        company="TCS",
        industry="IT Services",
        career_desc="Java Spring Boot development for banking clients at TCS. Oracle DB.",
        skill_name="Java",
        skill_duration=96,
    )
    services["career_history"].append({
        "title": "Developer",
        "company": "Infosys",
        "start_date": "2015-01-01",
        "end_date": "2019-01-01",
        "description": "Backend Java development. BFSI consulting.",
    })
    feats = extract_features(services)
    assert feats.services_only_flag > 0.5, "Pure services career should be flagged"


def test_research_only_is_flagged():
    """A pure academic/lab candidate should be flagged for research-only."""
    research = _candidate(
        "CAND_0000004",
        title="AI Research Engineer",
        company="IIT Research Lab",
        industry="Academia",
        career_desc=(
            "Published papers on transformer-based dense retrieval. PhD research at university lab. "
            "Thesis on information retrieval. Academic publications and paper presentations."
        ),
        skill_name="BERT",
        skill_duration=60,
    )
    research["career_history"].append({
        "title": "PhD Student",
        "company": "University of Hyderabad",
        "start_date": "2015-01-01",
        "end_date": "2019-01-01",
        "description": "Academic dissertation on dense retrieval. Journal papers, thesis on NLP.",
    })
    feats = extract_features(research)
    assert feats.research_only_flag > 0.5, "Pure academic career should be flagged as research-only"


def test_honeypot_is_flagged():
    """A candidate with impossible career dates should be flagged as honeypot."""
    honeypot = _candidate(
        "CAND_0000005",
        title="Senior AI Engineer",
        company="Startup AI",
        industry="AI/ML",
        career_desc="50 years of production vector database experience.",
        skill_duration=0,
        proficiency="expert",
    )
    honeypot["career_history"][0]["duration_months"] = 600  # 50 years at one job
    honeypot["career_history"][0]["start_date"] = "1975-01-01"
    feats = extract_features(honeypot)
    assert feats.is_honeypot > 0.5, "Impossible career duration should be flagged as honeypot"


def test_plain_substance_beats_keyword_stuffer():
    """Genuine product engineer without AI buzzwords beats keyword stuffer."""
    plain = _candidate(
        "CAND_0000006",
        title="Senior Software Engineer",
        company="Myntra",
        industry="E-Commerce",
        career_desc=(
            "Built product search ranking system with Elasticsearch approximate nearest neighbor "
            "and dense retrieval. Improved CTR by 23%. Python, production serving millions of users. "
            "Offline evaluation using NDCG@10 metrics. A/B tested ranking models."
        ),
        skill_name="Python",
        skill_duration=72,
    )
    stuffer = _candidate(
        "CAND_0000007",
        title="Marketing Manager",
        company="Corp",
        industry="Consumer Goods",
        summary="Marketing manager excited about AI!",
        career_desc="Brand strategy and digital marketing campaigns.",
        skill_name="RAG",
        skill_duration=0,
        proficiency="expert",
    )
    stuffer["skills"] += [
        {"name": "Pinecone", "proficiency": "expert", "duration_months": 0},
        {"name": "LLM", "proficiency": "expert", "duration_months": 0},
    ]

    candidates = [plain, stuffer]
    feats = [features_to_dict(extract_features(c)) for c in candidates]
    df = pd.DataFrame(feats)
    semantic = np.array([0.7, 0.7], dtype=np.float32)
    ranked = combine_scores(["CAND_0000006", "CAND_0000007"], df, semantic)
    scores = {r["candidate_id"]: float(r["score"]) for _, r in ranked.iterrows()}
    assert scores["CAND_0000006"] > scores["CAND_0000007"], (
        f"Plain substance ({scores['CAND_0000006']:.4f}) should beat keyword stuffer ({scores['CAND_0000007']:.4f})"
    )


def test_score_ordering():
    """Strong > plain substance > services-only > research-only > honeypot > keyword stuffer."""
    strong = _candidate(
        "CAND_T1",
        title="ML Engineer",
        company="Swiggy",
        industry="Food Delivery",
        career_desc="Built production RAG pipeline using Qdrant vector DB and dense retrieval embeddings deployed to millions. Ran NDCG and MRR evaluation. Python, sentence-transformers.",
        skill_duration=48,
    )
    services = _candidate(
        "CAND_T2",
        title="Software Engineer",
        company="TCS",
        industry="IT Services",
        career_desc="Java Spring Boot development for banking clients. No AI work.",
        skill_name="Java",
        skill_duration=60,
    )
    services["career_history"].append({
        "title": "Developer", "company": "Infosys", "start_date": "2015-01-01", "end_date": "2019-01-01",
        "description": "BFSI consulting. Backend Java.",
    })
    honeypot = _candidate(
        "CAND_T3",
        title="Senior AI Engineer",
        company="Startup AI",
        industry="AI/ML",
        career_desc="Expert everything AI engineer.",
        skill_duration=0,
        proficiency="expert",
    )
    honeypot["career_history"][0]["duration_months"] = 600
    honeypot["career_history"][0]["start_date"] = "1975-01-01"

    stuffer = _candidate(
        "CAND_T4",
        title="Marketing Manager",
        company="Brand Corp",
        industry="Consumer Goods",
        summary="Marketing professional excited about AI!",
        career_desc="Digital marketing campaigns.",
        skill_name="RAG",
        skill_duration=0,
        proficiency="expert",
    )
    stuffer["skills"] += [
        {"name": "Pinecone", "proficiency": "expert", "duration_months": 0},
        {"name": "LLM", "proficiency": "expert", "duration_months": 0},
    ]

    all_cands = [strong, services, honeypot, stuffer]
    feats = [features_to_dict(extract_features(c)) for c in all_cands]
    df = pd.DataFrame(feats)
    ids = [c["candidate_id"] for c in all_cands]
    semantic = np.array([0.75] * len(all_cands), dtype=np.float32)
    ranked = combine_scores(ids, df, semantic)
    scores = {r["candidate_id"]: float(r["score"]) for _, r in ranked.iterrows()}

    assert scores["CAND_T1"] > scores["CAND_T2"], "Strong should beat services-only"
    assert scores["CAND_T1"] > scores["CAND_T3"], "Strong should beat honeypot"
    assert scores["CAND_T1"] > scores["CAND_T4"], "Strong should beat keyword stuffer"
    assert scores["CAND_T2"] > scores["CAND_T3"], "Services-only should beat honeypot"
    assert scores["CAND_T4"] < 0.05, "Keyword stuffer should score near-zero"


def test_location_from_profile():
    """Test that location is read from profile.location, not top-level."""
    candidate = _candidate("CAND_L1")
    # Top-level location is None, profile.location = "Pune"
    feats = extract_features(candidate)
    assert feats.location_match_score >= 0.95, "Pune location should score 1.0"

    # Now change to international
    candidate["profile"]["location"] = "New York"
    candidate["profile"]["country"] = "USA"
    feats2 = extract_features(candidate)
    assert feats2.location_match_score < 0.8, "International location should score lower"


def test_years_from_profile():
    """Test that years are read from profile.years_of_experience."""
    candidate = _candidate("CAND_Y1")
    candidate["profile"]["years_of_experience"] = 7.5
    feats = extract_features(candidate)
    assert abs(feats.years_total - 7.5) < 0.1, "Should read years from profile field"
