from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class ScoringWeights:
    # Structured-vs-semantic split
    structured_weight: float = 0.68
    semantic_weight: float = 0.32

    # Component weights inside structured score (must sum to ~1.0)
    w_title_relevance: float = 0.22      # Current title alignment — strong signal
    w_retrieval_prod: float = 0.18       # Production embedding/retrieval evidence
    w_vector_db: float = 0.12           # Vector DB hands-on
    w_python: float = 0.10              # Python evidence in career history
    w_ranking_eval: float = 0.10        # NDCG/MRR eval experience
    w_product_exp: float = 0.06         # Product company exposure
    w_career_corroboration: float = 0.08  # Skills corroborated by career text
    w_years_soft: float = 0.06          # Experience band
    w_location: float = 0.05            # Location match
    w_notice: float = 0.03             # Notice period


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < 1e-8:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def compute_structured_score(df: pd.DataFrame, weights: ScoringWeights | None = None) -> np.ndarray:
    w = weights or ScoringWeights()

    years_soft = df["years_soft_band_score"].to_numpy(dtype=np.float32)
    title_rel = df["title_relevance_score"].to_numpy(dtype=np.float32)
    retrieval = df["retrieval_prod_score"].to_numpy(dtype=np.float32)
    vector_db = df["vector_db_score"].to_numpy(dtype=np.float32)
    python_ev = df["python_evidence_score"].to_numpy(dtype=np.float32)
    eval_sc = df["ranking_eval_score"].to_numpy(dtype=np.float32)
    product_exp = df["product_experience_score"].to_numpy(dtype=np.float32)
    corroboration = df["career_corroboration_score"].to_numpy(dtype=np.float32)
    location = df["location_match_score"].to_numpy(dtype=np.float32)
    notice = df["notice_period_score"].to_numpy(dtype=np.float32)

    base = (
        w.w_title_relevance * title_rel
        + w.w_retrieval_prod * retrieval
        + w.w_vector_db * vector_db
        + w.w_python * python_ev
        + w.w_ranking_eval * eval_sc
        + w.w_product_exp * product_exp
        + w.w_career_corroboration * corroboration
        + w.w_years_soft * years_soft
        + w.w_location * location
        + w.w_notice * notice
    )

    penalties = np.ones(len(df), dtype=np.float32)

    # Research-only → very strong penalty (pure academic, no production)
    penalties *= (1.0 - 0.88 * df["research_only_flag"].to_numpy(dtype=np.float32))

    # Services-only → moderate penalty (TCS/Wipro/Infosys with zero product company)
    penalties *= (1.0 - 0.50 * df["services_only_flag"].to_numpy(dtype=np.float32))

    # Keyword stuffing (irrelevant title + AI skills list) → near-zero
    penalties *= (1.0 - 0.95 * df["keyword_stuffing_flag"].to_numpy(dtype=np.float32))

    # Consistency penalty (mild inconsistencies)
    penalties *= df["consistency_penalty"].to_numpy(dtype=np.float32)

    # Honeypot → near-zero
    penalties *= (1.0 - 0.97 * df["is_honeypot"].to_numpy(dtype=np.float32))

    # Title hard cap: if title is clearly irrelevant (score < 0.05), don't exceed 0.12
    # This handles the "Marketing Manager" → "AI skills" keyword stuffer case
    irrelevant_title_mask = title_rel < 0.05
    result = np.clip(base * penalties, 0.0, 1.0)
    result[irrelevant_title_mask] = np.minimum(result[irrelevant_title_mask], 0.12)

    return result


def combine_scores(
    candidate_ids: List[str],
    feature_df: pd.DataFrame,
    semantic_scores: np.ndarray,
    weights: ScoringWeights | None = None,
) -> pd.DataFrame:
    weights = weights or ScoringWeights()

    semantic_norm = _minmax(semantic_scores.astype(np.float32))
    structured = compute_structured_score(feature_df, weights)

    combined = weights.structured_weight * structured + weights.semantic_weight * semantic_norm

    behavioral = feature_df["redrob_behavioral_multiplier"].to_numpy(dtype=np.float32)
    honeypot = feature_df["is_honeypot"].to_numpy(dtype=np.float32)
    honeypot_cap = 1.0 - 0.98 * honeypot
    final = np.clip(combined * behavioral * honeypot_cap, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "structured_score": structured,
            "semantic_score": semantic_norm,
            "behavioral_multiplier": behavioral,
            "score": final,
            "is_honeypot": honeypot,
            "research_only_flag": feature_df["research_only_flag"].to_numpy(dtype=np.float32),
            "services_only_flag": feature_df["services_only_flag"].to_numpy(dtype=np.float32),
            "keyword_stuffing_flag": feature_df["keyword_stuffing_flag"].to_numpy(dtype=np.float32),
            "title_relevance_score": feature_df["title_relevance_score"].to_numpy(dtype=np.float32),
            "years_total": feature_df["years_total"].to_numpy(dtype=np.float32),
            "retrieval_prod_score": feature_df["retrieval_prod_score"].to_numpy(dtype=np.float32),
            "vector_db_score": feature_df["vector_db_score"].to_numpy(dtype=np.float32),
            "python_evidence_score": feature_df["python_evidence_score"].to_numpy(dtype=np.float32),
            "ranking_eval_score": feature_df["ranking_eval_score"].to_numpy(dtype=np.float32),
            "location_match_score": feature_df["location_match_score"].to_numpy(dtype=np.float32),
            "notice_period_score": feature_df["notice_period_score"].to_numpy(dtype=np.float32),
            "career_corroboration_score": feature_df["career_corroboration_score"].to_numpy(dtype=np.float32),
            # Profile string fields for reasoning
            "name": feature_df["name"].tolist(),
            "current_title": feature_df["current_title"].tolist(),
            "current_company": feature_df["current_company"].tolist(),
            "location_str": feature_df["location_str"].tolist(),
            "country": feature_df["country"].tolist(),
        }
    )

    out = out.sort_values(["score", "candidate_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
