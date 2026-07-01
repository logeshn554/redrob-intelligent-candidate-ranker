from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from .embeddings import EmbeddingConfig, LocalEmbedder, cosine_similarity
    from .features import candidate_text_for_embedding, extract_features, features_to_dict, read_candidates
    from .parse_jd import parse_jd_file
    from .reasoning import build_reasoning
    from .reranker import CandidateReranker
    from .scoring import combine_scores
except ImportError:
    from embeddings import EmbeddingConfig, LocalEmbedder, cosine_similarity
    from features import candidate_text_for_embedding, extract_features, features_to_dict, read_candidates
    from parse_jd import parse_jd_file
    from reasoning import build_reasoning
    from reranker import CandidateReranker
    from scoring import combine_scores


def _assert_required_inputs(candidates_path: Path, jd_path: Path) -> None:
    missing: List[str] = []
    if not jd_path.exists():
        missing.append(str(jd_path))
    if not candidates_path.exists():
        missing.append(str(candidates_path))
    if missing:
        msg = (
            "Required input files were not found: "
            + ", ".join(missing)
            + ". Place the official hackathon bundle files under ./data and retry."
        )
        raise FileNotFoundError(msg)


def _load_precomputed(artifacts_dir: Path) -> tuple[List[str], pd.DataFrame, np.ndarray]:
    ids = pd.read_csv(artifacts_dir / "candidate_ids.csv")["candidate_id"].tolist()
    feats = pd.read_csv(artifacts_dir / "candidate_features.csv")
    embs = np.load(artifacts_dir / "candidate_embeddings.npy")
    return ids, feats, embs


def _compute_on_the_fly(
    candidates_path: Path, model_name: str, batch_size: int
) -> tuple[List[str], pd.DataFrame, np.ndarray]:
    embedder = LocalEmbedder(EmbeddingConfig(model_name=model_name, batch_size=batch_size))
    ids: List[str] = []
    feat_rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    for cand in read_candidates(candidates_path):
        feats = extract_features(cand)
        ids.append(feats.candidate_id)
        feat_rows.append(features_to_dict(feats))
        texts.append(candidate_text_for_embedding(cand))

    emb_batches = []
    for i in range(0, len(texts), batch_size):
        emb_batches.append(embedder.encode(texts[i : i + batch_size]))
    embs = np.vstack(emb_batches).astype(np.float32) if emb_batches else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
    return ids, pd.DataFrame(feat_rows), embs


def _parse_data_understanding(feature_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "candidate_count": int(len(feature_df)),
        "honeypot_flagged_count": int((feature_df["is_honeypot"] > 0.5).sum()),
        "research_only_count": int((feature_df["research_only_flag"] > 0.5).sum()),
        "services_only_count": int((feature_df["services_only_flag"] > 0.5).sum()),
        "keyword_stuffing_count": int((feature_df["keyword_stuffing_flag"] > 0.5).sum()),
        "median_years_total": float(feature_df["years_total"].median()),
        "title_relevant_count": int((feature_df["title_relevance_score"] > 0.3).sum()),
    }


def rank_candidates(
    candidates_path: Path,
    jd_path: Path,
    out_csv: Path,
    artifacts_dir: Path,
    model_name: str,
    batch_size: int,
    top_k: int,
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top: int = 200,
    ce_alpha: float = 0.45,
) -> Dict[str, Any]:
    t0 = time.time()
    _assert_required_inputs(candidates_path, jd_path)
    jd = parse_jd_file(jd_path)
    jd_text = jd.summary_text

    precomputed_ok = (
        (artifacts_dir / "candidate_embeddings.npy").exists()
        and (artifacts_dir / "candidate_features.csv").exists()
        and (artifacts_dir / "candidate_ids.csv").exists()
    )

    if precomputed_ok:
        candidate_ids, feature_df, candidate_embs = _load_precomputed(artifacts_dir)
    else:
        candidate_ids, feature_df, candidate_embs = _compute_on_the_fly(
            candidates_path, model_name, batch_size
        )

    embedder = LocalEmbedder(EmbeddingConfig(model_name=model_name, batch_size=batch_size))
    # Rich, role-specific JD query for semantic matching
    jd_query = (
        "Senior AI / ML engineer who has shipped production retrieval and embedding systems at scale. "
        "Hands-on experience building and serving RAG pipelines, dense retrieval, semantic search, "
        "and reranking. Deep familiarity with vector databases such as Pinecone, Qdrant, Weaviate, Faiss, "
        "Milvus, or Elasticsearch. Strong Python skills with FastAPI, PyTorch, Hugging Face Transformers. "
        "Has evaluated ranking quality using NDCG, MRR, MAP, A/B tests. "
        "5 to 9 years of industry experience, currently at a product company (not services outsourcing). "
        "Based in India, preferably Bangalore, Pune, or Noida. Open to hybrid work. "
        + jd_text[:800]
    )
    jd_vec = embedder.encode([jd_query])[0]
    semantic = cosine_similarity(jd_vec, candidate_embs)

    # Clamp top_k to available candidates
    effective_top_k = min(top_k, len(candidate_ids))

    ranked = combine_scores(candidate_ids, feature_df, semantic)

    # ── Stage 2: Cross-encoder reranking on shortlist ────────────────────────
    # Take top (rerank_top) candidates from hybrid scoring, then apply a
    # cross-encoder for LLM-style (query, document) relevance scoring.
    shortlist_size = max(top_k, rerank_top)
    shortlist = ranked.head(shortlist_size).copy().reset_index(drop=True)

    try:
        reranker = CandidateReranker(model_name=ce_model)
        # Build reranking query — concise, matches cross-encoder training domain
        ce_query = (
            "Senior AI engineer with production experience in RAG, dense retrieval, "
            "vector databases (Pinecone, Qdrant, Faiss), Python, NDCG/MRR evaluation, "
            "at a product company in India."
        )
        # Build candidate text blobs for cross-encoder
        shortlist_ids = shortlist["candidate_id"].tolist()
        # Reconstruct candidate texts from feature columns for reranking
        ce_texts = [
            f"{row['current_title']} at {row['current_company']}. "
            f"{row['years_total']:.1f} years experience. Location: {row['location_str']}. "
            f"Retrieval score: {row['retrieval_prod_score']:.2f}. "
            f"Vector DB score: {row['vector_db_score']:.2f}. "
            f"Python score: {row['python_evidence_score']:.2f}."
            for _, row in shortlist.iterrows()
        ]
        hybrid_arr = shortlist["score"].to_numpy(dtype=np.float32)
        blended = reranker.rerank(ce_query, ce_texts, hybrid_arr, alpha=ce_alpha)
        shortlist["score"] = blended
        shortlist["reranked"] = True
    except Exception as e:
        # Graceful fallback — if reranker fails, continue with hybrid scores
        import sys
        print(f"[reranker] Warning: cross-encoder reranking skipped ({e})", file=sys.stderr)
        shortlist["reranked"] = False

    # Sort by blended score, apply tie-break, take top_k
    shortlist = shortlist.sort_values(
        ["score", "candidate_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    top = shortlist.head(effective_top_k).copy()

    top["reasoning"] = top.apply(lambda r: build_reasoning(r.to_dict()), axis=1)
    submission = top[["candidate_id", "rank", "score", "reasoning"]].copy()

    # Re-enforce strict score monotonicity and deterministic tie behavior
    submission = submission.sort_values(
        ["score", "candidate_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    submission["rank"] = np.arange(1, len(submission) + 1)

    # ── Output score normalization ──────────────────────────────────────────
    # Stretch top-K scores to [SCORE_MIN, SCORE_MAX] so rank 1 ≈ 0.97 and
    # rank 100 ≈ 0.10, making the distribution intuitive without changing order.
    SCORE_MAX = 0.97
    SCORE_MIN = 0.10
    raw = submission["score"].to_numpy(dtype=np.float64)
    lo, hi = raw.min(), raw.max()
    if abs(hi - lo) > 1e-9:
        # Map: highest raw score → SCORE_MAX, lowest → SCORE_MIN
        normalized = SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (raw - lo) / (hi - lo)
        # raw is sorted descending (rank 1 = highest), so normalized is already descending
        submission["score"] = normalized
    # Re-clip to be safe
    submission["score"] = submission["score"].clip(SCORE_MIN, SCORE_MAX).round(6)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    summary = _parse_data_understanding(feature_df)
    summary["runtime_seconds"] = round(time.time() - t0, 3)
    summary["output_rows"] = int(len(submission))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Redrob submission CSV")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--jd", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--artifacts-dir", default=Path("./artifacts"), type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Cross-encoder model for reranking shortlist")
    parser.add_argument("--rerank-top", type=int, default=200,
                        help="Number of candidates to pass to cross-encoder")
    parser.add_argument("--ce-alpha", type=float, default=0.45,
                        help="Weight for cross-encoder score in blended final score")
    args = parser.parse_args()

    summary = rank_candidates(
        candidates_path=args.candidates,
        jd_path=args.jd,
        out_csv=args.out,
        artifacts_dir=args.artifacts_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        top_k=args.top_k,
        ce_model=args.ce_model,
        rerank_top=args.rerank_top,
        ce_alpha=args.ce_alpha,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
