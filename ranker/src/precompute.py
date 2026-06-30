from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .embeddings import EmbeddingConfig, LocalEmbedder
    from .features import candidate_text_for_embedding, extract_features, features_to_dict, read_candidates
    from .parse_jd import parse_jd_file
except ImportError:
    from embeddings import EmbeddingConfig, LocalEmbedder
    from features import candidate_text_for_embedding, extract_features, features_to_dict, read_candidates
    from parse_jd import parse_jd_file


def precompute(
    candidates_path: Path,
    jd_path: Path,
    out_dir: Path,
    model_name: str,
    batch_size: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_jd = parse_jd_file(jd_path)
    (out_dir / "parsed_jd.json").write_text(json.dumps(parsed_jd.__dict__, indent=2), encoding="utf-8")

    embedder = LocalEmbedder(EmbeddingConfig(model_name=model_name, batch_size=batch_size))

    candidate_ids: List[str] = []
    feature_rows: List[Dict[str, Any]] = []
    texts: List[str] = []

    print("Pass 1/2: Extracting features and building text corpus...")
    for cand in tqdm(read_candidates(candidates_path), desc="Candidates", unit="cand"):
        feats = extract_features(cand)
        candidate_ids.append(feats.candidate_id)
        feature_rows.append(features_to_dict(feats))
        texts.append(candidate_text_for_embedding(cand))

    print(f"\nExtracted {len(candidate_ids)} candidates.")
    honeypot_count = sum(1 for r in feature_rows if r.get("is_honeypot", 0) > 0.5)
    services_count = sum(1 for r in feature_rows if r.get("services_only_flag", 0) > 0.5)
    title_relevant = sum(1 for r in feature_rows if r.get("title_relevance_score", 0) > 0.3)
    print(f"  Honeypot flagged: {honeypot_count}")
    print(f"  Services-only flagged: {services_count}")
    print(f"  Title-relevant candidates: {title_relevant}")

    # Save features to CSV (numeric + string columns)
    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(out_dir / "candidate_features.csv", index=False)
    pd.DataFrame({"candidate_id": candidate_ids}).to_csv(out_dir / "candidate_ids.csv", index=False)
    print(f"\nSaved features to {out_dir / 'candidate_features.csv'}")

    print("\nPass 2/2: Computing embeddings...")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        emb = embedder.encode(batch)
        embeddings.append(emb)

    emb_matrix = (
        np.vstack(embeddings).astype(np.float32)
        if embeddings
        else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
    )
    np.save(out_dir / "candidate_embeddings.npy", emb_matrix)
    print(f"Saved embeddings: shape={emb_matrix.shape} to {out_dir / 'candidate_embeddings.npy'}")

    return {
        "candidate_count": len(candidate_ids),
        "honeypot_count": honeypot_count,
        "services_only_count": services_count,
        "title_relevant_count": title_relevant,
        "embedding_dim": int(emb_matrix.shape[1]) if emb_matrix.size else int(embedder.embedding_dim),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute features and candidate embeddings (offline step)")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--jd", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("./artifacts"), type=Path)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", default=256, type=int)
    args = parser.parse_args()

    summary = precompute(
        candidates_path=args.candidates,
        jd_path=args.jd,
        out_dir=args.out_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
