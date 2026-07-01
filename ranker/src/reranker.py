"""
Cross-encoder reranker — small local transformer model for LLM-style ranking.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (~66 MB, CPU-only) to score
(JD_query, candidate_text) pairs for the top-N candidates shortlisted by
the hybrid scoring pipeline.

Why cross-encoder over bi-encoder?
- Bi-encoders (all-MiniLM) embed query and document independently → fast
  for 100K candidates but misses token-level interactions.
- Cross-encoders attend jointly over (query, document) → much higher quality
  relevance signal, but too slow for 100K candidates.
- Solution: hybrid pipeline shortlists top-200 candidates in <5s, then
  cross-encoder reranks those 200 in ~15-25s on CPU.

The final score blends both signals:
  final = alpha * cross_encoder_score + (1 - alpha) * hybrid_score
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from sentence_transformers import CrossEncoder
    _CE_AVAILABLE = True
except ImportError:
    _CE_AVAILABLE = False


DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CandidateReranker:
    """
    Wraps a sentence-transformers CrossEncoder for candidate reranking.
    """

    def __init__(self, model_name: str = DEFAULT_CE_MODEL) -> None:
        if not _CE_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Run: pip install sentence-transformers"
            )
        # Load cross-encoder (cached after first run)
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name

    def score(self, query: str, candidate_texts: List[str]) -> np.ndarray:
        """
        Score each (query, candidate_text) pair.

        Returns an ndarray of shape (N,) with raw logit scores.
        Higher = more relevant.
        """
        pairs: List[Tuple[str, str]] = [(query, t) for t in candidate_texts]
        raw = self.model.predict(pairs, show_progress_bar=False)
        return np.array(raw, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidate_texts: List[str],
        hybrid_scores: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Blend cross-encoder score with hybrid score.

        alpha controls cross-encoder weight:
          final = alpha * ce_norm + (1 - alpha) * hybrid_score

        Returns blended scores in same order as input.
        """
        ce_raw = self.score(query, candidate_texts)

        # Min-max normalize cross-encoder scores to [0, 1]
        lo, hi = ce_raw.min(), ce_raw.max()
        if abs(hi - lo) < 1e-8:
            ce_norm = np.full_like(ce_raw, 0.5)
        else:
            ce_norm = (ce_raw - lo) / (hi - lo)

        # Normalize hybrid scores to [0, 1] as well
        h_lo, h_hi = hybrid_scores.min(), hybrid_scores.max()
        if abs(h_hi - h_lo) < 1e-8:
            h_norm = np.full_like(hybrid_scores, 0.5, dtype=np.float32)
        else:
            h_norm = (hybrid_scores - h_lo) / (h_hi - h_lo)

        blended = alpha * ce_norm + (1.0 - alpha) * h_norm
        return blended.astype(np.float32)
