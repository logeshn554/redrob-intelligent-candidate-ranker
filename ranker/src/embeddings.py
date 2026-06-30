from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    normalize: bool = True


class LocalEmbedder:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._backend = "sentence_transformers"
        self._model = None
        self._fallback = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name)
            self._backend = "sentence_transformers"
        except Exception:
            from sklearn.feature_extraction.text import HashingVectorizer

            self._fallback = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)
            self._backend = "hashing"

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._backend == "sentence_transformers":
            emb = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize,
            )
            return np.asarray(emb, dtype=np.float32)

        mat = self._fallback.transform(texts)
        arr = mat.toarray().astype(np.float32)
        if self.config.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            arr = arr / norms
        return arr

    @property
    def embedding_dim(self) -> int:
        if self._backend == "sentence_transformers" and self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return 384


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]
    scores = doc_matrix @ q.T
    return scores.squeeze(-1)
