"""
Train-only hybrid retriever for S1 (DR-004, DR-018).

Two exact inner-product indexes over the TRAIN split only:
  - visual: L2-normalised LLaVA-Med CLIP image embeddings (query = the patient's X-ray)
  - text:   L2-normalised MedCPT article embeddings of indication + report
            (query = MedCPT query embedding of the patient's indication)
Scores are fused as  alpha * cos_visual + (1 - alpha) * cos_text;  alpha is tuned on val.

FAISS IndexFlatIP is used when it can be imported; otherwise an exact NumPy inner product gives
identical scores (FAISS's DLL is blocked by Windows Application Control on the dev machine).
"""

from typing import Sequence, Tuple

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class ExactIPIndex:
    """Exact inner-product search over a fixed matrix; FAISS-backed when available."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = l2_normalize(vectors)
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
            self.backend = "faiss.IndexFlatIP"
        except ImportError:
            self._faiss = None
            self.backend = "numpy"

    def all_scores(self, queries: np.ndarray) -> np.ndarray:
        """Cosine similarity of each query to every indexed row, in row order, shape (Q, N)."""
        q = l2_normalize(queries)
        if self._faiss is None:
            return q @ self.vectors.T
        n = self._faiss.ntotal
        sims, ids = self._faiss.search(q, n)
        dense = np.empty_like(sims)
        np.put_along_axis(dense, ids, sims, axis=1)
        return dense


class HybridRetriever:
    def __init__(self, train_uids: Sequence, image_emb: np.ndarray, text_emb: np.ndarray, alpha: float = 0.5):
        if not (len(train_uids) == len(image_emb) == len(text_emb)):
            raise ValueError("train_uids, image_emb and text_emb must align row-wise")
        self.uids = np.asarray(train_uids)
        self.alpha = alpha
        self.image_index = ExactIPIndex(image_emb)
        self.text_index = ExactIPIndex(text_emb)

    def search(self, image_queries: np.ndarray, text_queries: np.ndarray, k: int = 3,
               alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (uids, fused scores), each shape (Q, k), best first."""
        a = self.alpha if alpha is None else alpha
        fused = a * self.image_index.all_scores(image_queries) + (1 - a) * self.text_index.all_scores(text_queries)
        top = np.argsort(-fused, axis=1, kind="stable")[:, :k]
        return self.uids[top], np.take_along_axis(fused, top, axis=1)
