"""
MedCPT dual encoders (NCBI) for the text channel of S1 retrieval.

Queries (the patient's indication) use the MedCPT query encoder; train-split documents
(indication + reference report) use the MedCPT article encoder, matching how MedCPT was trained.
"""

from typing import List, Sequence

import numpy as np
import torch

QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"


class MedCPTEncoder:
    def __init__(self, device: str = None, batch_size: int = 32):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.q_tok = AutoTokenizer.from_pretrained(QUERY_ENCODER)
        self.q_model = AutoModel.from_pretrained(QUERY_ENCODER).to(self.device).eval()
        self.a_tok = AutoTokenizer.from_pretrained(ARTICLE_ENCODER)
        self.a_model = AutoModel.from_pretrained(ARTICLE_ENCODER).to(self.device).eval()

    @torch.no_grad()
    def _encode(self, tok, model, texts, max_length) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            cls = model(**enc.to(self.device)).last_hidden_state[:, 0]
            chunks.append(cls.float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, 768), dtype=np.float32)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode(self.q_tok, self.q_model, [str(q) for q in queries], max_length=64)

    def encode_articles(self, titles: Sequence[str], bodies: Sequence[str]) -> np.ndarray:
        pairs: List[List[str]] = [[str(t), str(b)] for t, b in zip(titles, bodies)]
        return self._encode(self.a_tok, self.a_model, pairs, max_length=512)
