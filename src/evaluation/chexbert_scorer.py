"""
Symmetric CheXbert labeling of reference and generated reports (DR-013, DR-017).

Wraps the Stanford AIMI CheXbert checkpoint (`chexbert.pth` from the
`StanfordAIMI/RRG_scorers` Hugging Face repo) and returns the native 14-observation
CheXpert label vectors. The same labeler is applied to reference and generated text.

Raw CheXbert classes per head: 0 = blank, 1 = positive, 2 = negative, 3 = uncertain.
Binarisation follows the report-generation convention (positive or uncertain -> 1),
which is what `f1chexbert` / RRG benchmarks report.
"""

import sys
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

CHEXBERT_LABELS: List[str] = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
CHEXPERT_5: List[str] = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

CHEXBERT_REPO = "StanfordAIMI/RRG_scorers"
CHEXBERT_FILE = "chexbert.pth"
MAX_TOKENS = 512


def raw_to_binary(raw: np.ndarray, uncertain_as_positive: bool = True) -> np.ndarray:
    """Maps raw CheXbert classes (0 blank, 1 pos, 2 neg, 3 uncertain) to {0, 1}."""
    raw = np.asarray(raw)
    positive = raw == 1
    if uncertain_as_positive:
        positive |= raw == 3
    return positive.astype(np.int8)


class CheXbertLabeler:
    """Batched CheXbert inference over report strings."""

    def __init__(self, device: str = None, batch_size: int = 32):
        from f1chexbert.f1chexbert import bert_labeler
        from huggingface_hub import hf_hub_download
        from transformers import BertTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = bert_labeler(inference=True)

        checkpoint = hf_hub_download(repo_id=CHEXBERT_REPO, filename=CHEXBERT_FILE)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # Only BERT's position_ids buffer may legitimately differ across transformers versions.
        missing = [k for k in missing if not k.endswith("position_ids")]
        unexpected = [k for k in unexpected if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"CheXbert checkpoint mismatch: missing={missing} unexpected={unexpected}")

        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _encode(self, report: str) -> List[int]:
        text = " ".join(str(report).replace("\n", " ").split())
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        # Same ids as the original CheXbert `encode_plus(tokens)` ([CLS] ... [SEP]); encode_plus is gone in transformers 5.
        ids = ([self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tokens)
               + [self.tokenizer.sep_token_id])
        if len(ids) > MAX_TOKENS:
            ids = ids[: MAX_TOKENS - 1] + [self.tokenizer.sep_token_id]
        return ids

    @torch.no_grad()
    def label_raw(self, reports: Sequence[str]) -> np.ndarray:
        """Returns raw CheXbert classes, shape (N, 14), dtype int8."""
        encoded = [self._encode(r) for r in reports]
        out = np.zeros((len(encoded), len(CHEXBERT_LABELS)), dtype=np.int8)
        for start in range(0, len(encoded), self.batch_size):
            batch = encoded[start: start + self.batch_size]
            max_len = max(len(ids) for ids in batch)
            input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            attention = torch.zeros((len(batch), max_len), dtype=torch.float)
            for i, ids in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids)
                attention[i, : len(ids)] = 1.0
            logits = self.model(input_ids.to(self.device), attention.to(self.device))
            preds = torch.stack([head.argmax(dim=1) for head in logits], dim=1)
            out[start: start + len(batch)] = preds.cpu().numpy().astype(np.int8)
        return out

    def label(self, reports: Sequence[str], uncertain_as_positive: bool = True) -> np.ndarray:
        """Returns binary 14-observation label vectors, shape (N, 14)."""
        return raw_to_binary(self.label_raw(reports), uncertain_as_positive)

    def label_frame(self, reports: Sequence[str], ids: Sequence = None) -> pd.DataFrame:
        """Binary labels as a DataFrame with CheXbert column names (plus `uid` if given)."""
        df = pd.DataFrame(self.label(reports), columns=CHEXBERT_LABELS)
        if ids is not None:
            df.insert(0, "uid", list(ids))
        return df
