"""
Clinical efficacy metrics between CheXbert label vectors of reference and generated reports.

All inputs are binary matrices of shape (N, 14) in CHEXBERT_LABELS column order.
"""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS, CHEXPERT_5


def _prf(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def example_based_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-report F1 over positive labels; two empty label sets count as F1 = 1."""
    tp = (y_true & y_pred).sum(axis=1)
    denom = y_true.sum(axis=1) + y_pred.sum(axis=1)
    f1 = np.where(denom == 0, 1.0, 2 * tp / np.maximum(denom, 1))
    return float(f1.mean())


def clinical_efficacy(y_true: np.ndarray, y_pred: np.ndarray,
                      labels: List[str] = CHEXBERT_LABELS) -> Dict:
    """Micro/macro P/R/F1 over all 14 observations and the CheXpert-5 subset, plus per-label scores."""
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = np.asarray(y_pred).astype(np.int8)
    if y_true.shape != y_pred.shape or y_true.shape[1] != len(labels):
        raise ValueError(f"Shape mismatch: ref {y_true.shape}, hyp {y_pred.shape}, labels {len(labels)}")

    idx5 = [labels.index(name) for name in CHEXPERT_5]
    result = {
        "n_reports": int(len(y_true)),
        "micro_14": _prf(y_true, y_pred, "micro"),
        "macro_14": _prf(y_true, y_pred, "macro"),
        "micro_5": _prf(y_true[:, idx5], y_pred[:, idx5], "micro"),
        "macro_5": _prf(y_true[:, idx5], y_pred[:, idx5], "macro"),
        "example_f1_14": example_based_f1(y_true, y_pred),
        "exact_match_14": float((y_true == y_pred).all(axis=1).mean()),
        "per_label": {},
    }
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(labels):
        result["per_label"][name] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(s[i]), "predicted_positive": int(y_pred[:, i].sum()),
        }
    return result


def hallucination_rate(y_true: np.ndarray, y_pred: np.ndarray, exclude: List[str] = ("No Finding",),
                       labels: List[str] = CHEXBERT_LABELS) -> float:
    """Fraction of generated positive findings absent from the reference (pooled FP / predicted positives).

    'No Finding' is excluded because asserting normality is not a hallucinated finding.
    Returns NaN when the model asserts no findings at all.
    """
    keep = [i for i, name in enumerate(labels) if name not in exclude]
    t = np.asarray(y_true)[:, keep].astype(bool)
    p = np.asarray(y_pred)[:, keep].astype(bool)
    n_pred = p.sum()
    return float((p & ~t).sum() / n_pred) if n_pred else float("nan")
