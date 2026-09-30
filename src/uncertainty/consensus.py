"""
Two-tier uncertainty for S3 (DR-015) and the human-review flag (DR-020).

Tier 1: mean greedy token entropy (computed during generation, src/models/llava_med.py).
Tier 2: per-label CheXbert agreement of K sampled reports with the greedy report,
        Agreement_c = (1/K) * sum_k 1[y_c^(k) == y_c^(greedy)].

Review flag: each score is converted to its empirical percentile within the VAL distribution;
the combined uncertainty is the mean of the entropy percentile and the (1 - consensus)
percentile. Reports above the val (1 - budget) quantile are flagged.
"""

from typing import Dict

import numpy as np


def label_agreement(greedy: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """greedy (N, C), samples (N, K, C) binary -> per-label agreement (N, C) in [0, 1]."""
    greedy, samples = np.asarray(greedy), np.asarray(samples)
    if samples.ndim != 3 or samples.shape[0] != greedy.shape[0] or samples.shape[2] != greedy.shape[1]:
        raise ValueError(f"greedy {greedy.shape} vs samples {samples.shape}")
    return (samples == greedy[:, None, :]).mean(axis=1)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE over flattened (confidence, correctness) pairs."""
    conf = np.asarray(confidence, dtype=float).ravel()
    corr = np.asarray(correct, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(corr[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


class ReviewFlagger:
    """Percentile-combined uncertainty with a fixed review budget, fitted on val only."""

    def __init__(self, budget: float = 0.2):
        self.budget = budget

    def fit(self, val_entropy: np.ndarray, val_consensus: np.ndarray) -> "ReviewFlagger":
        self.entropy_ref = np.sort(np.asarray(val_entropy, dtype=float))
        self.disagreement_ref = np.sort(1.0 - np.asarray(val_consensus, dtype=float))
        self.threshold = float(np.quantile(self.score(val_entropy, val_consensus), 1.0 - self.budget))
        return self

    @staticmethod
    def _percentile(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, np.asarray(x, dtype=float), side="right") / len(ref)

    def score(self, entropy: np.ndarray, consensus: np.ndarray) -> np.ndarray:
        return 0.5 * (self._percentile(self.entropy_ref, entropy)
                      + self._percentile(self.disagreement_ref, 1.0 - np.asarray(consensus, dtype=float)))

    def flag(self, entropy: np.ndarray, consensus: np.ndarray) -> np.ndarray:
        return self.score(entropy, consensus) > self.threshold

    def to_dict(self) -> Dict:
        return {"budget": self.budget, "threshold": self.threshold,
                "entropy_ref": self.entropy_ref.tolist(), "disagreement_ref": self.disagreement_ref.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "ReviewFlagger":
        f = cls(d["budget"])
        f.entropy_ref = np.asarray(d["entropy_ref"])
        f.disagreement_ref = np.asarray(d["disagreement_ref"])
        f.threshold = d["threshold"]
        return f


def risk_coverage(uncertainty: np.ndarray, risk: np.ndarray) -> Dict:
    """Area under the risk-coverage curve (lower is better) when abstaining on the most uncertain first."""
    order = np.argsort(np.asarray(uncertainty, dtype=float), kind="stable")
    r = np.asarray(risk, dtype=float)[order]
    cum_risk = np.cumsum(r) / np.arange(1, len(r) + 1)
    return {"aurc": float(cum_risk.mean()), "full_coverage_risk": float(cum_risk[-1])}
