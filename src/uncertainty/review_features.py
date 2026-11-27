"""
Review-flag v2 (DR-028): reference-free per-report signals and a risk model selected on val.

Every feature is available at inference time without the reference report:
  entropy          mean greedy token entropy
  disagreement     1 - mean CheXbert agreement of the 5 samples with the greedy report
  copy             4-gram copy overlap with the closest retrieved report
  n_tokens         generated length
  n_positive       number of positive CheXbert labels in the generated report (No Finding excluded)
  pred_normal      generated report labelled No Finding
  retr_score       hybrid retrieval score of the top retrieved train report
  retr_disagree    label disagreement among the 3 retrieved train reports (mean per-label variance)
  retr_mismatch    share of the 14 labels where the generated report differs from the retrieved majority

Risk = 1 - example-based F1 (as in DR-020's risk-coverage evaluation). Candidates are compared by
cross-validated AURC on val only; the winner is refit on all val and its threshold set at the review budget.
"""

from typing import Dict, List, Sequence

import numpy as np

FEATURES = ["entropy", "disagreement", "copy", "n_tokens", "n_positive", "pred_normal",
            "retr_score", "retr_disagree", "retr_mismatch"]


def retrieved_label_stats(y_pred: np.ndarray, y_retrieved: np.ndarray) -> Dict[str, np.ndarray]:
    """y_pred (N, C) and y_retrieved (N, K, C) binary -> disagreement among the K retrieved reports
    and mismatch between the generated labels and the retrieved majority vote."""
    y_pred, y_retrieved = np.asarray(y_pred, float), np.asarray(y_retrieved, float)
    if y_retrieved.ndim != 3 or y_retrieved.shape[0] != y_pred.shape[0] or y_retrieved.shape[2] != y_pred.shape[1]:
        raise ValueError(f"y_pred {y_pred.shape} vs y_retrieved {y_retrieved.shape}")
    p = y_retrieved.mean(axis=1)
    majority = (p > 0.5).astype(float)
    return {"retr_disagree": (p * (1 - p)).mean(axis=1), "retr_mismatch": (majority != y_pred).mean(axis=1)}


def aurc(uncertainty: np.ndarray, risk: np.ndarray) -> float:
    order = np.argsort(np.asarray(uncertainty, float), kind="stable")
    r = np.asarray(risk, float)[order]
    return float((np.cumsum(r) / np.arange(1, len(r) + 1)).mean())


class LinearRiskModel:
    """Ridge regression of risk on standardized features (closed form, no sklearn dependency)."""

    def __init__(self, features: Sequence[str], l2: float = 1.0):
        self.features, self.l2 = list(features), l2

    def fit(self, X: np.ndarray, risk: np.ndarray) -> "LinearRiskModel":
        X = np.asarray(X, float)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mean) / self.std]
        reg = self.l2 * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ np.asarray(risk, float))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = np.c_[np.ones(len(X)), (np.asarray(X, float) - self.mean) / self.std]
        return Z @ self.coef

    def to_dict(self) -> Dict:
        return {"type": "ridge", "features": self.features, "l2": self.l2, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "coef": self.coef.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "LinearRiskModel":
        m = cls(d["features"], d["l2"])
        m.mean, m.std, m.coef = np.asarray(d["mean"]), np.asarray(d["std"]), np.asarray(d["coef"])
        return m


def cv_aurc(X: np.ndarray, risk: np.ndarray, make_scorer, n_folds: int = 5, seed: int = 42) -> float:
    """Out-of-fold uncertainty for every val report, then one AURC over all of them."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(risk)), n_folds)
    u = np.empty(len(risk))
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        u[test] = make_scorer(X[train], risk[train])(X[test])
    return aurc(u, risk)


def single_feature_scorer(col: int, sign: float):
    return lambda Xtr, rtr: (lambda X: sign * X[:, col])


def ridge_scorer(features: List[str], l2: float):
    return lambda Xtr, rtr: LinearRiskModel(features, l2).fit(Xtr, rtr).score
