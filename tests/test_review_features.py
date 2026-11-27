import numpy as np
import pytest

from src.uncertainty.review_features import (LinearRiskModel, aurc, cv_aurc, retrieved_label_stats,
                                             ridge_scorer)


def test_retrieved_label_stats():
    y_pred = np.array([[1, 0], [0, 0]])
    y_retr = np.array([[[1, 0], [1, 0], [1, 0]],      # all agree, majority == prediction
                       [[1, 0], [0, 0], [1, 1]]])     # split votes on both labels
    s = retrieved_label_stats(y_pred, y_retr)
    assert s["retr_disagree"][0] == 0.0
    assert s["retr_disagree"][1] == pytest.approx((2 / 9 + 2 / 9) / 2)
    assert s["retr_mismatch"].tolist() == [0.0, 0.5]   # majority [1, 0] vs prediction [0, 0]


def test_retrieved_label_stats_shape_check():
    with pytest.raises(ValueError):
        retrieved_label_stats(np.zeros((2, 3)), np.zeros((2, 3, 4)))


def test_aurc_perfect_vs_reversed_ranking():
    risk = np.array([0.0, 0.0, 1.0, 1.0])
    good = aurc(np.array([0, 1, 2, 3]), risk)   # least uncertain are the correct ones
    bad = aurc(np.array([3, 2, 1, 0]), risk)
    assert good < risk.mean() < bad


def test_linear_risk_model_roundtrip_and_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    risk = 0.5 + 0.2 * X[:, 0]
    m = LinearRiskModel(["a", "b"], l2=0.0).fit(X, risk)
    assert np.allclose(m.score(X), risk, atol=1e-6)
    m2 = LinearRiskModel.from_dict(m.to_dict())
    assert np.allclose(m2.score(X), m.score(X))


def test_cv_aurc_is_deterministic_and_uses_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    risk = (X[:, 1] > 0).astype(float)
    a = cv_aurc(X, risk, ridge_scorer(["x", "y", "z"], 1.0), seed=42)
    assert a == cv_aurc(X, risk, ridge_scorer(["x", "y", "z"], 1.0), seed=42)
    assert a < risk.mean()
