"""
Unit tests for S3 uncertainty: sample agreement (DR-015), label-level ECE and the
percentile-fused review flag (DR-020), and risk-coverage.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.consensus import (ReviewFlagger, expected_calibration_error, label_agreement,
                                       risk_coverage)


# ---- agreement ------------------------------------------------------------------

def test_label_agreement_matches_dr015_formula():
    greedy = np.array([[1, 0, 1]])
    samples = np.array([[[1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 1]]])  # K = 5
    assert label_agreement(greedy, samples).tolist() == [[0.8, 0.8, 0.8]]


def test_label_agreement_full_consensus():
    greedy = np.array([[0, 1]])
    samples = np.repeat(greedy[:, None, :], 5, axis=1)
    assert np.all(label_agreement(greedy, samples) == 1.0)


def test_label_agreement_shape_mismatch_raises():
    with pytest.raises(ValueError):
        label_agreement(np.zeros((2, 14)), np.zeros((2, 5, 13)))


# ---- ECE --------------------------------------------------------------------------

def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_ece_overconfident_constant_one_equals_error_rate():
    """DR-020: stages without confidence are scored at 1.0, so ECE = 1 - accuracy."""
    correct = np.array([1, 1, 0, 1, 0])
    assert expected_calibration_error(np.ones(5), correct) == pytest.approx(0.4)


def test_ece_uses_bins_weighted_by_size():
    conf = np.array([0.95, 0.95, 0.15, 0.15])
    correct = np.array([1, 1, 0, 0])
    # bin 0.9-1.0: |1 - 0.95| = 0.05; bin 0.1-0.2: |0 - 0.15| = 0.15; each weight 0.5
    assert expected_calibration_error(conf, correct) == pytest.approx(0.1)


# ---- review flag --------------------------------------------------------------------

def _val_scores(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 0.3, n), rng.uniform(0.5, 1.0, n)


def test_flag_rate_on_val_matches_budget():
    ent, cons = _val_scores(200)
    flagger = ReviewFlagger(budget=0.2).fit(ent, cons)
    rate = flagger.flag(ent, cons).mean()
    assert 0.15 <= rate <= 0.2


def test_flag_is_monotonic_in_both_signals():
    ent, cons = _val_scores()
    f = ReviewFlagger(0.2).fit(ent, cons)
    base = f.score(np.array([0.5]), np.array([0.8]))
    assert f.score(np.array([1.5]), np.array([0.8])) >= base      # more entropy -> more uncertain
    assert f.score(np.array([0.5]), np.array([0.6])) >= base      # less consensus -> more uncertain


def test_extremes_are_flagged_and_confident_is_not():
    ent, cons = _val_scores()
    f = ReviewFlagger(0.2).fit(ent, cons)
    assert f.flag(np.array([ent.max() + 1]), np.array([0.0]))[0]
    assert not f.flag(np.array([0.0]), np.array([1.0]))[0]


def test_flagger_roundtrip_serialisation():
    ent, cons = _val_scores()
    f = ReviewFlagger(0.2).fit(ent, cons)
    g = ReviewFlagger.from_dict(f.to_dict())
    assert np.array_equal(f.flag(ent, cons), g.flag(ent, cons))


# ---- risk-coverage -------------------------------------------------------------------

def test_risk_coverage_informative_beats_uninformative():
    risk = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    good = risk_coverage(uncertainty=np.array([0.1, 0.2, 0.3, 0.8, 0.9]), risk=risk)
    bad = risk_coverage(uncertainty=np.array([0.9, 0.8, 0.7, 0.1, 0.2]), risk=risk)
    assert good["aurc"] < bad["aurc"]
    assert good["full_coverage_risk"] == pytest.approx(0.4) == bad["full_coverage_risk"]
