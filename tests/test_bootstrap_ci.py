"""
Unit tests for paired bootstrap confidence intervals.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap_ci import compare_calibration, compare_text_stages, micro_f1, paired_bootstrap


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 14)) < 0.15).astype(np.int8)
    return y, rng


def test_micro_f1_known():
    y = np.array([[1, 0], [0, 1]], dtype=np.int8)
    p = np.array([[1, 1], [0, 0]], dtype=np.int8)
    assert micro_f1(y, p) == pytest.approx(0.5)   # TP=1, FP=1, FN=1


def test_identical_stages_give_zero_delta():
    y, rng = _data()
    p = (rng.random(y.shape) < 0.15).astype(np.int8)
    r = compare_text_stages(y, p, p, n_boot=300)
    assert r["chexbert_micro_f1"]["delta"] == 0.0
    assert r["chexbert_micro_f1"]["ci95"] == [0.0, 0.0]


def test_clearly_better_stage_has_ci_above_zero():
    y, rng = _data()
    good = y.copy()
    good[rng.random(y.shape) < 0.05] ^= 1          # few errors
    bad = (rng.random(y.shape) < 0.15).astype(np.int8)  # random guessing
    r = compare_text_stages(y, good, bad, n_boot=500)
    f1 = r["chexbert_micro_f1"]
    assert f1["delta"] > 0 and f1["ci95"][0] > 0 and f1["p_value"] < 0.01
    assert r["hallucination_rate"]["ci95"][1] < 0   # the better stage hallucinates less


def test_radgraph_delta_uses_per_report_means():
    y, _ = _data(50)
    r = compare_text_stages(y, y, y, rg_a=np.full(50, 0.6), rg_b=np.full(50, 0.4), n_boot=100)
    assert r["radgraph_f1"]["delta"] == pytest.approx(0.2)


def test_calibration_comparison_rewards_honest_confidence():
    rng = np.random.default_rng(1)
    correct = rng.random((300, 14)) < 0.7
    honest = np.full(correct.shape, 0.7)   # calibrated
    overconfident = np.ones(correct.shape)
    r = compare_calibration(correct, honest, overconfident, n_boot=300)
    assert r["ece"]["delta"] < 0 and r["ece"]["ci95"][1] < 0


def test_bootstrap_is_reproducible():
    vals = np.random.default_rng(3).random(100)
    a = paired_bootstrap(lambda i: vals[i].mean(), 100, n_boot=200, seed=42)
    b = paired_bootstrap(lambda i: vals[i].mean(), 100, n_boot=200, seed=42)
    assert a == b
