"""
Unit tests for clinical efficacy metrics and CheXbert label binarisation.
"""

import sys
from pathlib import Path
import math

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS, raw_to_binary
from src.evaluation.clinical_metrics import clinical_efficacy, example_based_f1, hallucination_rate


def _matrix(rows):
    m = np.zeros((len(rows), len(CHEXBERT_LABELS)), dtype=np.int8)
    for i, names in enumerate(rows):
        for n in names:
            m[i, CHEXBERT_LABELS.index(n)] = 1
    return m


def test_raw_to_binary_uncertain_handling():
    raw = np.array([[0, 1, 2, 3]])
    assert raw_to_binary(raw).tolist() == [[0, 1, 0, 1]]
    assert raw_to_binary(raw, uncertain_as_positive=False).tolist() == [[0, 1, 0, 0]]


def test_perfect_agreement():
    y = _matrix([["Cardiomegaly"], ["No Finding"], ["Edema", "Pleural Effusion"]])
    res = clinical_efficacy(y, y)
    assert res["micro_14"]["f1"] == pytest.approx(1.0)
    assert res["micro_5"]["f1"] == pytest.approx(1.0)
    assert res["exact_match_14"] == pytest.approx(1.0)


def test_known_micro_scores():
    ref = _matrix([["Cardiomegaly", "Edema"], ["No Finding"]])
    hyp = _matrix([["Cardiomegaly"], ["Atelectasis"]])
    res = clinical_efficacy(ref, hyp)
    # TP=1, FP=1, FN=2 -> P=0.5, R=1/3, F1=0.4
    assert res["micro_14"]["precision"] == pytest.approx(0.5)
    assert res["micro_14"]["recall"] == pytest.approx(1 / 3)
    assert res["micro_14"]["f1"] == pytest.approx(0.4)
    assert res["per_label"]["Cardiomegaly"]["f1"] == pytest.approx(1.0)


def test_example_based_f1_empty_sets():
    empty = np.zeros((2, 14), dtype=np.int8)
    assert example_based_f1(empty, empty) == pytest.approx(1.0)


def test_hallucination_rate():
    ref = _matrix([["Cardiomegaly"], ["No Finding"]])
    hyp = _matrix([["Cardiomegaly", "Edema"], ["No Finding"]])
    assert hallucination_rate(ref, hyp) == pytest.approx(0.5)
    assert math.isnan(hallucination_rate(ref, _matrix([["No Finding"], []])))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        clinical_efficacy(np.zeros((2, 14)), np.zeros((3, 14)))
