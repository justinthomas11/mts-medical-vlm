"""
Unit tests for the val-only prompt selection scoring (DR-024).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.select_prompt import CRITERIA, format_compliant, prompt_scores


def test_format_compliance_case_insensitive():
    assert format_compliant("Findings: x\nImpression: y")
    assert not format_compliant("The heart is normal.")


def test_all_normal_predictor_is_exposed_by_abnormal_counts():
    """The round-1 failure mode: calling everything normal finds zero abnormal labels."""
    y_true = np.zeros((4, 14), dtype=np.int8)
    y_true[:2, 13] = 1          # two normal studies
    y_true[2:, 1] = 1           # two cardiomegaly studies
    y_pred = np.zeros((4, 14), dtype=np.int8)
    y_pred[:, 13] = 1           # model calls all four normal
    s = prompt_scores(y_true, y_pred, ["FINDINGS: a IMPRESSION: b"] * 4)
    assert s["reports_called_normal"] == 4
    assert s["abnormal_labels_correct"] == 0 and s["abnormal_labels_in_reference"] == 2
    assert s["chexbert_micro_f1_14"] > s["chexbert_macro_f1_14"]   # micro rewards it, macro does not
    assert s["format_compliance"] == pytest.approx(1.0)


def test_criteria_keys_exist_in_scores():
    s = prompt_scores(np.zeros((1, 14), dtype=np.int8), np.zeros((1, 14), dtype=np.int8), ["x"])
    for key in CRITERIA.values():
        assert key in s
