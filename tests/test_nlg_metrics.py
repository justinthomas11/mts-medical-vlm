"""
Unit tests for lexical (BLEU/ROUGE-L) and RadGraph report-generation metrics.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.nlg_metrics import lexical_metrics, radgraph_f1, tokenize_report

REF = "FINDINGS: The heart is normal in size. No pleural effusion.\nIMPRESSION: No acute disease."


def test_tokenize_drops_headers_and_lowercases():
    assert tokenize_report("FINDINGS: Heart normal.\nIMPRESSION: Clear.") == "heart normal . clear ."


def test_identical_reports_score_one():
    scores = lexical_metrics([REF, REF], [REF, REF])
    for key in ("bleu_1", "bleu_4", "rouge_l"):
        assert scores[key] == pytest.approx(1.0, abs=1e-6)


def test_disjoint_reports_score_low():
    scores = lexical_metrics([REF], ["Left rib fracture with pneumothorax."])
    assert scores["rouge_l"] < 0.2
    assert scores["bleu_4"] < 0.01


def test_empty_hypothesis_does_not_crash():
    scores = lexical_metrics([REF], [""])
    assert 0.0 <= scores["rouge_l"] < 0.2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        lexical_metrics([REF], [REF, REF])


@pytest.mark.skipif("RADGRAPH_PYTHON" not in os.environ, reason="RadGraph venv not configured")
def test_radgraph_identical_report_scores_one():
    out = radgraph_f1(["Mild cardiomegaly. Small left pleural effusion."],
                      ["Mild cardiomegaly. Small left pleural effusion."])
    assert out["per_report"][0] == pytest.approx(1.0)
