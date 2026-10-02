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

from src.evaluation.nlg_metrics import copy_overlap, lexical_metrics, radgraph_f1, tokenize_report

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


def test_copy_overlap_verbatim_copy_is_one():
    src = "The heart is normal in size. No pleural effusion or pneumothorax."
    assert copy_overlap(src, ["Unrelated text about bones only.", src]) == pytest.approx(1.0)


def test_copy_overlap_partial_and_max_over_sources():
    gen = "heart is normal in size . left lower lobe opacity"   # 10 tokens -> 7 four-grams
    src_a = "the heart is normal in size ."                      # shares 3 four-grams
    src_b = "left lower lobe opacity"                            # shares 1 four-gram
    assert copy_overlap(gen, [src_b, src_a]) == pytest.approx(3 / 7)   # best single source, not the sum


def test_copy_overlap_edge_cases():
    assert copy_overlap("too short", ["too short"]) == 0.0   # fewer than 4 tokens
    assert copy_overlap("a b c d e f", []) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        lexical_metrics([REF], [REF, REF])


@pytest.mark.skipif("RADGRAPH_PYTHON" not in os.environ, reason="RadGraph venv not configured")
def test_radgraph_identical_report_scores_one():
    out = radgraph_f1(["Mild cardiomegaly. Small left pleural effusion."],
                      ["Mild cardiomegaly. Small left pleural effusion."])
    assert out["per_report"][0] == pytest.approx(1.0)
