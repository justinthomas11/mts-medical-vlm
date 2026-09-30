"""
Unit tests for the S0–S3 ablation table builder, using synthetic metric dicts.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.build_ablation import build_table, to_markdown

WEIGHTS = {"primary": {"diagnostic": 0.4, "reliability": 0.3, "explainability": 0.3},
           "equal": {"diagnostic": 1 / 3, "reliability": 1 / 3, "explainability": 1 / 3}}


def _metrics(f1, rg, ece, hall, flag=None):
    m = {"n_reports": 10, "lexical": {"bleu_1": 0.3, "bleu_4": 0.05, "rouge_l": 0.2},
         "clinical": {"micro_14": {"f1": f1}, "macro_14": {"f1": f1 / 2}, "micro_5": {"f1": f1}},
         "radgraph": {"radgraph_f1": rg}, "ece": ece, "hallucination_rate": hall}
    if flag is not None:
        m["review_flag"] = {"flag_rate": flag, "micro_f1_unflagged": f1 + 0.05, "aurc": 0.4}
    return m


LOC = {"all_positive_pairs": {"overall": {"pointing_game": 0.2, "smr": 0.1}}}


def test_four_stages_and_composition():
    t = build_table({"s0": _metrics(0.2, 0.1, 0.2, 0.8), "s1": _metrics(0.3, 0.2, 0.15, 0.6),
                     "s3": _metrics(0.3, 0.2, 0.05, 0.6, flag=0.2)}, LOC, WEIGHTS).set_index("stage")
    assert list(t.index) == ["S0", "S1 (+RAG)", "S2 (+Grad-CAM)", "S3 (+Uncertainty)"]
    # S0/S1 have no explanation; S2/S3 share the Grad-CAM explainability
    assert t.loc["S0", "explainability"] == 0 and t.loc["S1 (+RAG)", "explainability"] == 0
    assert t.loc["S2 (+Grad-CAM)", "explainability"] == pytest.approx(0.15)
    # S2 reuses S1 text metrics exactly
    assert t.loc["S2 (+Grad-CAM)", "chexbert_micro_f1"] == t.loc["S1 (+RAG)", "chexbert_micro_f1"]
    # S3 differs only through calibration (ECE) and the flag
    assert t.loc["S3 (+Uncertainty)", "reliability"] > t.loc["S2 (+Grad-CAM)", "reliability"]
    assert t.loc["S3 (+Uncertainty)", "review_flag_rate"] == 0.2


def test_mts_values_match_formula():
    t = build_table({"s0": _metrics(0.2, 0.1, 0.2, 0.8), "s1": _metrics(0.3, 0.2, 0.15, 0.6)}, None, WEIGHTS)
    s0 = t.iloc[0]
    d, r = (0.2 + 0.1) / 2, ((1 - 0.2) + (1 - 0.8)) / 2
    assert s0["mts_primary"] == pytest.approx(0.4 * d + 0.3 * r)
    assert s0["mts_equal"] == pytest.approx((d + r) / 3)
    assert len(t) == 2   # without localization only S0 and S1 can be built


def test_markdown_renders_missing_as_dash():
    import pandas as pd

    md = to_markdown(pd.DataFrame({"stage": ["S0"], "pg": [None], "f1": [0.12345]}))
    assert md.splitlines()[0] == "| stage | pg | f1 |"
    assert md.splitlines()[2] == "| S0 | — | 0.1235 |"


def test_missing_metric_is_an_error_not_a_default():
    bad = _metrics(0.2, 0.1, 0.2, 0.8)
    bad["radgraph"]["radgraph_f1"] = float("nan")
    with pytest.raises(ValueError):
        build_table({"s0": bad, "s1": _metrics(0.3, 0.2, 0.15, 0.6)}, None, WEIGHTS)
