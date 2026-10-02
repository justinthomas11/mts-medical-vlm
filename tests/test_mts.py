"""
Unit tests for the Medical Trustworthiness Score (DR-021).
"""

import math
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.mts import mts, mts_components

WEIGHTS = {"diagnostic": 0.4, "reliability": 0.3, "explainability": 0.3}


def test_components_follow_dr021_definitions():
    c = mts_components(chexbert_micro_f1=0.4, radgraph_f1=0.2, ece=0.1, hallucination_rate=0.3,
                       pointing_game=0.6, smr=0.4, chance=0.2)
    assert c["diagnostic"] == pytest.approx(0.3)
    assert c["reliability"] == pytest.approx(0.8)
    # DR-026: cc(0.6) = 0.4/0.8 = 0.5, cc(0.4) = 0.2/0.8 = 0.25 -> mean 0.375
    assert c["explainability"] == pytest.approx(0.375)


def test_chance_level_heatmaps_earn_no_explainability():
    c = mts_components(0.4, 0.2, 0.1, 0.3, pointing_game=0.11, smr=0.10, chance=0.11)
    assert c["explainability"] == 0.0   # below/at chance is floored at 0


def test_chance_corrected_bounds():
    from src.evaluation.mts import chance_corrected

    assert chance_corrected(1.0, 0.3) == pytest.approx(1.0)
    assert chance_corrected(0.3, 0.3) == 0.0
    with pytest.raises(ValueError):
        chance_corrected(0.5, 1.0)


def test_explainability_requires_chance():
    with pytest.raises(ValueError):
        mts_components(0.4, 0.2, 0.1, 0.3, pointing_game=0.5, smr=0.3)


def test_explainability_zero_without_visual_explanation():
    c = mts_components(0.4, 0.2, 0.1, 0.3)
    assert c["explainability"] == 0.0


def test_mts_weighted_sum():
    c = {"diagnostic": 0.3, "reliability": 0.8, "explainability": 0.4}
    assert mts(c, WEIGHTS) == pytest.approx(0.4 * 0.3 + 0.3 * 0.8 + 0.3 * 0.4)


def test_perfect_system_scores_one():
    c = mts_components(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, chance=0.1)
    assert mts(c, WEIGHTS) == pytest.approx(1.0)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        mts({"diagnostic": 1, "reliability": 1, "explainability": 1}, {"diagnostic": 0.5, "reliability": 0.5,
                                                                      "explainability": 0.5})


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_component_raises(missing):
    with pytest.raises(ValueError):
        mts_components(0.4, 0.2, missing, 0.3)
    with pytest.raises(ValueError):
        mts_components(0.4, missing, 0.1, 0.3)
    with pytest.raises(ValueError):
        mts_components(0.4, 0.2, 0.1, 0.3, pointing_game=0.5, smr=missing, chance=0.1)


def test_config_weights_are_valid():
    """The pre-registered weights (DR-021) and sensitivity sets must each sum to 1."""
    with open(PROJECT_ROOT / "configs" / "experiment_config.yaml") as f:
        cfg = yaml.safe_load(f)["mts"]
    assert cfg["weights"] == WEIGHTS
    for name, w in cfg["sensitivity_weights"].items():
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9), name
