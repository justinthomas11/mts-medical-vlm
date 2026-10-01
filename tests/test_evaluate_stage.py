"""
Unit tests for stage scoring with a fake CheXbert labeler (no model download).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
from src.evaluation.evaluate_stage import flagger_filename, load_generations, score_generations
from src.uncertainty.consensus import ReviewFlagger

CARDIO, NOFIND = CHEXBERT_LABELS.index("Cardiomegaly"), CHEXBERT_LABELS.index("No Finding")


class KeywordLabeler:
    """Marks Cardiomegaly if 'enlarged' appears, otherwise No Finding."""

    def label(self, texts):
        out = np.zeros((len(texts), 14), dtype=np.int8)
        for i, t in enumerate(texts):
            out[i, CARDIO if "enlarged" in t else NOFIND] = 1
        return out


def _truth(kinds):
    y = np.zeros((len(kinds), 14), dtype=np.int8)
    for i, k in enumerate(kinds):
        y[i, CARDIO if k == "c" else NOFIND] = 1
    return y


def test_no_samples_scores_confidence_one():
    recs = [{"uid": 1, "generated_report": "heart enlarged"}, {"uid": 2, "generated_report": "normal"},
            {"uid": 3, "generated_report": "heart enlarged"}]
    res = score_generations(recs, _truth("cnn"), KeywordLabeler())
    assert res["clinical"]["micro_14"]["f1"] == pytest.approx(2 * 2 / (3 + 3))  # TP=2, FP=1, FN=1
    assert res["hallucination_rate"] == pytest.approx(0.5)                      # 1 of 2 cardiomegaly calls wrong
    # 3 reports x 14 labels, 2 wrong labels -> accuracy 40/42 -> ECE = 2/42 at confidence 1
    assert res["ece"] == pytest.approx(2 / 42)
    assert "review_flag" not in res


def _uncertain_records():
    recs = []
    for uid in range(20):
        greedy = "heart enlarged" if uid % 2 else "normal"
        stable = uid < 10
        samples = [greedy] * 5 if stable else [greedy, "normal", "heart enlarged", "normal", "heart enlarged"]
        # Entropy is continuous in real generations, so unstable reports get distinct values.
        recs.append({"uid": uid, "generated_report": greedy, "samples": samples,
                     "mean_token_entropy": 0.2 if stable else 1.0 + 0.01 * uid})
    return recs


def test_uncertainty_fit_flags_unstable_reports():
    recs = _uncertain_records()
    truth = _truth(["c" if u % 2 else "n" for u in range(20)])
    res = score_generations(recs, truth, KeywordLabeler(), use_uncertainty=True, fit_flagger_budget=0.2)
    flagged = res["per_report"]["flagged"].to_numpy()
    assert flagged[:10].sum() == 0                     # stable, low-entropy reports are never flagged
    assert 0 < res["review_flag"]["flag_rate"] <= 0.2
    assert res["y_samples"].shape == (20, 5, 14)
    assert res["per_report"]["consensus"].iloc[0] == pytest.approx(1.0)


def test_flag_budget_is_an_upper_bound_under_exact_ties():
    """With exact score ties at the threshold, nothing above the budget is flagged (strict '>')."""
    ent = np.array([0.2] * 10 + [1.0] * 10)
    cons = np.array([1.0] * 10 + [0.5] * 10)
    f = ReviewFlagger(0.2).fit(ent, cons)
    assert f.flag(ent, cons).mean() <= 0.2


def test_applying_a_val_flagger_does_not_refit():
    recs = _uncertain_records()
    truth = _truth(["c" if u % 2 else "n" for u in range(20)])
    fixed = ReviewFlagger(0.2).fit(np.array([0.1, 0.2, 0.3, 5.0]), np.array([1.0, 1.0, 1.0, 0.0]))
    res = score_generations(recs, truth, KeywordLabeler(), use_uncertainty=True, flagger=fixed)
    assert res["review_flag"]["threshold"] == fixed.threshold
    assert "fitted_flagger" not in res


def test_flagger_file_is_separate_for_tagged_runs():
    """A flag fitted on val_smoke20 must be applied only to test_smoke20, never to the full test run."""
    assert flagger_filename("generations_val", "val") == "review_flagger.json"
    assert flagger_filename("generations_test", "test") == "review_flagger.json"
    assert flagger_filename("generations_val_smoke20", "val") == "review_flagger_smoke20.json"
    assert flagger_filename("generations_test_smoke20", "test") == "review_flagger_smoke20.json"


def test_load_generations_rejects_duplicate_uids(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"uid": 1, "generated_report": "a"}\n{"uid": 1, "generated_report": "b"}\n')
    with pytest.raises(ValueError):
        load_generations(p)
