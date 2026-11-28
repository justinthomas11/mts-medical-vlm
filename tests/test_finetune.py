import os

import pytest
import torch

from src.models.finetune import IGNORE_INDEX, build_examples, mask_prompt

ROWS = [{"uid": u, "clean_indication": "cough", "primary_image_filename": f"{u}.png",
         "target_report": f"report {u}"} for u in (1, 2, 3, 4)]
REPORTS = {u: f"report {u}" for u in (1, 2, 3, 4, 9)}


def test_build_examples_mixes_formats_and_never_shows_own_report():
    retrieved = {1: [2, 3, 9], 2: [1, 3, 9], 3: [1, 2, 9], 4: [1, 2, 3]}
    ex = build_examples(ROWS, REPORTS, retrieved, "baseline_v1", "query_first", rag_fraction=0.5, seed=0)
    assert [e.uid for e in ex] == [1, 2, 3, 4]
    for e in ex:
        assert e.target == f"report {e.uid}"
        if e.uses_rag:
            assert "Reference report 1" in e.prompt and f"report {e.uid}\n" not in e.prompt + "\n"
        else:
            assert "Reference report" not in e.prompt
    assert build_examples(ROWS, REPORTS, retrieved, "baseline_v1", "query_first", 0.5, 0)[2].prompt == ex[2].prompt


def test_build_examples_extremes():
    retrieved = {u: [9, 9, 9] for u in (1, 2, 3, 4)}
    assert not any(e.uses_rag for e in build_examples(ROWS, REPORTS, retrieved, "baseline_v1", "query_first", 0.0, 0))
    assert all(e.uses_rag for e in build_examples(ROWS, REPORTS, retrieved, "baseline_v1", "query_first", 1.0, 0))


def test_build_examples_rejects_self_retrieval():
    with pytest.raises(ValueError):
        build_examples(ROWS[:1], REPORTS, {1: [1, 2, 3]}, "baseline_v1", "query_first", 1.0, 0)


def test_mask_prompt():
    ids = torch.arange(10)[None]
    lab = mask_prompt(ids, 6)
    assert (lab[0, :6] == IGNORE_INDEX).all() and lab[0, 6:].tolist() == [6, 7, 8, 9]
    assert ids[0, 0] == 0   # input untouched


def test_results_dir_override(monkeypatch):
    from src.pipeline.common import load_configs
    monkeypatch.setenv("MTS_RESULTS_DIR", "results_v2")
    assert load_configs()[1]["paths"]["results_dir"] == "results_v2"
    monkeypatch.delenv("MTS_RESULTS_DIR")
    assert load_configs()[1]["paths"]["results_dir"] == "results"


def test_sampling_weights_upweight_rare_findings_only():
    import numpy as np

    from src.models.finetune import sampling_weights
    names = ["Common", "Rare", "No Finding"]
    labels = np.array([[1, 0, 0]] * 16 + [[0, 1, 0]] * 1 + [[0, 0, 1]] * 20 + [[0, 0, 0]] * 3)
    w = sampling_weights(labels, names, cap=5.0)
    assert np.allclose(w[20:], 1.0)                       # normal / unlabeled reports keep weight 1
    assert w[16] == pytest.approx(min(5.0, (20 / 1) ** 0.5))   # rare finding capped
    assert w[0] == pytest.approx((20 / 16) ** 0.5)
    both = sampling_weights(np.array([[1, 1, 0]] + labels.tolist()), names, cap=100.0)
    assert both[0] == pytest.approx((20 / 2) ** 0.5)      # rarest positive label decides


def test_selection_score_rejects_template_collapse():
    import numpy as np

    from src.models.finetune import selection_score
    names = ["A", "B", "No Finding"]
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    diverse = selection_score(y_true, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]]),
                              ["a", "b", "c", "d"], names)
    assert diverse["eligible"] and diverse["findings_micro_f1"] == pytest.approx(2 / 3)
    collapsed = selection_score(y_true, np.array([[0, 0, 1]] * 4), ["n", "n", "n", "n"], names)
    assert not collapsed["eligible"] and collapsed["score"] == float("-inf")
    assert collapsed["n_predicted_findings"] == 0
