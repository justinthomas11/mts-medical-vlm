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
