"""
Tests for the Kaggle generation notebook and the split verification it depends on.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.verify_splits import EXPECTED_COUNTS, split_mismatches

NOTEBOOK = PROJECT_ROOT / "notebooks" / "kaggle_s0_s3_generation.ipynb"


def _code():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_code_cells_compile():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))
            compile(src, "<cell>", "exec")


def test_notebook_references_existing_scripts():
    nb_text = NOTEBOOK.read_text(encoding="utf-8")
    scripts = set(re.findall(r"src/[\w/]+\.py", nb_text))
    assert scripts, "notebook should call repo scripts"
    for s in scripts:
        assert (PROJECT_ROOT / s).exists(), s


def test_notebook_runs_fp16_and_verifies_split_before_generation():
    code = _code()
    assert 'QUANT = "fp16"' in code
    assert code.index("verify_splits.py") < code.index("run_generation.py")
    assert '"--split", "test"' in code and '"--samples", "5"' in code


def _frame(assign):
    return pd.DataFrame({"uid": list(assign), "split": list(assign.values())})


def _reference():
    uid = iter(range(1, 10_000))
    return {next(uid): s for s, n in EXPECTED_COUNTS.items() for _ in range(n)}


def test_split_check_passes_identical_split():
    ref = _reference()
    assert split_mismatches(_frame(ref), _frame(ref)) == []


def test_split_check_detects_moved_patient():
    ref = _reference()
    got = dict(ref)
    a = next(u for u, s in got.items() if s == "train")
    b = next(u for u, s in got.items() if s == "test")
    got[a], got[b] = "test", "train"          # same counts, different patients
    problems = split_mismatches(_frame(got), _frame(ref))
    assert any("different split" in p for p in problems)


def test_split_check_detects_count_change_and_ignores_excluded_rows():
    ref = _reference()
    got = {u: s for u, s in ref.items() if u != 1}
    got[99_999] = "excluded_lateral_only"
    problems = split_mismatches(_frame(got), _frame(ref))
    assert any("train:" in p for p in problems) and any("missing" in p for p in problems)
    assert not any("unexpected" in p for p in problems)


def test_notebook_scores_in_order_without_retraining_the_head():
    code = _code()
    pos = lambda s: code.index(s)
    # Review flag is fitted on val before it is applied to test (DR-020).
    fit = pos('"results/s1/generations_val.jsonl", "--uncertainty", "--fit-flagger"')
    apply = pos('"results/s1/generations_test.jsonl", "--uncertainty")')
    assert pos("run_generation.py") < fit < apply
    # Localization needs features first; the ablation needs every metric file; CIs come last.
    loc_test = pos('"src/pipeline/eval_localization.py", "--split", "test"')
    assert pos("extract_features.py") < loc_test
    assert max(apply, loc_test) < pos('"src/evaluation/build_ablation.py"') < pos('"src/evaluation/bootstrap_ci.py"')
    # The Grad-CAM head is uploaded, never retrained on Kaggle (train split only, DR-016).
    assert "train_xai_head.py" not in code and "HEAD_PATH" in code
    # Scoring dependencies (incl. the RadGraph venv) are installed before generation starts.
    assert pos("RADGRAPH_PYTHON") < pos("run_generation.py")


def test_notebook_splits_generation_across_sessions():
    code = _code()
    # Generation stops at a deadline so each Kaggle session ends normally and keeps its output.
    assert '"--deadline", str(DEADLINE)' in code
    # Scoring is skipped unless all three generation files are complete.
    assert code.count("if not COMPLETE:") == 3
    assert code.index("COMPLETE = all(") < code.index('"src/evaluation/evaluate_stage.py"')


def test_run_generation_deadline():
    from src.pipeline.run_generation import past_deadline
    import time
    assert not past_deadline(None)
    assert past_deadline(time.time() - 1)
    assert not past_deadline(time.time() + 60)
