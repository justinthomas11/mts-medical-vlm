"""
Tests for the deterministic (non-cherry-picked) Grad-CAM example selection.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.plot_gradcam_examples import select_examples


def test_selects_lowest_uid_hit_and_miss_per_condition():
    loc = pd.DataFrame({
        "uid": [9, 3, 5, 7, 2, 4],
        "condition": ["Cardiomegaly"] * 4 + ["Pneumothorax"] * 2,
        "hit": [True, False, True, False, False, False],
    })
    picks = select_examples(loc)
    assert picks == [("Cardiomegaly", "hit", 5), ("Cardiomegaly", "miss", 3)]   # Pneumothorax: no hit -> skipped


def test_selection_is_order_independent():
    loc = pd.DataFrame({"uid": [1, 2, 3, 4], "condition": ["Fracture"] * 4, "hit": [False, True, True, False]})
    assert select_examples(loc) == select_examples(loc.sample(frac=1, random_state=0))
