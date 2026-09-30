"""
Unit tests for the S2 explainability components: classification head, Grad-CAM,
derived anatomical targets and localization metrics (DR-016, DR-019).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.anatomy import CONDITION_TARGETS, derive_targets, pad_to_square_gray
from src.xai.gradcam import gradcam, upsample
from src.xai.head import PatchTokenHead, tune_thresholds
from src.xai.localization import pointing_game_hit, saliency_mass_ratio, summarize, wilson_interval

STRUCTURES = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula', 'Left Lung', 'Right Lung',
              'Left Hilus Pulmonis', 'Right Hilus Pulmonis', 'Heart', 'Aorta', 'Facies Diaphragmatica',
              'Mediastinum', 'Weasand', 'Spine']


def _toy_structures(size=100):
    s = {name: np.zeros((size, size), dtype=bool) for name in STRUCTURES}
    s["Left Lung"][10:90, 60:90] = True
    s["Right Lung"][10:90, 10:40] = True
    s["Heart"][50:80, 40:65] = True
    s["Mediastinum"][10:50, 42:58] = True
    s["Spine"][:, 48:52] = True
    return s


# ---- head ----------------------------------------------------------------

def test_head_output_shape():
    head = PatchTokenHead(in_dim=16, hidden=8, n_classes=14)
    assert head(torch.randn(3, 576, 16)).shape == (3, 14)


def test_tune_thresholds_separable_class():
    y = np.array([[1], [1], [0], [0]])
    p = np.array([[0.9], [0.8], [0.2], [0.1]])
    t = tune_thresholds(y, p)
    assert ((p[:, 0] >= t[0]) == y[:, 0].astype(bool)).all()


def test_tune_thresholds_no_positive_defaults_half():
    assert tune_thresholds(np.zeros((4, 1), dtype=int), np.random.rand(4, 1))[0] == 0.5


# ---- Grad-CAM --------------------------------------------------------------

def test_gradcam_localises_the_driving_patch():
    """A head whose class logit depends on one patch's features must light up that patch."""
    torch.manual_seed(0)
    head = PatchTokenHead(in_dim=4, hidden=4, n_classes=1, dropout=0.0)
    with torch.no_grad():
        head.token_mlp[0].weight.fill_(1.0)
        head.token_mlp[0].bias.zero_()
        head.token_mlp[1].weight.copy_(torch.eye(4))
        head.token_mlp[1].bias.zero_()
        head.classifier.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.classifier.bias.zero_()
    tokens = torch.zeros(576, 4)
    tokens[:, 1] = 1.0                         # background feature, ignored by the classifier
    tokens[5 * 24 + 7] = torch.tensor([5.0, 0.0, 0.0, 0.0])  # patch (row 5, col 7) drives the class
    cam = gradcam(head, tokens, class_idx=0)
    assert cam.shape == (24, 24)
    assert np.unravel_index(np.argmax(cam), cam.shape) == (5, 7)
    assert cam.max() == pytest.approx(1.0)


def test_upsample_shape():
    assert upsample(np.random.rand(24, 24), 512).shape == (512, 512)


# ---- anatomy ---------------------------------------------------------------

def test_pad_to_square_gray():
    from PIL import Image

    out = pad_to_square_gray(Image.new("L", (80, 40), 200))
    assert out.size == (80, 80)
    assert out.getpixel((40, 0)) == 0 and out.getpixel((40, 40)) == 200


def test_derived_targets_cover_all_conditions():
    targets = derive_targets(_toy_structures())
    assert set(CONDITION_TARGETS.values()) <= set(targets)


def test_lower_zone_is_bottom_of_lungs():
    radius = max(1, int(0.06 * 100))
    lower = derive_targets(_toy_structures(), rim_fraction=0.06)["costophrenic_lower_zone"]
    rows = np.where(lower.any(axis=1))[0]
    assert rows.min() >= 10 + int(2 / 3 * 80) - radius   # bottom third of the lung, widened by the rim width
    assert not lower[20, 20]                               # upper lung excluded
    assert lower[89 + radius - 1, 20]                      # reaches just below the lung base (costophrenic angle)


def test_effusion_target_excludes_subdiaphragmatic_region():
    """DR-023: the Facies Diaphragmatica mask (abdomen below the lungs) must not be part of the target."""
    s = _toy_structures()
    s["Facies Diaphragmatica"][95:100, :] = True
    lower = derive_targets(s)["costophrenic_lower_zone"]
    assert not lower[99, 50]


def test_pleural_rim_excludes_lung_core_but_includes_apex():
    rim = derive_targets(_toy_structures(), rim_fraction=0.03)["pleural_rim"]
    assert rim[50, 10]       # lateral edge of right lung
    assert not rim[50, 25]   # centre of right lung
    assert rim[12, 25]       # apex


# ---- localization metrics ----------------------------------------------------

def test_pointing_game_and_smr():
    heat = np.zeros((10, 10))
    heat[2, 3] = 1.0
    heat[8, 8] = 0.5
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 3] = True
    assert pointing_game_hit(heat, mask)
    assert saliency_mass_ratio(heat, mask) == pytest.approx(1.0 / 1.5)


def test_zero_heatmap_is_a_miss():
    mask = np.ones((5, 5), dtype=bool)
    assert not pointing_game_hit(np.zeros((5, 5)), mask)
    assert saliency_mass_ratio(np.zeros((5, 5)), mask) == 0.0


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 < lo < 0.8 < hi < 1.0


def test_summarize_groups_by_condition():
    recs = [{"condition": "Cardiomegaly", "hit": True, "smr": 0.6, "mask_area_fraction": 0.1},
            {"condition": "Cardiomegaly", "hit": False, "smr": 0.2, "mask_area_fraction": 0.1},
            {"condition": "Fracture", "hit": True, "smr": 0.5, "mask_area_fraction": 0.2}]
    s = summarize(recs)
    assert s["overall"]["n"] == 3 and s["overall"]["pointing_game"] == pytest.approx(2 / 3)
    assert s["Cardiomegaly"]["smr"] == pytest.approx(0.4)
