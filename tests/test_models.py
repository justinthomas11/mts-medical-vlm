"""
Unit tests for VLM helpers that do not need the 7B checkpoint: image padding, token entropy
and prompt construction.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.llava_med import CLIP_MEAN_RGB, pad_to_square, token_entropies
from src.models.prompts import MISSING_INDICATION, PROMPT_TEMPLATES, RAG_HEADER, build_query, rag_prompt, s0_prompt


def test_pad_to_square_centres_grayscale_image():
    img = Image.new("L", (100, 60), 255)
    out = pad_to_square(img)
    assert out.size == (100, 100) and out.mode == "RGB"
    assert out.getpixel((50, 0)) == CLIP_MEAN_RGB   # padding band
    assert out.getpixel((50, 50)) == (255, 255, 255)  # original content


def test_pad_to_square_keeps_square_images():
    img = Image.new("RGB", (64, 64), (10, 20, 30))
    assert pad_to_square(img).getpixel((0, 0)) == (10, 20, 30)


def test_token_entropy_uniform_and_peaked():
    vocab = 8
    uniform = torch.zeros(1, vocab)
    peaked = torch.tensor([[100.0] + [0.0] * (vocab - 1)])
    h = token_entropies(torch.cat([uniform, peaked]))
    assert h[0].item() == pytest.approx(math.log(vocab), rel=1e-5)
    assert h[1].item() == pytest.approx(0.0, abs=1e-4)


def test_s0_prompt_is_clinical_query():
    q = "Indication: cough. Analyze this chest radiograph and provide detailed findings and diagnostic impression."
    assert s0_prompt(f"  {q} ") == q


def test_baseline_template_reproduces_original_query():
    """baseline_v1 must match the roadmap template used to build `clinical_query` in preprocessing."""
    q = build_query("Positive TB test", "baseline_v1")
    assert q == ("Indication: Positive TB test. Analyze this chest radiograph and provide detailed findings "
                 "and diagnostic impression.")


@pytest.mark.parametrize("missing", [None, "", "nan", float("nan"), "None."])
def test_build_query_handles_missing_indication(missing):
    for tid in PROMPT_TEMPLATES:
        assert MISSING_INDICATION in build_query(missing, tid)


def test_build_query_avoids_double_period():
    assert ".." not in build_query("Chest pain.", "report_v3")


def test_rag_prompt_orders_context_before_query():
    p = rag_prompt("Indication: cough.", ["Report A.", "Report B."])
    assert p.startswith(RAG_HEADER)
    assert p.index("Reference report 1:\nReport A.") < p.index("Reference report 2:\nReport B.") < p.index("Indication: cough.")
