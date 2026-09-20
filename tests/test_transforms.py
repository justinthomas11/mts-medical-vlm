"""
Unit tests for model-agnostic image transforms and PyTorch Dataset class.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import (
    load_image,
    resize_with_aspect_ratio,
    standard_resize,
    MedicalImageTransform,
    IUXRayDataset
)


@pytest.fixture
def sample_image():
    """Generates a synthetic grayscale CXR-like PIL image."""
    arr = (np.random.rand(1024, 800) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def test_resize_with_aspect_ratio(sample_image):
    target_size = (512, 512)
    resized = resize_with_aspect_ratio(sample_image, target_size=target_size, fill_color=0)
    assert resized.size == target_size
    # Check that image is not stretched: original aspect ratio is 800/1024 = 0.78125
    # Target height 512 -> scaled width should be int(800 * 512 / 1024) = 400
    # So left/right margins should have fill color 0
    arr = np.array(resized)
    assert arr[:, 0].sum() == 0  # left margin padded


def test_standard_resize(sample_image):
    target_size = (256, 256)
    resized = standard_resize(sample_image, target_size=target_size)
    assert resized.size == target_size


def test_transform_pipeline_tensor_output(sample_image):
    transform = MedicalImageTransform(target_size=(384, 384), preserve_aspect_ratio=True, to_rgb=True, normalize_imagenet=True)
    tensor = transform(sample_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 384, 384)
    assert tensor.dtype == torch.float32


def test_dataset_item_loading():
    data_path = PROJECT_ROOT / "data" / "processed" / "iu_xray_processed.csv"
    images_dir = PROJECT_ROOT / "data" / "raw" / "images" / "images_normalized"
    if not data_path.exists():
        pytest.skip("Processed dataset not yet generated.")

    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"].head(5)

    dataset = IUXRayDataset(train_df, images_dir=images_dir)
    assert len(dataset) == 5

    item = dataset[0]
    assert "uid" in item
    assert "image" in item
    assert "query" in item
    assert "target_report" in item
    assert "is_abnormal" in item
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 512, 512)
