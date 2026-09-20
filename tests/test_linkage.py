"""
Unit tests for verifying image-report linkage integrity and physical file existence.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def processed_df():
    path = PROJECT_ROOT / "data" / "processed" / "iu_xray_processed.csv"
    assert path.exists(), f"Processed CSV not found at {path}"
    return pd.read_csv(path)


@pytest.fixture
def images_dir():
    path = PROJECT_ROOT / "data" / "raw" / "images" / "images_normalized"
    assert path.exists(), f"Images directory not found at {path}"
    return path


def test_primary_image_linkage(processed_df, images_dir):
    """Verify that every train, val, and test record links to an existing, readable image."""
    benchmark_df = processed_df[processed_df["split"].isin(["train", "val", "test"])]
    assert len(benchmark_df) > 0

    # Test all records for existence on disk
    missing_files = []
    for _, row in benchmark_df.iterrows():
        img_name = str(row["primary_image_filename"])
        img_path = images_dir / img_name
        if not img_path.exists():
            missing_files.append((row["uid"], img_name))

    assert len(missing_files) == 0, f"Found {len(missing_files)} missing image files: {missing_files[:10]}"


def test_image_readability_sample(processed_df, images_dir):
    """Verify a sample of linked images can be opened and verified via PIL."""
    sample_df = processed_df[processed_df["split"].isin(["train", "val", "test"])].sample(50, random_state=42)
    for _, row in sample_df.iterrows():
        img_path = images_dir / str(row["primary_image_filename"])
        with Image.open(img_path) as img:
            assert img.size[0] > 500
            assert img.size[1] > 500
            assert img.mode == "L"


def test_no_broken_secondary_links(processed_df, images_dir):
    """Verify secondary linked image files exist on disk."""
    sample_with_multiple = processed_df[processed_df["num_total_images"] > 1].sample(30, random_state=42)
    for _, row in sample_with_multiple.iterrows():
        all_imgs = str(row["all_image_filenames"]).split(";")
        for fname in all_imgs:
            if fname.strip():
                img_path = images_dir / fname.strip()
                assert img_path.exists(), f"Secondary image {fname} not found on disk for uid {row['uid']}"
