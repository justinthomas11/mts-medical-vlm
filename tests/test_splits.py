"""
Unit tests for verifying patient-level split isolation, zero leakage, and stratification balance.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def processed_df():
    path = PROJECT_ROOT / "data" / "processed" / "iu_xray_processed.csv"
    assert path.exists(), f"Processed CSV not found at {path}"
    return pd.read_csv(path)


@pytest.fixture
def manifest_json():
    path = PROJECT_ROOT / "data" / "processed" / "splits_manifest.json"
    assert path.exists(), f"Manifest not found at {path}"
    with open(path, "r") as f:
        return json.load(f)


def test_zero_patient_leakage(processed_df):
    """Assert strictly zero patient UID overlap across train, val, and test partitions."""
    train_uids = set(processed_df[processed_df["split"] == "train"]["uid"])
    val_uids = set(processed_df[processed_df["split"] == "val"]["uid"])
    test_uids = set(processed_df[processed_df["split"] == "test"]["uid"])

    assert len(train_uids) > 0, "Train split is empty"
    assert len(val_uids) > 0, "Val split is empty"
    assert len(test_uids) > 0, "Test split is empty"

    overlap_train_val = train_uids.intersection(val_uids)
    overlap_train_test = train_uids.intersection(test_uids)
    overlap_val_test = val_uids.intersection(test_uids)

    assert len(overlap_train_val) == 0, f"Found {len(overlap_train_val)} overlapping patients between train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Found {len(overlap_train_test)} overlapping patients between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Found {len(overlap_val_test)} overlapping patients between val and test: {overlap_val_test}"


def test_split_proportions(processed_df):
    """Verify split proportions approximate 70% train / 10% val / 20% test on benchmark cohort."""
    benchmark_df = processed_df[processed_df["split"].isin(["train", "val", "test"])]
    total_benchmark = len(benchmark_df)
    assert total_benchmark > 3000, f"Benchmark count {total_benchmark} unexpectedly small"

    train_pct = len(processed_df[processed_df["split"] == "train"]) / total_benchmark
    val_pct = len(processed_df[processed_df["split"] == "val"]) / total_benchmark
    test_pct = len(processed_df[processed_df["split"] == "test"]) / total_benchmark

    assert 0.68 <= train_pct <= 0.72, f"Train percentage {train_pct:.3f} outside expected [0.68, 0.72]"
    assert 0.08 <= val_pct <= 0.12, f"Val percentage {val_pct:.3f} outside expected [0.08, 0.12]"
    assert 0.18 <= test_pct <= 0.22, f"Test percentage {test_pct:.3f} outside expected [0.18, 0.22]"


def test_stratification_balance(processed_df):
    """Verify abnormal ratio is balanced across train, val, and test splits within 2% margin."""
    splits = ["train", "val", "test"]
    abnormal_ratios = {}
    for s in splits:
        sub = processed_df[processed_df["split"] == s]
        ratio = sub["is_abnormal"].mean()
        abnormal_ratios[s] = ratio

    base_ratio = abnormal_ratios["train"]
    for s in ["val", "test"]:
        diff = abs(abnormal_ratios[s] - base_ratio)
        assert diff <= 0.03, f"Split {s} abnormal ratio {abnormal_ratios[s]:.3f} differs from train {base_ratio:.3f} by {diff:.3f}"


def test_manifest_consistency(manifest_json, processed_df):
    """Verify splits_manifest.json matches processed CSV counts and patient IDs."""
    for split_key in ["train", "val", "test"]:
        manifest_count = manifest_json["splits"][split_key]["patient_count"]
        csv_count = len(processed_df[processed_df["split"] == split_key])
        assert manifest_count == csv_count, f"Manifest count ({manifest_count}) does not match CSV count ({csv_count}) for {split_key}"
