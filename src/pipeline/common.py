"""
Shared config/data helpers for the Phase II pipeline scripts.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS

BENCHMARK_SPLITS = ("train", "val", "test")


def load_configs(data_config: str = "configs/data_config.yaml",
                 exp_config: str = "configs/experiment_config.yaml") -> Tuple[Dict, Dict]:
    with open(PROJECT_ROOT / data_config) as f:
        dcfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / exp_config) as f:
        ecfg = yaml.safe_load(f)
    return dcfg, ecfg


def path(cfg_value: str) -> Path:
    return PROJECT_ROOT / cfg_value


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmark(dcfg: Dict) -> pd.DataFrame:
    """Benchmark rows (train/val/test) in a fixed order: split, then uid."""
    df = pd.read_csv(path(dcfg["paths"]["processed_master_csv"]))
    df = df[df["split"].isin(BENCHMARK_SPLITS)].copy()
    df["split"] = pd.Categorical(df["split"], BENCHMARK_SPLITS, ordered=True)
    return df.sort_values(["split", "uid"]).reset_index(drop=True).astype({"split": str})


def load_image(dcfg: Dict, filename: str) -> Image.Image:
    return Image.open(path(dcfg["paths"]["raw_images_dir"]) / filename)


def load_reference_labels(ecfg: Dict) -> pd.DataFrame:
    p = path(ecfg["paths"]["reference_labels_csv"])
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run python src/evaluation/label_references.py first")
    return pd.read_csv(p)


def labels_for(ref: pd.DataFrame, uids) -> np.ndarray:
    return ref.set_index("uid").loc[list(uids), CHEXBERT_LABELS].to_numpy(dtype=np.int8)
