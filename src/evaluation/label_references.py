"""
Labels every benchmark reference report (train/val/test) with CheXbert once and caches
the 14-observation binary vectors, so later stages never re-label reference text.

Usage: python src/evaluation/label_references.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd
import yaml

from src.evaluation.chexbert_scorer import CheXbertLabeler, CHEXBERT_LABELS

BENCHMARK_SPLITS = ("train", "val", "test")


def main(data_config: str = "configs/data_config.yaml",
         exp_config: str = "configs/experiment_config.yaml") -> pd.DataFrame:
    with open(PROJECT_ROOT / data_config) as f:
        dcfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / exp_config) as f:
        ecfg = yaml.safe_load(f)

    df = pd.read_csv(PROJECT_ROOT / dcfg["paths"]["processed_master_csv"])
    df = df[df["split"].isin(BENCHMARK_SPLITS)].reset_index(drop=True)
    print(f"Labeling {len(df)} benchmark reference reports with CheXbert...")

    labeler = CheXbertLabeler(batch_size=ecfg["evaluation"]["batch_size"])
    labels = labeler.label_frame(df["target_report"].fillna("").tolist(), ids=df["uid"])
    labels.insert(1, "split", df["split"].values)

    out_path = PROJECT_ROOT / ecfg["paths"]["reference_labels_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)

    prevalence = labels.groupby("split")[CHEXBERT_LABELS].mean().round(4)
    summary = {
        "labeler": f"{ecfg['evaluation']['chexbert_repo']}/{ecfg['evaluation']['chexbert_file']}",
        "uncertain_as_positive": True,
        "n_reports": {s: int((labels["split"] == s).sum()) for s in BENCHMARK_SPLITS},
        "prevalence": prevalence.to_dict(orient="index"),
    }
    with open(out_path.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(prevalence.T.to_string())
    print(f"Saved {out_path}")
    return labels


if __name__ == "__main__":
    main()
