"""
Verifies that a (re)built data/processed/iu_xray_processed.csv has exactly the committed patient
split: counts 2,566 / 366 / 734 and the same uid in the same split as
results/labels/chexbert_reference.csv. Used before any Kaggle/Colab run, where data/processed
has to be regenerated from the raw CSVs.

Exit code 0 = identical; 1 = mismatch (stop — do not run any stage).

Usage: python src/pipeline/verify_splits.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List

import pandas as pd

EXPECTED_COUNTS = {"train": 2566, "val": 366, "test": 734}


def split_mismatches(processed: pd.DataFrame, reference: pd.DataFrame) -> List[str]:
    problems = []
    bench = processed[processed["split"].isin(EXPECTED_COUNTS)]
    counts = bench["split"].value_counts().to_dict()
    for split, n in EXPECTED_COUNTS.items():
        if counts.get(split, 0) != n:
            problems.append(f"{split}: {counts.get(split, 0)} patients, expected {n}")
    got = dict(zip(bench["uid"], bench["split"]))
    want = dict(zip(reference["uid"], reference["split"]))
    missing = set(want) - set(got)
    extra = set(got) - set(want)
    moved = [u for u in set(got) & set(want) if got[u] != want[u]]
    if missing:
        problems.append(f"{len(missing)} committed uids missing, e.g. {sorted(missing)[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected uids, e.g. {sorted(extra)[:5]}")
    if moved:
        problems.append(f"{len(moved)} uids in a different split, e.g. {sorted(moved)[:5]}")
    return problems


def main() -> int:
    from src.pipeline.common import load_configs, path

    dcfg, ecfg = load_configs()
    processed = pd.read_csv(path(dcfg["paths"]["processed_master_csv"]))
    reference = pd.read_csv(path(ecfg["paths"]["reference_labels_csv"]), usecols=["uid", "split"])
    problems = split_mismatches(processed, reference)
    if problems:
        print("SPLIT MISMATCH — stop and report:\n  " + "\n  ".join(problems))
        return 1
    print("Splits identical to the committed benchmark: " +
          ", ".join(f"{k} {v}" for k, v in EXPECTED_COUNTS.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
