"""
Non-VLM reference baselines (DR-033), written in the generations format so evaluate_stage.py scores them
exactly like S0-S3. Nothing here is tuned; both baselines use the TRAIN split only.

  template       : every patient gets the most frequent TRAIN reference report whose CheXbert reference
                   label is "No Finding" only (a constant normal report). Shows what a model scores by
                   writing one normal report for everyone. (The most frequent train report overall is
                   labelled Cardiomegaly by CheXbert; it is kept as `template_any`, DR-033.)
  retrieval_top1 : every patient gets the TRAIN report of its top-1 S1 neighbour (same hybrid index and
                   alpha as S1), with no VLM. Shows how much of S1 the retriever alone explains.

Outputs: results/baselines/generations_<split>_<name>.jsonl

Usage:
  python src/pipeline/run_baselines.py --split test
  python src/evaluation/evaluate_stage.py results/baselines/generations_test_template.jsonl
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from collections import Counter
from typing import Dict, List

import pandas as pd


def most_frequent_report(train_reports: List[str]) -> str:
    """Most frequent report text; ties broken by the lexicographically smallest text (deterministic)."""
    counts = Counter(str(r).strip() for r in train_reports)
    top = max(counts.values())
    return min(r for r, c in counts.items() if c == top)


def baseline_records(split: str, uids: List[int], reports: Dict[int, str], retrieval: pd.DataFrame,
                     template: str, k: int, template_any: str = None) -> Dict[str, List[Dict]]:
    retr = retrieval.set_index("uid")
    out = {"template": [], "retrieval_top1": []}
    if template_any is not None:
        out["template_any"] = []
    for u in uids:
        ctx = [int(retr.loc[u, f"retrieved_uid_{j + 1}"]) for j in range(k)]
        base = {"uid": int(u), "split": split, "prompt": "", "n_tokens": None, "mean_token_entropy": None,
                "latency_s": 0.0, "samples": [], "samples_latency_s": 0.0}
        out["template"].append({**base, "stage": "baseline_template", "retrieved_uids": [],
                                "generated_report": template})
        if template_any is not None:
            out["template_any"].append({**base, "stage": "baseline_template_any", "retrieved_uids": [],
                                        "generated_report": template_any})
        out["retrieval_top1"].append({**base, "stage": "baseline_retrieval_top1", "retrieved_uids": ctx,
                                      "generated_report": str(reports[ctx[0]]).strip()})
    return out


def main():
    from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
    from src.pipeline.common import load_benchmark, load_configs, load_reference_labels, path

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    dcfg, ecfg = load_configs()
    df = load_benchmark(dcfg)
    reports = df.set_index("uid")["target_report"].fillna("").to_dict()
    train_uids = set(df.loc[df["split"] == "train", "uid"])
    train = df.loc[df["split"] == "train", ["uid", "target_report"]]
    template_any = most_frequent_report(train["target_report"].fillna("").tolist())
    ref = load_reference_labels(ecfg).set_index("uid")
    findings = [c for c in CHEXBERT_LABELS if c != "No Finding"]
    normal_only = ref.index[(ref["No Finding"] == 1) & (ref[findings].sum(axis=1) == 0)]
    template = most_frequent_report(train.loc[train["uid"].isin(normal_only), "target_report"].fillna("").tolist())
    res = path(ecfg["paths"]["results_dir"])
    retrieval = pd.read_csv(res / "s1" / f"retrieval_{args.split}.csv")
    k = ecfg["rag"]["k"]
    assert set(retrieval[[f"retrieved_uid_{j + 1}" for j in range(k)]].to_numpy().ravel()) <= train_uids
    uids = df.loc[df["split"] == args.split, "uid"].astype(int).tolist()

    out_dir = res / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in baseline_records(args.split, uids, reports, retrieval, template, k, template_any).items():
        p = out_dir / f"generations_{args.split}_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{p}: {len(recs)} reports")
    print("template:", template.replace("\n", " ")[:160])


if __name__ == "__main__":
    main()
