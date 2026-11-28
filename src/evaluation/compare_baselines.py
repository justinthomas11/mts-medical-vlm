"""
Paired bootstrap of S0 and S1 against the non-VLM baselines (DR-033) on one split.

delta = stage minus baseline, for CheXbert micro-F1, macro-F1 (14), hallucination rate and RadGraph F1;
undefined metrics (e.g. hallucination rate of a report that asserts no finding) are reported as missing.

Output: results/baselines/comparison_<split>.json

Usage: python src/evaluation/compare_baselines.py --split test
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import pandas as pd

from src.evaluation.bootstrap_ci import _labels, compare_text_stages, paired_bootstrap
from src.evaluation.clinical_metrics import clinical_efficacy

BASELINES = ["template", "retrieval_top1"]
STAGES = {"S0": "s0", "S1": "s1"}


def macro_f1(y_true, y_pred) -> float:
    return clinical_efficacy(y_true, y_pred)["macro_14"]["f1"]


def main():
    from src.pipeline.common import labels_for, load_configs, load_reference_labels, path

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    dcfg, ecfg = load_configs()
    seed = dcfg["project"]["seed"]
    res = path(ecfg["paths"]["results_dir"])

    def load(folder: Path, name: str):
        rep = pd.read_csv(folder / f"per_report_{name}.csv").set_index("uid")
        return rep, folder / f"chexbert_generated_{name}.csv"

    systems = {s: load(res / d, args.split) for s, d in STAGES.items()}
    systems.update({b: load(res / "baselines", f"{args.split}_{b}") for b in BASELINES})
    uids = sorted(set.intersection(*(set(rep.index) for rep, _ in systems.values())))
    if any(len(rep) != len(uids) for rep, _ in systems.values()):
        raise SystemExit("Systems were scored on different patients; paired comparison impossible.")
    y_true = labels_for(load_reference_labels(ecfg), uids)
    pred = {s: _labels(p, uids) for s, (_, p) in systems.items()}
    rg = {s: rep.loc[uids, "radgraph_f1"].to_numpy() for s, (rep, _) in systems.items() if "radgraph_f1" in rep}

    out = {"split": args.split, "n_patients": len(uids), "n_boot": args.n_boot, "seed": seed,
           "convention": "delta = stage minus baseline"}
    for s in STAGES:
        for b in BASELINES:
            cmp = compare_text_stages(y_true, pred[s], pred[b], rg.get(s), rg.get(b), args.n_boot, seed)
            cmp["chexbert_macro_f1"] = paired_bootstrap(
                lambda i: macro_f1(y_true[i], pred[s][i]) - macro_f1(y_true[i], pred[b][i]), len(uids),
                args.n_boot, seed)
            out[f"{s}_minus_{b}"] = cmp
            for metric, r in cmp.items():
                txt = ("undefined" if r["delta"] is None else
                       f"delta={r['delta']:+.4f} CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p={r['p_value']:.3f}")
                print(f"{s} - {b:15s} {metric:20s} {txt}")

    out_path = res / "baselines" / f"comparison_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
