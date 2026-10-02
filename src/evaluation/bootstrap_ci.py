"""
Paired bootstrap confidence intervals for stage-to-stage differences (S0 vs S1, S1 vs S3).

Patients are resampled with replacement (B = 2,000, seed 42); every metric is recomputed for both
stages on the same resample, giving a 95% percentile interval for the difference and a two-sided
bootstrap p-value. Inputs are the raw files saved by src/evaluation/evaluate_stage.py, so no model
is re-run.

Usage: python src/evaluation/bootstrap_ci.py --split test [--tag smoke20]
Output: results/ablation/bootstrap_<split>[_<tag>].json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Callable, Dict

import numpy as np
import pandas as pd

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
from src.evaluation.clinical_metrics import hallucination_rate
from src.uncertainty.consensus import expected_calibration_error


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = (y_true & y_pred).sum()
    denom = y_true.sum() + y_pred.sum()
    return float(2 * tp / denom) if denom else 0.0


def paired_bootstrap(metric: Callable[[np.ndarray], float], n: int, n_boot: int = 2000, seed: int = 42) -> Dict:
    """`metric(idx)` returns (value_a - value_b) on the resampled patient indices `idx`."""
    rng = np.random.default_rng(seed)
    observed = metric(np.arange(n))
    deltas = np.array([metric(rng.integers(0, n, n)) for _ in range(n_boot)])
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": float(observed), "ci95": [float(lo), float(hi)], "p_value": float(min(p, 1.0)),
            "n_boot": int(len(deltas))}


def compare_text_stages(y_true, pred_a, pred_b, rg_a=None, rg_b=None, n_boot=2000, seed=42) -> Dict:
    """Differences (a - b) in CheXbert micro-F1, hallucination rate and RadGraph F1."""
    n = len(y_true)
    out = {
        "chexbert_micro_f1": paired_bootstrap(
            lambda i: micro_f1(y_true[i], pred_a[i]) - micro_f1(y_true[i], pred_b[i]), n, n_boot, seed),
        "hallucination_rate": paired_bootstrap(
            lambda i: hallucination_rate(y_true[i], pred_a[i]) - hallucination_rate(y_true[i], pred_b[i]),
            n, n_boot, seed),
    }
    if rg_a is not None and rg_b is not None:
        out["radgraph_f1"] = paired_bootstrap(lambda i: rg_a[i].mean() - rg_b[i].mean(), n, n_boot, seed)
    return out


def compare_calibration(correct: np.ndarray, conf_a: np.ndarray, conf_b: np.ndarray, bins: int = 10,
                        n_boot: int = 2000, seed: int = 42) -> Dict:
    """ECE difference (a - b) with per-patient (N, 14) correctness and confidences."""
    return {"ece": paired_bootstrap(
        lambda i: expected_calibration_error(conf_a[i], correct[i], bins)
        - expected_calibration_error(conf_b[i], correct[i], bins), len(correct), n_boot, seed)}


def _labels_frame(df: pd.DataFrame, uids) -> np.ndarray:
    return df.set_index("uid").loc[list(uids), CHEXBERT_LABELS].to_numpy(dtype=np.int8)


def _labels(path: Path, uids) -> np.ndarray:
    return _labels_frame(pd.read_csv(path), uids)


def main():
    from src.pipeline.common import labels_for, load_configs, load_reference_labels, path

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--tag", default="")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    dcfg, ecfg = load_configs()
    seed = dcfg["project"]["seed"]
    res = path(ecfg["paths"]["results_dir"])
    suffix = f"{args.split}{'_' + args.tag if args.tag else ''}"

    s0_rep = pd.read_csv(res / "s0" / f"per_report_{suffix}.csv")
    s1_rep = pd.read_csv(res / "s1" / f"per_report_{suffix}.csv")
    uids = sorted(set(s0_rep["uid"]) & set(s1_rep["uid"]))
    if len(uids) != len(s0_rep) or len(uids) != len(s1_rep):
        raise SystemExit("S0 and S1 were scored on different patients — paired comparison impossible.")
    y_true = labels_for(load_reference_labels(ecfg), uids)
    p0 = _labels(res / "s0" / f"chexbert_generated_{suffix}.csv", uids)
    p1 = _labels(res / "s1" / f"chexbert_generated_{suffix}.csv", uids)
    rg = {}
    for name, df in (("s0", s0_rep), ("s1", s1_rep)):
        if "radgraph_f1" in df:
            rg[name] = df.set_index("uid").loc[uids, "radgraph_f1"].to_numpy()

    out = {"split": args.split, "tag": args.tag, "n_patients": len(uids), "n_boot": args.n_boot, "seed": seed,
           "convention": "delta = later stage minus earlier stage",
           "s1_minus_s0": compare_text_stages(y_true, p1, p0, rg.get("s1"), rg.get("s0"), args.n_boot, seed)}

    samples_path = res / "s1" / f"chexbert_samples_{suffix}_s3.csv"
    if samples_path.exists():
        s = pd.read_csv(samples_path)
        k = int(s["sample"].max()) + 1
        samples = np.stack([_labels_frame(s[s["sample"] == j], uids) for j in range(k)], axis=1)
        agreement = (samples == p1[:, None, :]).mean(axis=1)
        correct = (p1 == y_true)
        out["s3_minus_s1"] = compare_calibration(correct, agreement, np.ones_like(agreement),
                                                 ecfg["evaluation"]["ece_bins"], args.n_boot, seed)

    out_path = res / "ablation" / f"bootstrap_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    for comp in ("s1_minus_s0", "s3_minus_s1"):
        for metric, r in out.get(comp, {}).items():
            print(f"{comp:12s} {metric:20s} delta={r['delta']:+.4f} CI95=[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                  f"p={r['p_value']:.3f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
