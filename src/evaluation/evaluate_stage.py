"""
Scores one generations file (from src/pipeline/run_generation.py) for the S0–S3 table.

Metrics: CheXbert clinical efficacy (DR-013/017), BLEU-1..4 / ROUGE-L, RadGraph F1,
hallucination rate, label-level ECE (DR-020), latency; and, when sampled reports are present
and --uncertainty is given, S3 consensus, the review flag, selective F1 and AURC.

The review flag is FITTED on val (--fit-flagger, saved to results/s3/review_flagger.json) and only
APPLIED on test. Raw outputs are saved so metrics can be recomputed without re-running models:
  <out>/chexbert_generated_<name>.csv, <out>/chexbert_samples_<name>.csv (if samples),
  <out>/per_report_<name>.csv, <out>/metrics_<name>.json

Usage:
  python src/evaluation/evaluate_stage.py results/s0/generations_test.jsonl
  python src/evaluation/evaluate_stage.py results/s1/generations_val.jsonl --uncertainty --fit-flagger
  python src/evaluation/evaluate_stage.py results/s1/generations_test.jsonl --uncertainty
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
from src.evaluation.clinical_metrics import clinical_efficacy, example_based_f1, hallucination_rate
from src.uncertainty.consensus import ReviewFlagger, expected_calibration_error, label_agreement, risk_coverage


def load_generations(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    uids = [r["uid"] for r in records]
    if len(uids) != len(set(uids)):
        raise ValueError(f"{path} has duplicate uids")
    return records


def flagger_filename(generations_stem: str, split: str) -> str:
    """review_flagger.json for full runs; review_flagger_<tag>.json for tagged (e.g. smoke) runs, so a
    flag fitted on a smoke subset can never be applied to the full test run."""
    tag = generations_stem.replace("generations_", "", 1)
    tag = tag[len(split):].lstrip("_") if tag.startswith(split) else tag
    return f"review_flagger_{tag}.json" if tag else "review_flagger.json"


def per_report_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array([example_based_f1(y_true[i:i + 1], y_pred[i:i + 1]) for i in range(len(y_true))])


def score_generations(records: List[Dict], y_true: np.ndarray, labeler, ece_bins: int = 10,
                      use_uncertainty: bool = False, flagger: Optional[ReviewFlagger] = None,
                      fit_flagger_budget: Optional[float] = None,
                      retrieved_texts: Optional[List[List[str]]] = None) -> Dict:
    """Core scoring (no file I/O, no RadGraph). `labeler.label(texts)` -> (N, 14) binary array.

    `retrieved_texts[i]` (S1+ only) are the reports shown to the model for record i; used to measure
    how much of the output is copied verbatim from them (4-gram copy overlap)."""
    from src.evaluation.nlg_metrics import copy_overlap

    hyps = [r["generated_report"] for r in records]
    y_pred = np.asarray(labeler.label(hyps), dtype=np.int8)
    f1_each = per_report_f1(y_true, y_pred)

    out = {"n_reports": len(records),
           # Empty reports are scored as-is (CheXbert gives an empty label vector) and reported (DR-025).
           "empty_report_rate": float(np.mean([not h.strip() for h in hyps])),
           "clinical": clinical_efficacy(y_true, y_pred),
           "hallucination_rate": hallucination_rate(y_true, y_pred),
           "y_pred": y_pred, "per_report": pd.DataFrame({"uid": [r["uid"] for r in records], "example_f1": f1_each})}
    if "latency_s" in records[0]:
        out["latency_s_mean"] = float(np.mean([r["latency_s"] for r in records]))
    if retrieved_texts is not None:
        overlap = np.array([copy_overlap(h, src) for h, src in zip(hyps, retrieved_texts)])
        out["per_report"]["copy_overlap_4gram"] = overlap
        out["copy_overlap"] = {"mean_4gram": float(overlap.mean()),
                               "share_over_half_copied": float((overlap > 0.5).mean())}

    correct = (y_pred == y_true)
    has_samples = use_uncertainty and all(r.get("samples") for r in records)
    if not has_samples:
        # DR-020: a stage that states no confidence is scored as asserting every label with confidence 1.
        out["ece"] = expected_calibration_error(np.ones_like(correct, dtype=float), correct, ece_bins)
        out["confidence_source"] = "none (confidence = 1.0)"
        return out

    k = len(records[0]["samples"])
    flat = [s for r in records for s in r["samples"]]
    y_samples = np.asarray(labeler.label(flat), dtype=np.int8).reshape(len(records), k, len(CHEXBERT_LABELS))
    agreement = label_agreement(y_pred, y_samples)
    consensus = agreement.mean(axis=1)
    entropy = np.array([r["mean_token_entropy"] for r in records], dtype=float)

    out["y_samples"] = y_samples
    out["ece"] = expected_calibration_error(agreement, correct, ece_bins)
    out["confidence_source"] = f"CheXbert agreement of {k} samples with the greedy report"
    out["per_report"]["mean_token_entropy"] = entropy
    out["per_report"]["consensus"] = consensus

    if fit_flagger_budget is not None:
        flagger = ReviewFlagger(fit_flagger_budget).fit(entropy, consensus)
        out["fitted_flagger"] = flagger
    if flagger is not None:
        u = flagger.score(entropy, consensus)
        flagged = flagger.flag(entropy, consensus)
        out["per_report"]["uncertainty"] = u
        out["per_report"]["flagged"] = flagged
        keep = ~flagged
        out["review_flag"] = {
            "flag_rate": float(flagged.mean()),
            "threshold": flagger.threshold,
            "example_f1_unflagged": float(f1_each[keep].mean()) if keep.any() else float("nan"),
            "example_f1_flagged": float(f1_each[flagged].mean()) if flagged.any() else float("nan"),
            "micro_f1_unflagged": clinical_efficacy(y_true[keep], y_pred[keep])["micro_14"]["f1"] if keep.any() else float("nan"),
            **risk_coverage(u, 1.0 - f1_each),
        }
    out["uncertainty_summary"] = {"mean_token_entropy": float(entropy.mean()), "mean_consensus": float(consensus.mean()),
                                  "n_samples": k}
    return out


def main():
    from src.evaluation.chexbert_scorer import CheXbertLabeler
    from src.evaluation.nlg_metrics import lexical_metrics, radgraph_f1
    from src.pipeline.common import labels_for, load_benchmark, load_configs, load_reference_labels, path

    parser = argparse.ArgumentParser()
    parser.add_argument("generations", type=Path)
    parser.add_argument("--uncertainty", action="store_true", help="score sampled reports (S3)")
    parser.add_argument("--fit-flagger", action="store_true", help="fit the review flag here (val only)")
    parser.add_argument("--skip-radgraph", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to the generations file's folder")
    args = parser.parse_args()

    dcfg, ecfg = load_configs()
    records = load_generations(args.generations)
    split = records[0].get("split") or ("val" if "_val" in args.generations.name else "test")
    if args.fit_flagger and split != "val":
        raise SystemExit("The review flag may only be fitted on val (DR-020).")
    name = args.generations.stem.replace("generations_", "") + ("_s3" if args.uncertainty else "")
    out_dir = args.out_dir or args.generations.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    uids = [r["uid"] for r in records]
    y_true = labels_for(load_reference_labels(ecfg), uids)
    all_reports = load_benchmark(dcfg).set_index("uid")["target_report"].fillna("")
    refs = all_reports.loc[uids].tolist()
    retrieved_texts = None
    if any(r.get("retrieved_uids") for r in records):
        retrieved_texts = [all_reports.loc[r.get("retrieved_uids", [])].tolist() for r in records]

    flagger_path = path(ecfg["paths"]["results_dir"]) / "s3" / flagger_filename(args.generations.stem, split)
    flagger = None
    if args.uncertainty and not args.fit_flagger:
        if not flagger_path.exists():
            raise SystemExit(f"{flagger_path} missing — score the val run with --uncertainty --fit-flagger first")
        with open(flagger_path) as f:
            flagger = ReviewFlagger.from_dict(json.load(f))

    labeler = CheXbertLabeler(batch_size=ecfg["evaluation"]["batch_size"])
    res = score_generations(records, y_true, labeler, ece_bins=ecfg["evaluation"]["ece_bins"],
                            use_uncertainty=args.uncertainty, flagger=flagger,
                            fit_flagger_budget=ecfg["uncertainty"]["review_budget"] if args.fit_flagger else None,
                            retrieved_texts=retrieved_texts)

    hyps = [r["generated_report"] for r in records]
    metrics = {"generations": str(args.generations.relative_to(PROJECT_ROOT) if args.generations.is_absolute()
                                  else args.generations),
               "split": split, "n_reports": res["n_reports"], "empty_report_rate": res["empty_report_rate"],
               "clinical": res["clinical"],
               "hallucination_rate": res["hallucination_rate"], "ece": res["ece"],
               "confidence_source": res["confidence_source"], "lexical": lexical_metrics(refs, hyps)}
    for key in ("latency_s_mean", "copy_overlap", "uncertainty_summary", "review_flag"):
        if key in res:
            metrics[key] = res[key]
    if not args.skip_radgraph:
        rg = radgraph_f1(refs, hyps, reward_level=ecfg["evaluation"]["radgraph_reward_level"])
        metrics["radgraph"] = {"radgraph_f1": rg["radgraph_f1"], "reward_level": rg["reward_level"]}
        res["per_report"]["radgraph_f1"] = rg["per_report"]

    pd.DataFrame(res["y_pred"], columns=CHEXBERT_LABELS).assign(uid=uids).to_csv(
        out_dir / f"chexbert_generated_{name}.csv", index=False)
    if "y_samples" in res:
        k = res["y_samples"].shape[1]
        rows = [{"uid": u, "sample": j, **dict(zip(CHEXBERT_LABELS, res["y_samples"][i, j].tolist()))}
                for i, u in enumerate(uids) for j in range(k)]
        pd.DataFrame(rows).to_csv(out_dir / f"chexbert_samples_{name}.csv", index=False)
    res["per_report"].to_csv(out_dir / f"per_report_{name}.csv", index=False)
    with open(out_dir / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if "fitted_flagger" in res:
        flagger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flagger_path, "w") as f:
            json.dump({"fitted_on": metrics["generations"], **res["fitted_flagger"].to_dict()}, f)

    c = metrics["clinical"]
    print(f"{name}: n={metrics['n_reports']} CheXbert micro-F1={c['micro_14']['f1']:.4f} "
          f"macro-F1={c['macro_14']['f1']:.4f} halluc={metrics['hallucination_rate']:.4f} ECE={metrics['ece']:.4f} "
          f"BLEU-4={metrics['lexical']['bleu_4']:.4f} ROUGE-L={metrics['lexical']['rouge_l']:.4f}"
          + (f" RadGraph={metrics['radgraph']['radgraph_f1']:.4f}" if "radgraph" in metrics else ""))
    print(f"Saved metrics_{name}.json to {out_dir}")


if __name__ == "__main__":
    main()
