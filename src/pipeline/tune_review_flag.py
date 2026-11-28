"""
Review-flag v2 (DR-028): select a reference-free risk scorer on val by cross-validated AURC, refit it on
all val, fix the threshold at the review budget, and apply it once to test.

Reads only saved outputs (no model, no GPU):
  results/s1/per_report_<split>_s3.csv, generations_<split>.jsonl, chexbert_generated_<split>_s3.csv,
  results/s1/retrieval_<split>.csv, CheXbert reference labels of the retrieved TRAIN reports.
Writes results/s3/review_flag_v2_selection.json, review_flagger_v2.json, review_flag_v2_<split>.json.

Usage: python src/pipeline/tune_review_flag.py [--apply-test] [--stage s0]
  --stage s0: for a model run without retrieval (MAIRA-2, DR-035); the copy feature is then 0 for every
  report, and the retrieved-report features still use the S1 retrieval of the same patient.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
from src.evaluation.clinical_metrics import clinical_efficacy
from src.pipeline.common import labels_for, load_configs, load_reference_labels, path
from src.uncertainty.consensus import ReviewFlagger
from src.uncertainty.review_features import (FEATURES, LinearRiskModel, aurc, cv_aurc, retrieved_label_stats,
                                             ridge_scorer)


def feature_table(ecfg, split: str, ref: pd.DataFrame, stage: str = "s1"):
    s1 = path(ecfg["paths"]["results_dir"]) / stage
    per = pd.read_csv(s1 / f"per_report_{split}_s3.csv")
    if "copy_overlap_4gram" not in per:          # no retrieval in the prompt (DR-035)
        per["copy_overlap_4gram"] = 0.0
    gens = {}
    with open(s1 / f"generations_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gens[r["uid"]] = r["n_tokens"]
    pred = pd.read_csv(s1 / f"chexbert_generated_{split}_s3.csv").set_index("uid").loc[per["uid"], CHEXBERT_LABELS]
    retr_dir = path(ecfg["paths"]["results_dir"]) / "s1"
    retr = pd.read_csv(retr_dir / f"retrieval_{split}.csv").set_index("uid").loc[per["uid"]]
    k = ecfg["rag"]["k"]
    y_pred = pred.to_numpy(dtype=np.int8)
    y_retr = np.stack([labels_for(ref, retr[f"retrieved_uid_{j + 1}"]) for j in range(k)], axis=1)
    rs = retrieved_label_stats(y_pred, y_retr)
    feats = pd.DataFrame({
        "entropy": per["mean_token_entropy"], "disagreement": 1.0 - per["consensus"],
        "copy": per["copy_overlap_4gram"], "n_tokens": per["uid"].map(gens).astype(float),
        "n_positive": y_pred[:, :-1].sum(1), "pred_normal": y_pred[:, -1],
        "retr_score": retr["score_1"].to_numpy(), "retr_disagree": rs["retr_disagree"],
        "retr_mismatch": rs["retr_mismatch"]})
    assert list(feats.columns) == FEATURES
    y_true = labels_for(ref, per["uid"])
    return per, feats[FEATURES].to_numpy(float), 1.0 - per["example_f1"].to_numpy(float), y_true, y_pred


def v1_scorer(Xtr, rtr):
    e, d = FEATURES.index("entropy"), FEATURES.index("disagreement")
    f = ReviewFlagger(0.2).fit(Xtr[:, e], 1.0 - Xtr[:, d])
    return lambda X: f.score(X[:, e], 1.0 - X[:, d])


def signed_single(col):
    # sign chosen on the training fold only: higher score = more uncertain
    def make(Xtr, rtr):
        sign = 1.0 if np.corrcoef(Xtr[:, col], rtr)[0, 1] >= 0 else -1.0
        return lambda X: sign * X[:, col]
    return make


def flag_report(u, threshold, risk, y_true, y_pred):
    flagged = u > threshold
    keep = ~flagged
    f1 = 1.0 - risk
    return {"n": int(len(u)), "flag_rate": float(flagged.mean()),
            "example_f1_unflagged": float(f1[keep].mean()), "example_f1_flagged": float(f1[flagged].mean()),
            "micro_f1_all": clinical_efficacy(y_true, y_pred)["micro_14"]["f1"],
            "micro_f1_unflagged": clinical_efficacy(y_true[keep], y_pred[keep])["micro_14"]["f1"],
            "aurc": aurc(u, risk), "full_coverage_risk": float(risk.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-test", action="store_true", help="apply the val-selected flag to test (once)")
    ap.add_argument("--stage", choices=["s0", "s1"], default="s1", help="generations folder (s0 for MAIRA-2)")
    args = ap.parse_args()
    dcfg, ecfg = load_configs()
    seed, budget = dcfg["project"]["seed"], ecfg["uncertainty"].get("review_budget", 0.2)
    ref = load_reference_labels(ecfg)
    out_dir = path(ecfg["paths"]["results_dir"]) / "s3"

    _, Xv, rv, ytv, ypv = feature_table(ecfg, "val", ref, args.stage)
    candidates = {"v1_entropy+disagreement_percentiles": v1_scorer}
    for i, name in enumerate(FEATURES):
        candidates[f"single:{name}"] = signed_single(i)
    for l2 in (1.0, 10.0, 100.0):
        candidates[f"ridge_all_l2={l2:g}"] = ridge_scorer(FEATURES, l2)
    cv = {name: cv_aurc(Xv, rv, make, seed=seed) for name, make in candidates.items()}
    best = min(cv, key=cv.get)
    print("val 5-fold CV AURC (lower is better); full-coverage risk", round(float(rv.mean()), 4))
    for name, v in sorted(cv.items(), key=lambda kv: kv[1]):
        print(f"  {v:.4f}  {name}" + ("   <- selected" if name == best else ""))

    if best.startswith("ridge"):
        model = LinearRiskModel(FEATURES, float(best.split("=")[1])).fit(Xv, rv)
        spec, score = model.to_dict(), model.score
    else:
        fn = candidates[best](Xv, rv)
        spec, score = {"type": best}, fn
        if best.startswith("single"):
            col = FEATURES.index(best.split(":")[1])
            spec["sign"] = 1.0 if np.corrcoef(Xv[:, col], rv)[0, 1] >= 0 else -1.0
    uv = score(Xv)
    threshold = float(np.quantile(uv, 1.0 - budget))
    spec.update({"budget": budget, "threshold": threshold, "fitted_on": f"{args.stage}/*_val_s3 ({len(rv)} val patients)"})
    sel = {"cv_aurc_val": cv, "selected": best, "val_full_coverage_risk": float(rv.mean()), "folds": 5, "seed": seed}
    (out_dir / "review_flag_v2_selection.json").write_text(json.dumps(sel, indent=2))
    (out_dir / "review_flagger_v2.json").write_text(json.dumps(spec, indent=2))
    val_rep = flag_report(uv, threshold, rv, ytv, ypv)
    (out_dir / "review_flag_v2_val.json").write_text(json.dumps(val_rep, indent=2))
    print("val (in-sample):", {k: round(v, 4) if isinstance(v, float) else v for k, v in val_rep.items()})

    if args.apply_test:
        _, Xt, rt, ytt, ypt = feature_table(ecfg, "test", ref, args.stage)
        rep = flag_report(score(Xt), threshold, rt, ytt, ypt)
        rep["selected"] = best
        (out_dir / "review_flag_v2_test.json").write_text(json.dumps(rep, indent=2))
        print("test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})


if __name__ == "__main__":
    main()
