"""
Selects the S0–S3 query template on VAL patients only (DR-023). Test is never read.

Each candidate in src/models/prompts.py generates a greedy report for the first N val patients
(by uid). Selection criterion: CheXbert micro-F1 (14 observations) against the val reference
labels; FINDINGS/IMPRESSION format compliance and mean length are reported alongside.

Outputs: results/prompt_selection/generations_val_<template>.jsonl, val_prompt_selection.json

Usage: python src/pipeline/select_prompt.py --n 40 --quant 4bit
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np

from src.models.prompts import PROMPT_TEMPLATES, build_query
from src.pipeline.common import labels_for, load_benchmark, load_configs, load_image, load_reference_labels, path, set_seed


CRITERIA = {"micro": "chexbert_micro_f1_14", "macro": "chexbert_macro_f1_14"}


def format_compliant(text: str) -> bool:
    upper = text.upper()
    return "FINDINGS" in upper and "IMPRESSION" in upper


def load_existing(p: Path):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_scores(y_true: np.ndarray, y_pred: np.ndarray, texts) -> dict:
    """Clinical scores of one template; 'abnormal' = the 13 observations other than No Finding."""
    from src.evaluation.clinical_metrics import clinical_efficacy

    ce = clinical_efficacy(y_true, y_pred)
    t, p = y_true[:, :13].astype(bool), y_pred[:, :13].astype(bool)
    return {
        "chexbert_micro_f1_14": ce["micro_14"]["f1"],
        "chexbert_macro_f1_14": ce["macro_14"]["f1"],
        "abnormal_labels_predicted": int(p.sum()),
        "abnormal_labels_correct": int((t & p).sum()),
        "abnormal_labels_in_reference": int(t.sum()),
        "reports_called_normal": int(y_pred[:, 13].sum()),
        "format_compliance": float(np.mean([format_compliant(x) for x in texts])),
        "mean_words": float(np.mean([len(x.split()) for x in texts])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--quant", choices=["4bit", "8bit", "fp16"], default="4bit")
    parser.add_argument("--criterion", choices=list(CRITERIA), default="micro")
    parser.add_argument("--output", default="val_prompt_selection.json")
    args = parser.parse_args()

    dcfg, ecfg = load_configs()
    seed = dcfg["project"]["seed"]
    set_seed(seed)
    df = load_benchmark(dcfg)
    df = df[df["split"] == "val"].head(args.n).reset_index(drop=True)
    ref = load_reference_labels(ecfg)
    y_true = labels_for(ref, df["uid"])

    out_dir = path(ecfg["paths"]["results_dir"]) / "prompt_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluation.chexbert_scorer import CheXbertLabeler
    from src.models.llava_med import LlavaMedGenerator

    uids = df["uid"].astype(int).tolist()
    texts, gen = {}, None
    for tid in PROMPT_TEMPLATES:
        path_out = out_dir / f"generations_val_{tid}.jsonl"
        existing = load_existing(path_out)
        if existing is not None and [r["uid"] for r in existing] == uids \
                and all(r["prompt"] == build_query(row.clean_indication, tid) for r, row in zip(existing, df.itertuples())):
            texts[tid] = [r["generated_report"] for r in existing]   # same patients and prompt: reuse
            print(f"  {tid}: reusing {len(uids)} existing val reports")
            continue
        # Resume: keep earlier lines only if they are a prefix of this patient list with this exact prompt.
        done = {}
        if existing:
            for r, row in zip(existing, df.itertuples()):
                if r["uid"] != row.uid or r["prompt"] != build_query(row.clean_indication, tid):
                    break
                done[r["uid"]] = r
        if gen is None:
            gen = LlavaMedGenerator(ecfg["vlm"]["model_id"], quantization=args.quant,
                                    max_new_tokens=ecfg["vlm"]["max_new_tokens"])
        with open(path_out, "w", encoding="utf-8") as f:
            outs = []
            for row in df.itertuples(index=False):
                prompt = build_query(row.clean_indication, tid)
                if row.uid in done:
                    rec = done[row.uid]
                else:
                    res = gen.generate(load_image(dcfg, row.primary_image_filename), prompt)
                    rec = {"uid": int(row.uid), "template": tid, "prompt": prompt,
                           "generated_report": res.text, "n_tokens": res.n_tokens}
                outs.append(rec["generated_report"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
        texts[tid] = outs
        print(f"  {tid}: {len(done)} resumed, {len(outs) - len(done)} generated", flush=True)

    if gen is not None:
        del gen
        import torch
        torch.cuda.empty_cache()

    labeler = CheXbertLabeler()
    results = {tid: prompt_scores(y_true, labeler.label(outs), outs) for tid, outs in texts.items()}
    for tid, r in results.items():
        print(f"  {tid}: {r}")

    key = CRITERIA[args.criterion]
    best = max(results, key=lambda t: (results[t][key], results[t]["format_compliance"]))
    with open(out_dir / args.output, "w") as f:
        json.dump({"split": "val", "n_patients": int(len(df)), "uids": uids,
                   "quantization": args.quant, "model_id": ecfg["vlm"]["model_id"],
                   "criterion": f"{key}, tie-break format compliance",
                   "templates": PROMPT_TEMPLATES, "results": results, "selected": best}, f, indent=2)
    print(f"Selected template ({key}): {best}")


if __name__ == "__main__":
    main()
