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


def format_compliant(text: str) -> bool:
    upper = text.upper()
    return "FINDINGS" in upper and "IMPRESSION" in upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--quant", choices=["4bit", "8bit", "fp16"], default="4bit")
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
    from src.evaluation.clinical_metrics import clinical_efficacy
    from src.models.llava_med import LlavaMedGenerator

    gen = LlavaMedGenerator(ecfg["vlm"]["model_id"], quantization=args.quant,
                            max_new_tokens=ecfg["vlm"]["max_new_tokens"])
    texts = {}
    for tid in PROMPT_TEMPLATES:
        path_out = out_dir / f"generations_val_{tid}.jsonl"
        with open(path_out, "w", encoding="utf-8") as f:
            outs = []
            for row in df.itertuples(index=False):
                prompt = build_query(row.clean_indication, tid)
                res = gen.generate(load_image(dcfg, row.primary_image_filename), prompt)
                outs.append(res.text)
                f.write(json.dumps({"uid": int(row.uid), "template": tid, "prompt": prompt,
                                    "generated_report": res.text, "n_tokens": res.n_tokens}, ensure_ascii=False) + "\n")
        texts[tid] = outs
        print(f"  {tid}: generated {len(outs)} val reports")

    del gen
    import torch
    torch.cuda.empty_cache()

    labeler = CheXbertLabeler()
    results = {}
    for tid, outs in texts.items():
        ce = clinical_efficacy(y_true, labeler.label(outs))
        results[tid] = {
            "chexbert_micro_f1_14": ce["micro_14"]["f1"],
            "chexbert_macro_f1_14": ce["macro_14"]["f1"],
            "format_compliance": float(np.mean([format_compliant(t) for t in outs])),
            "mean_words": float(np.mean([len(t.split()) for t in outs])),
        }
        print(f"  {tid}: {results[tid]}")

    best = max(results, key=lambda t: (results[t]["chexbert_micro_f1_14"], results[t]["format_compliance"]))
    with open(out_dir / "val_prompt_selection.json", "w") as f:
        json.dump({"split": "val", "n_patients": int(len(df)), "uids": df["uid"].astype(int).tolist(),
                   "quantization": args.quant, "model_id": ecfg["vlm"]["model_id"],
                   "criterion": "CheXbert micro-F1 (14), tie-break format compliance",
                   "templates": PROMPT_TEMPLATES, "results": results, "selected": best}, f, indent=2)
    print(f"Selected template: {best}")


if __name__ == "__main__":
    main()
