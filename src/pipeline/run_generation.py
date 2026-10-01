"""
LLaVA-Med report generation for S0 (image + indication) and S1 (+ top-k train-report RAG).

S2 reuses the S1 text (Grad-CAM does not change generation, DR-016); S3 reuses the S1 greedy
text and adds K stochastic samples, produced here with --samples 5 in the same pass.

Output is append-only JSONL (one line per patient) so interrupted Kaggle sessions resume:
  results/<stage>/generations_<split>.jsonl
  results/<stage>/run_config_<split>.json   (model, quantization, prompt, decoding, seed)

Usage:
  python src/pipeline/run_generation.py --stage s0 --split test --limit 20 --quant 4bit
  python src/pipeline/run_generation.py --stage s1 --split val --samples 5 --quant fp16
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import platform

import pandas as pd

from src.models.prompts import PROMPT_TEMPLATES, RAG_HEADER, RAG_LAYOUTS, build_query, rag_prompt, s0_prompt
from src.pipeline.common import load_benchmark, load_configs, load_image, path, set_seed


def output_paths(ecfg, stage: str, split: str, tag: str = ""):
    out_dir = path(ecfg["paths"]["results_dir"]) / stage
    suffix = f"{split}{'_' + tag if tag else ''}"
    return out_dir, out_dir / f"generations_{suffix}.jsonl", out_dir / f"run_config_{suffix}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["s0", "s1"], required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--quant", choices=["4bit", "8bit", "fp16"], default="4bit")
    parser.add_argument("--samples", type=int, default=0, help="stochastic samples per patient (S3 uses 5)")
    parser.add_argument("--limit", type=int, default=None, help="first N patients (smoke tests)")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. smoke20")
    parser.add_argument("--rag-layout", choices=list(RAG_LAYOUTS), default=None,
                        help="override rag.layout from the config (val comparisons only)")
    args = parser.parse_args()

    dcfg, ecfg = load_configs()
    seed = dcfg["project"]["seed"]
    set_seed(seed)

    df = load_benchmark(dcfg)
    reports = df.set_index("uid")["target_report"]
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    retrieval = None
    if args.stage == "s1":
        retrieval = pd.read_csv(path(ecfg["paths"]["results_dir"]) / "s1" / f"retrieval_{args.split}.csv").set_index("uid")
        k = ecfg["rag"]["k"]
        # Retrieved reports must come from train patients only (DR-004).
        train_uids = set(load_benchmark(dcfg).query("split == 'train'")["uid"])
        retrieved = retrieval[[f"retrieved_uid_{j + 1}" for j in range(k)]].to_numpy().ravel()
        assert set(retrieved) <= train_uids, "retrieval file references non-train patients"

    out_dir, gen_path, cfg_path = output_paths(ecfg, args.stage, args.split, args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if gen_path.exists():
        with open(gen_path) as f:
            done = {json.loads(line)["uid"] for line in f if line.strip()}
    todo = df[~df["uid"].isin(done)]
    print(f"{args.stage}/{args.split}: {len(df)} patients, {len(done)} already done, {len(todo)} to generate")

    from src.models.llava_med import LlavaMedGenerator
    import torch
    import transformers

    unc = ecfg["uncertainty"]
    template_id = ecfg["vlm"]["prompt_template_id"]
    layout = args.rag_layout or ecfg["rag"]["layout"]
    gen = LlavaMedGenerator(ecfg["vlm"]["model_id"], quantization=args.quant,
                            max_new_tokens=ecfg["vlm"]["max_new_tokens"])
    run_config = {
        "stage": args.stage, "split": args.split, "n_patients": int(len(df)), "limit": args.limit,
        "model_id": ecfg["vlm"]["model_id"], "quantization": args.quant,
        "decoding_greedy": {"do_sample": False, "num_beams": 1, "max_new_tokens": ecfg["vlm"]["max_new_tokens"]},
        "decoding_samples": ({"n": args.samples, "temperature": unc["temperature"], "top_p": unc["top_p"],
                              "top_k": 0, "seed": f"{seed} + uid"} if args.samples else None),
        "prompt_template_id": template_id, "prompt_template": PROMPT_TEMPLATES[template_id],
        "rag": ({"k": ecfg["rag"]["k"], "layout": layout, "header": RAG_HEADER} if args.stage == "s1" else None),
        "image_preprocessing": "pad-to-square (CLIP mean colour) -> CLIPImageProcessor 336px",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(), "seed": seed,
    }
    with open(cfg_path, "w") as f:
        json.dump(run_config, f, indent=2)

    with open(gen_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo.itertuples(index=False)):
            query = build_query(row.clean_indication, template_id)
            if args.stage == "s1":
                r = retrieval.loc[row.uid]
                context_uids = [int(r[f"retrieved_uid_{j + 1}"]) for j in range(ecfg["rag"]["k"])]
                prompt = rag_prompt(query, [reports[u] for u in context_uids], layout=layout)
            else:
                context_uids = []
                prompt = s0_prompt(query)

            res = gen.generate(load_image(dcfg, row.primary_image_filename), prompt, n_samples=args.samples,
                               temperature=unc["temperature"], top_p=unc["top_p"], seed=seed + int(row.uid))
            record = {"uid": int(row.uid), "stage": args.stage, "split": args.split, "prompt": prompt,
                      "retrieved_uids": context_uids, "generated_report": res.text, "n_tokens": res.n_tokens,
                      "mean_token_entropy": res.mean_token_entropy, "latency_s": round(res.latency_s, 3),
                      "samples": res.samples, "samples_latency_s": round(res.samples_latency_s, 3)}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{len(done) + i + 1}/{len(df)}] uid={row.uid} tokens={res.n_tokens} "
                  f"H={res.mean_token_entropy:.3f} t={res.latency_s:.1f}s", flush=True)
    print(f"Wrote {gen_path}")


if __name__ == "__main__":
    main()
