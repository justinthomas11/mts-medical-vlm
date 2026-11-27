"""
Phase III (DR-030): LoRA fine-tuning of LLaVA-Med on the TRAIN split, model selection on val loss.

Steps:
  1. Train-split retrieval with self-exclusion (same hybrid index and alpha as S1) ->
     results/finetune/retrieval_train.csv
  2. Examples: S0 or S1 prompt per patient (rag_fraction), target = own reference report.
  3. QLoRA (4-bit, local/Kaggle) or LoRA on FP16 weights; loss on report tokens only.
     Val loss on a fixed val subset after every `eval_every` optimizer steps; best adapter kept.
Outputs: checkpoints/<adapter_dir>/ (git-ignored), results/finetune/train_log_<tag>.json

Usage:
  python src/pipeline/finetune_vlm.py --quant 4bit --max-steps 4 --tag smoke      # memory check
  python src/pipeline/finetune_vlm.py --quant 4bit --epochs 1
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import math
import time

import numpy as np
import pandas as pd


def train_retrieval(dcfg, ecfg, k: int) -> pd.DataFrame:
    """Top-k TRAIN neighbours of every TRAIN patient, excluding the patient itself."""
    from src.pipeline.common import load_benchmark, path
    from src.rag.retriever import HybridRetriever

    feat = path(ecfg["paths"]["features_dir"])
    index = pd.read_csv(feat / "feature_index.csv")
    image_emb = np.load(feat / "clip_pooled.npy")
    art = np.load(feat / "medcpt_article.npy")
    qry = np.load(feat / "medcpt_query.npy")
    alpha = json.loads((path(ecfg["paths"]["results_dir"]) / "s1" / "retrieval_tuning.json").read_text())["selected_alpha"]
    tr = np.where(index["split"] == "train")[0]
    uids = index["uid"].to_numpy()[tr]
    retriever = HybridRetriever(uids, image_emb[tr], art[tr], alpha=alpha)
    found, scores = retriever.search(image_emb[tr], qry[tr], k=k + 1)
    rows = []
    for u, cand, sc in zip(uids, found, scores):
        keep = [(c, s) for c, s in zip(cand, sc) if c != u][:k]
        row = {"uid": int(u)}
        for j, (c, s) in enumerate(keep):
            row[f"retrieved_uid_{j + 1}"], row[f"score_{j + 1}"] = int(c), float(s)
        rows.append(row)
    assert set(load_benchmark(dcfg).query("split == 'train'")["uid"]) == set(uids)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["4bit", "fp16"], default="4bit")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="optimizer steps (smoke tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-target-tokens", type=int, default=200)
    ap.add_argument("--model-id", default=None, help="override vlm.model_id (tiny model for code smoke tests)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    from src.models.finetune import LORA_TARGETS, build_examples, encode
    from src.models.llava_med import pad_to_square
    from src.pipeline.common import load_benchmark, load_configs, load_image, path, set_seed

    dcfg, ecfg = load_configs()
    ft = ecfg["finetune"]
    seed = dcfg["project"]["seed"]
    set_seed(seed)
    k = ecfg["rag"]["k"]
    out_dir = path(ecfg["paths"]["results_dir"]) / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    retr_path = out_dir / "retrieval_train.csv"
    if not retr_path.exists():
        train_retrieval(dcfg, ecfg, k).to_csv(retr_path, index=False)
    retr = pd.read_csv(retr_path).set_index("uid")
    retrieved_train = {int(u): [int(r[f"retrieved_uid_{j + 1}"]) for j in range(k)] for u, r in retr.iterrows()}

    df = load_benchmark(dcfg)
    reports = df.set_index("uid")["target_report"].to_dict()
    cols = ["uid", "clean_indication", "primary_image_filename", "target_report"]
    template_id, layout = ecfg["vlm"]["prompt_template_id"], ecfg["rag"]["layout"]
    train_rows = df[df["split"] == "train"][cols].to_dict("records")
    train_ex = build_examples(train_rows, reports, retrieved_train, template_id, layout, ft["rag_fraction"], seed)

    # Val loss subset: S1 prompts use the committed val retrieval (train-only index).
    val_retr = pd.read_csv(path(ecfg["paths"]["results_dir"]) / "s1" / "retrieval_val.csv").set_index("uid")
    retrieved_val = {int(u): [int(r[f"retrieved_uid_{j + 1}"]) for j in range(k)] for u, r in val_retr.iterrows()}
    val_rows = df[df["split"] == "val"].sort_values("uid").head(ft["val_loss_patients"])[cols].to_dict("records")
    val_ex = build_examples(val_rows, reports, retrieved_val, template_id, layout, ft["rag_fraction"], seed + 1)

    model_id = args.model_id or ecfg["vlm"]["model_id"]
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "dtype": torch.float16}
    if args.quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=ft["lora_r"], lora_alpha=ft["lora_alpha"], lora_dropout=ft["lora_dropout"],
                      target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable LoRA parameters: {n_train:,}")
    # LoRA weights in fp32 for stable fp16 mixed-precision updates.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    device = next(model.parameters()).device
    tok = processor.tokenizer

    def batch_for(ex):
        tgt_ids = tok(ex.target, add_special_tokens=False)["input_ids"][: args.max_target_tokens]
        target = tok.decode(tgt_ids)
        enc = encode(processor, pad_to_square(load_image(dcfg, ex.image_file)), ex.prompt, target)
        return {k_: (v.to(device, torch.float16) if v.dtype.is_floating_point else v.to(device))
                for k_, v in enc.items()}

    def val_loss():
        model.eval()
        tot = n = 0.0
        with torch.no_grad():
            for ex in val_ex:
                b = batch_for(ex)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**b).loss
                tot += float(loss)
                n += 1
        model.train()
        return tot / n

    accum = ft["grad_accum"]
    steps_per_epoch = math.ceil(len(train_ex) / accum)
    total_steps = args.max_steps or int(math.ceil((args.epochs or ft["epochs"]) * steps_per_epoch))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ft["lr"], weight_decay=0.0)
    warmup = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda")

    adapter_dir = path(ecfg["paths"]["checkpoints_dir"]) / (ft["adapter_dir"] + (f"_{args.tag}" if args.tag else ""))
    log = {"model_id": model_id, "quant": args.quant, "config": ft, "n_train_examples": len(train_ex),
           "n_train_rag": int(sum(e.uses_rag for e in train_ex)), "n_val_loss_examples": len(val_ex),
           "total_optimizer_steps": total_steps, "trainable_params": n_train, "history": []}
    rng = np.random.default_rng(seed)
    order = []
    step, micro, best, t0 = 0, 0, float("inf"), time.time()
    model.train()
    if not args.max_steps:
        log["history"].append({"step": 0, "val_loss": val_loss(), "elapsed_s": 0.0})
        print(f"step 0 val_loss {log['history'][-1]['val_loss']:.4f}", flush=True)
    running = 0.0
    while step < total_steps:
        if not order:
            order = list(rng.permutation(len(train_ex)))
        ex = train_ex[order.pop()]
        b = batch_for(ex)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**b).loss / accum
        scaler.scale(loss).backward()
        running += float(loss)
        micro += 1
        if micro % accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 10 == 0 or args.max_steps:
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"step {step}/{total_steps} train_loss {running / (10 if not args.max_steps else 1):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} peak_mem {mem:.2f} GiB {time.time() - t0:.0f}s", flush=True)
            running = 0.0
        if not args.max_steps and (step % ft["eval_every"] == 0 or step == total_steps):
            vl = val_loss()
            log["history"].append({"step": step, "val_loss": vl, "elapsed_s": time.time() - t0})
            print(f"step {step} val_loss {vl:.4f}", flush=True)
            if vl < best:
                best = vl
                model.save_pretrained(adapter_dir)
                log["best"] = {"step": step, "val_loss": vl}
            (out_dir / f"train_log{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps(log, indent=2))
    if args.max_steps:
        model.save_pretrained(adapter_dir)
        log["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        log["sec_per_step"] = (time.time() - t0) / step
        (out_dir / f"train_log{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps(log, indent=2))
    print(f"done; adapter at {adapter_dir}")


if __name__ == "__main__":
    main()
