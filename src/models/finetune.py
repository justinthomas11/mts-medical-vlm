"""
LoRA fine-tuning of LLaVA-Med on TRAIN-split reports (Phase III, DR-030).

Each train patient contributes one example: the S0 query (image + indication) or, with probability
`rag_fraction`, the S1 query with the top-k retrieved TRAIN reports of OTHER patients (self excluded).
The target is the patient's own reference report. Mixing both formats lets one adapter serve S0 and S1,
and teaches the model that retrieved reports are evidence, not the answer: the target never equals them.

Loss is computed on the target tokens only (prompt and image tokens are masked with -100).
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from src.models.prompts import build_query, rag_prompt, s0_prompt

IGNORE_INDEX = -100
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


@dataclass
class Example:
    uid: int
    image_file: str
    prompt: str
    target: str
    uses_rag: bool


def build_examples(rows: Sequence[Dict], reports: Dict[int, str], retrieved: Dict[int, List[int]],
                   template_id: str, layout: str, rag_fraction: float, seed: int) -> List[Example]:
    """rows: dicts with uid, clean_indication, primary_image_filename, target_report.
    retrieved[uid]: retrieved TRAIN uids for that patient (must not contain uid itself)."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        uid = int(r["uid"])
        query = build_query(r["clean_indication"], template_id)
        use_rag = bool(rng.random() < rag_fraction)
        if use_rag:
            ctx = retrieved[uid]
            if uid in ctx:
                raise ValueError(f"patient {uid} retrieved itself; training retrieval must exclude self")
            prompt = rag_prompt(query, [reports[u] for u in ctx], layout=layout)
        else:
            prompt = s0_prompt(query)
        out.append(Example(uid, r["primary_image_filename"], prompt, str(r["target_report"]).strip(), use_rag))
    return out


def mask_prompt(input_ids, prompt_len: int):
    """Labels for causal-LM training: copy of input_ids with the first prompt_len positions ignored."""
    labels = input_ids.clone()
    labels[..., :prompt_len] = IGNORE_INDEX
    return labels


def encode(processor, image, prompt: str, target: str):
    """Returns processor inputs for prompt+target and labels masking the prompt (incl. image tokens)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    eos = processor.tokenizer.eos_token or ""
    prompt_ids = processor(images=image, text=prompt_text, return_tensors="pt")["input_ids"]
    full = processor(images=image, text=prompt_text + " " + target + eos, return_tensors="pt")
    full["labels"] = mask_prompt(full["input_ids"], prompt_ids.shape[1])
    return full
