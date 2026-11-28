"""
MAIRA-2 (microsoft/maira-2) as a second, chest-specific base VLM (DR-035).

MAIRA-2 was trained on MIMIC-CXR, PadChest and a private US set; its authors hold IU X-Ray out as an
external test set, so our IU test split is unseen. It writes the FINDINGS section from a frontal image
plus optional lateral, prior study, indication, technique and comparison. We pass the frontal image and
the indication only (same inputs as LLaVA-Med S0); no retrieval (MAIRA-2's prior-study slot is for the
same patient, so retrieved reports of other patients would be mislabelled as priors).

Same interface as LlavaMedGenerator: greedy report + mean token entropy, and K stochastic samples (DR-015).
Requires transformers 4.x with trust_remote_code and the gated model licence accepted on Hugging Face.
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time

import torch
from PIL import Image

from src.models.llava_med import GenerationResult, token_entropies

MAIRA2_ID = "microsoft/maira-2"


def clean_indication(text) -> Optional[str]:
    """MAIRA-2 expects a short sentence such as 'Dyspnea.'; empty / missing indications become None."""
    if text is None or (isinstance(text, float) and text != text):
        return None
    s = str(text).strip()
    if not s:
        return None
    return s if s.endswith(".") else s + "."


class Maira2Generator:
    def __init__(self, model_id: str = MAIRA2_ID, max_new_tokens: int = 300, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id, self.quantization, self.max_new_tokens = model_id, "fp16", max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16,
                                                          device_map=device_map).eval()
        tok = self.processor.tokenizer
        self.eos_token_id = tok.eos_token_id
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _inputs(self, image: Image.Image, indication: Optional[str]):
        inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=image.convert("L"), current_lateral=None, prior_frontal=None,
            indication=indication, technique=None, comparison=None, prior_report=None,
            return_tensors="pt", get_grounding=False)
        device = next(self.model.parameters()).device
        return {k: (v.to(device, torch.float16) if v.is_floating_point() else v.to(device)) for k, v in inputs.items()}

    def _decode(self, ids: torch.Tensor) -> str:
        text = self.processor.decode(ids, skip_special_tokens=True).lstrip()
        return str(self.processor.convert_output_to_plaintext_or_grounded_sequence(text)).strip()

    @torch.no_grad()
    def generate(self, image: Image.Image, indication: Optional[str], n_samples: int = 0, temperature: float = 0.7,
                 top_p: float = 0.9, seed: Optional[int] = None) -> GenerationResult:
        inputs = self._inputs(image, indication)
        prompt_len = inputs["input_ids"].shape[1]
        common = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.pad_token_id,
                      eos_token_id=self.eos_token_id, use_cache=True)
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, do_sample=False, num_beams=1, output_logits=True,
                                  return_dict_in_generate=True, **common)
        latency = time.perf_counter() - t0
        gen_ids = out.sequences[0, prompt_len:]
        eos_pos = (gen_ids == self.eos_token_id).nonzero()
        n_tokens = int(eos_pos[0, 0]) + 1 if len(eos_pos) else len(out.logits)
        step_logits = torch.stack([l[0] for l in out.logits[:n_tokens]])
        entropy = float(token_entropies(step_logits).mean()) if n_tokens else float("nan")
        result = GenerationResult(text=self._decode(gen_ids), n_tokens=n_tokens, mean_token_entropy=entropy,
                                  latency_s=latency)
        if n_samples > 0:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            t0 = time.perf_counter()
            sampled = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                                          num_return_sequences=n_samples, **common)
            result.samples_latency_s = time.perf_counter() - t0
            result.samples = [self._decode(s[prompt_len:]) for s in sampled]
        return result
