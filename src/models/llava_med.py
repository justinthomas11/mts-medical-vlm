"""
LLaVA-Med v1.5 (Mistral-7B, CLIP ViT-L/14-336) loader and report generator.

Uses the Hugging Face-format conversion of microsoft/llava-med-v1.5-mistral-7b so it runs on
stock `transformers` (LlavaForConditionalGeneration). Quantization: 4bit / 8bit (bitsandbytes,
local RTX 3050) or fp16 (Kaggle/Colab), per DR-001.

One call to `generate` produces the greedy report with its mean predictive token entropy
(DR-015 tier 1) and, optionally, K stochastic samples (DR-015 tier 2).
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time

import torch
from PIL import Image

# LLaVA-1.5 pads non-square images with the CLIP mean colour before resizing ("image_aspect_ratio: pad").
CLIP_MEAN_RGB = (122, 116, 104)


def pad_to_square(image: Image.Image, fill=CLIP_MEAN_RGB) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def token_entropies(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of each row of a (T, V) logit matrix."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


@dataclass
class GenerationResult:
    text: str
    n_tokens: int
    mean_token_entropy: float
    latency_s: float
    samples: List[str] = field(default_factory=list)
    samples_latency_s: float = 0.0


class LlavaMedGenerator:
    def __init__(self, model_id: str, quantization: str = "4bit", max_new_tokens: int = 256,
                 device_map: str = "auto"):
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

        self.model_id = model_id
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens

        kwargs = {"device_map": device_map, "low_cpu_mem_usage": True}
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                # Keep the CLIP vision tower and projector unquantized (small, and precision-sensitive).
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
            )
            kwargs["dtype"] = torch.float16
        elif quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
            kwargs["dtype"] = torch.float16
        elif quantization == "fp16":
            kwargs["dtype"] = torch.float16
        else:
            raise ValueError(f"Unknown quantization: {quantization}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id or self.eos_token_id

    def _inputs(self, image: Image.Image, prompt: str):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=pad_to_square(image), text=text, return_tensors="pt")
        return inputs.to(self.model.device, torch.float16)

    def _decode(self, sequences: torch.Tensor, prompt_len: int) -> List[str]:
        return [s.strip() for s in self.processor.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)]

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str, n_samples: int = 0, temperature: float = 0.7,
                 top_p: float = 0.9, seed: Optional[int] = None) -> GenerationResult:
        inputs = self._inputs(image, prompt)
        prompt_len = inputs["input_ids"].shape[1]
        common = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.pad_token_id,
                      eos_token_id=self.eos_token_id)

        t0 = time.perf_counter()
        out = self.model.generate(**inputs, do_sample=False, num_beams=1, output_logits=True,
                                  return_dict_in_generate=True, **common)
        latency = time.perf_counter() - t0

        gen_ids = out.sequences[0, prompt_len:]
        # Raw (unprocessed) logits per generated step; truncate at the first EOS (inclusive).
        n_steps = len(out.logits)
        eos_pos = (gen_ids == self.eos_token_id).nonzero()
        n_tokens = int(eos_pos[0, 0]) + 1 if len(eos_pos) else n_steps
        step_logits = torch.stack([l[0] for l in out.logits[:n_tokens]])
        entropy = float(token_entropies(step_logits).mean()) if n_tokens else float("nan")

        result = GenerationResult(text=self._decode(out.sequences, prompt_len)[0], n_tokens=n_tokens,
                                  mean_token_entropy=entropy, latency_s=latency)

        if n_samples > 0:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            t0 = time.perf_counter()
            sampled = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p,
                                          top_k=0, num_return_sequences=n_samples, **common)
            result.samples_latency_s = time.perf_counter() - t0
            result.samples = self._decode(sampled, prompt_len)
        return result
