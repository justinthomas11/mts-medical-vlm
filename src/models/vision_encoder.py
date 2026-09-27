"""
Frozen LLaVA-Med CLIP ViT-L/14-336 vision tower, loaded on its own (DR-016).

Only the `vision_tower.*` tensors of the LLaVA-Med checkpoint are loaded, so S1 retrieval
and the S2 Grad-CAM head use exactly the encoder the VLM sees, without the 7B language model.

Outputs:
  - patch tokens from layer -2 (LLaVA's `vision_feature_layer`), shape (B, 576, 1024)
  - pooled image embedding: post-layernorm CLS of the final layer, shape (B, 1024)
"""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import torch
from PIL import Image

from src.models.llava_med import pad_to_square

FEATURE_LAYER = -2
GRID = 24  # 336 / 14


class LlavaMedVisionEncoder(torch.nn.Module):
    def __init__(self, model_id: str, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.device, self.dtype = torch.device(device), dtype

        with open(hf_hub_download(model_id, "config.json")) as f:
            vision_cfg = json.load(f)["vision_config"]
        vision_cfg.pop("model_type", None)
        self.model = CLIPVisionModel(CLIPVisionConfig(**vision_cfg))

        with open(hf_hub_download(model_id, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        prefix = "vision_tower."
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
        state = {}
        for shard in shards:
            with safe_open(hf_hub_download(model_id, shard), framework="pt") as sf:
                for key in sf.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = sf.get_tensor(key)
        # The checkpoint stores `vision_model.*`; newer transformers name CLIPVisionModel weights without it.
        inner = "vision_model."
        if not any(k.startswith(inner) for k in self.model.state_dict()):
            state = {k[len(inner):] if k.startswith(inner) else k: v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"Vision tower mismatch: missing={missing} unexpected={unexpected}")

        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model.to(self.device, self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        squared = [pad_to_square(im) for im in images]
        return self.processor(images=squared, return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(pixel_values=pixel_values.to(self.device, self.dtype), output_hidden_states=True)
        return {
            "patch_tokens": out.hidden_states[FEATURE_LAYER][:, 1:],  # drop CLS
            "pooled": out.pooler_output,
        }
