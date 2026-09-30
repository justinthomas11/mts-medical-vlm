"""
Grad-CAM over the frozen vision encoder's layer -2 patch tokens (DR-016).

The target layer is the patch-token output that LLaVA-Med feeds to its projector. Because the
encoder is frozen, Grad-CAM needs gradients only through the classification head:
    weights_k = mean_t d(logit_c) / d(A_{t,k})
    CAM_t     = ReLU( sum_k weights_k * A_{t,k} )
reshaped to the 24x24 patch grid.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.models.vision_encoder import GRID


def gradcam(head: torch.nn.Module, tokens: torch.Tensor, class_idx: int) -> np.ndarray:
    """tokens: (576, D) patch tokens for one image. Returns a (24, 24) CAM scaled to [0, 1]."""
    head.eval()
    device = next(head.parameters()).device
    a = tokens.detach().to(device, torch.float32).unsqueeze(0).requires_grad_(True)
    logit = head(a)[0, class_idx]
    (grad,) = torch.autograd.grad(logit, a)
    weights = grad[0].mean(dim=0)                      # (D,)
    cam = F.relu((a[0] * weights).sum(dim=1)).detach()  # (576,)
    cam = cam.reshape(GRID, GRID).cpu().numpy()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsample of a square CAM to (size, size)."""
    t = torch.as_tensor(cam, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
