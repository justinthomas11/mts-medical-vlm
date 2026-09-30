"""
Condition-specific anatomical target masks for the Pointing Game / SMR (DR-016, DR-019).

Base structures come from the torchxrayvision ChestX-Det PSPNet (14 structures). Derived
compartments (lower lung zone, apical/peripheral pleural rim) are built from the lung masks.
All masks live in the padded-square frame shared with the Grad-CAM heatmaps.
"""

from typing import Dict

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

SEG_SIZE = 512

# CheXbert observation -> anatomical target compartment
CONDITION_TARGETS: Dict[str, str] = {
    "Cardiomegaly": "heart",
    "Pleural Effusion": "costophrenic_lower_zone",   # lower third of lungs, dilated (DR-023)
    "Pneumothorax": "pleural_rim",
    "Enlarged Cardiomediastinum": "mediastinum_aorta",
    "Fracture": "bony_thorax",
}


def pad_to_square_gray(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


class AnatomySegmenter:
    def __init__(self, device: str = None):
        import torchxrayvision as xrv

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(self.device).eval()
        self.targets = list(self.model.targets)

    @torch.no_grad()
    def segment(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """Returns boolean (512, 512) masks for the 14 PSPNet structures, in the padded-square frame."""
        square = pad_to_square_gray(image).resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
        x = np.asarray(square, dtype=np.float32) / 255.0 * 2048.0 - 1024.0  # xrv range [-1024, 1024]
        x = torch.from_numpy(x)[None, None].to(self.device)
        probs = torch.sigmoid(self.model(x))[0].cpu().numpy()
        return {name: probs[i] > 0.5 for i, name in enumerate(self.targets)}


def _vertical_fraction(mask: np.ndarray, start: float, end: float) -> np.ndarray:
    """Keeps rows of `mask` between fractions [start, end) of its own vertical extent."""
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return np.zeros_like(mask)
    top, bottom = rows[0], rows[-1] + 1
    lo, hi = top + int(start * (bottom - top)), top + int(end * (bottom - top))
    out = np.zeros_like(mask)
    out[lo:hi] = mask[lo:hi]
    return out


def derive_targets(structures: Dict[str, np.ndarray], rim_fraction: float = 0.06) -> Dict[str, np.ndarray]:
    """Builds the compartment masks named in CONDITION_TARGETS from the 14 base structures."""
    lungs = [structures["Left Lung"], structures["Right Lung"]]
    size = structures["Left Lung"].shape[0]

    lower_zone = np.zeros((size, size), dtype=bool)
    rim = np.zeros((size, size), dtype=bool)
    radius = max(1, int(rim_fraction * size))
    for lung in lungs:
        lower_zone |= _vertical_fraction(lung, 2 / 3, 1.0)
        peripheral = lung & ~ndimage.binary_erosion(lung, iterations=radius)
        rim |= peripheral | _vertical_fraction(lung, 0.0, 0.25)
    # DR-023: widen the lung bases by the rim width to reach the costophrenic angles, where fluid
    # blunts the lung edge. "Facies Diaphragmatica" is NOT used: PSPNet marks the sub-diaphragmatic
    # (abdominal) region with it, which would make effusion hits near-automatic.
    lower_zone = ndimage.binary_dilation(lower_zone, iterations=radius)

    return {
        "heart": structures["Heart"],
        "costophrenic_lower_zone": lower_zone,
        "pleural_rim": rim,
        "mediastinum_aorta": structures["Mediastinum"] | structures["Aorta"],
        "bony_thorax": (structures["Left Clavicle"] | structures["Right Clavicle"] | structures["Left Scapula"]
                        | structures["Right Scapula"] | structures["Spine"]),
    }
