"""
Localization metrics for Grad-CAM heatmaps against anatomical target masks (DR-016, DR-019).

  Pointing Game hit: the heatmap's arg-max pixel lies inside the target mask.
  Saliency Mass Ratio (SMR): share of total heatmap mass inside the target mask.
  Chance baseline: the mask's share of the image area (what a uniform heatmap scores).
"""

from typing import Dict, Sequence

import numpy as np


def pointing_game_hit(heatmap: np.ndarray, mask: np.ndarray) -> bool:
    if heatmap.shape != mask.shape:
        raise ValueError(f"heatmap {heatmap.shape} vs mask {mask.shape}")
    if heatmap.max() <= 0:
        return False
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(mask[y, x])


def saliency_mass_ratio(heatmap: np.ndarray, mask: np.ndarray) -> float:
    h = np.clip(heatmap, 0, None)
    total = h.sum()
    # An all-zero heatmap points nowhere: it scores 0 rather than being dropped from the average.
    return float(h[mask].sum() / total) if total > 0 else 0.0


def wilson_interval(hits: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def summarize(records: Sequence[Dict]) -> Dict:
    """records: dicts with keys condition, hit, smr, mask_area_fraction."""
    out = {}
    groups = {"overall": list(records)}
    for r in records:
        groups.setdefault(r["condition"], []).append(r)
    for name, rs in groups.items():
        n, hits = len(rs), sum(r["hit"] for r in rs)
        out[name] = {
            "n": n,
            "pointing_game": hits / n if n else float("nan"),
            "pointing_game_ci95": wilson_interval(hits, n),
            "smr": float(np.mean([r["smr"] for r in rs])) if n else float("nan"),
            "chance_area_fraction": float(np.mean([r["mask_area_fraction"] for r in rs])) if n else float("nan"),
        }
    return out
