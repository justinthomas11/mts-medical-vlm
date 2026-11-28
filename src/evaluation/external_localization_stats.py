"""
Paired statistics for the external NIH box test (DR-034), from results/external/nih_bbox_localization.csv.

Boxes in one image share a heatmap, so the bootstrap resamples IMAGES (B = 2,000, seed 42). For each
subset (all boxes; all boxes except Cardiomegaly, whose box is the central heart region a centre prior
hits by construction) it reports the Pointing Game of each method and the paired differences
gradcam_hidden - center_prior, gradcam_hidden - gradcam_tokens, and gradcam_hidden - uniform chance
(expected hit rate of a uniform heatmap = box area fraction).

Output: results/external/nih_bbox_stats.json
Usage:  python src/evaluation/external_localization_stats.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Dict

import numpy as np
import pandas as pd


def paired_image_bootstrap(wide: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 42) -> Dict:
    """wide: one row per box with columns image, a, b (hit indicators or chance). Delta = mean(a) - mean(b)."""
    images = wide["image"].unique()
    groups = {im: idx for im, idx in wide.groupby("image").indices.items()}
    va, vb = wide[a].to_numpy(float), wide[b].to_numpy(float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([groups[im] for im in rng.choice(images, len(images), replace=True)])
        deltas[k] = va[idx].mean() - vb[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": float(va.mean() - vb.mean()), "ci95": [float(lo), float(hi)], "p_value": float(min(p, 1.0)),
            "n_boxes": int(len(wide)), "n_images": int(len(images))}


def main():
    from src.pipeline.common import load_configs, path

    _, ecfg = load_configs()
    out_dir = path(ecfg["paths"]["results_dir"]) / "external"
    d = pd.read_csv(out_dir / "nih_bbox_localization.csv")
    keys = ["image", "nih_label", "mask_area_fraction"]
    wide = d.pivot_table(index=keys, columns="method", values="hit", aggfunc="first").reset_index()
    wide.columns.name = None
    wide["uniform_chance"] = wide["mask_area_fraction"]
    out = {"bootstrap": "images resampled, B=2000, seed 42", "subsets": {}}
    for name, sub in {"all_boxes": wide, "excluding_cardiomegaly": wide[wide["nih_label"] != "Cardiomegaly"]}.items():
        sub = sub.reset_index(drop=True)
        out["subsets"][name] = {
            "pointing_game": {m: float(sub[m].mean()) for m in ("gradcam_hidden", "gradcam_tokens", "center_prior",
                                                                "uniform_chance")},
            "hidden_minus_center": paired_image_bootstrap(sub, "gradcam_hidden", "center_prior"),
            "hidden_minus_tokens": paired_image_bootstrap(sub, "gradcam_hidden", "gradcam_tokens"),
            "hidden_minus_chance": paired_image_bootstrap(sub, "gradcam_hidden", "uniform_chance"),
        }
    (out_dir / "nih_bbox_stats.json").write_text(json.dumps(out, indent=2))
    for name, s in out["subsets"].items():
        print(name, {k: round(v, 3) for k, v in s["pointing_game"].items()})
        for c in ("hidden_minus_center", "hidden_minus_tokens", "hidden_minus_chance"):
            r = s[c]
            print(f"  {c:20s} {r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p={r['p_value']:.3f}")


if __name__ == "__main__":
    main()
