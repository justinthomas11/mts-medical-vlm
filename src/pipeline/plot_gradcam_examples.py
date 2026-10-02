"""
Qualitative Grad-CAM figure (val only): for each condition with at least one hit and one miss in
results/s2/localization_val.csv, shows the lowest-uid hit and the lowest-uid miss (deterministic, not
hand-picked): X-ray, Grad-CAM heatmap, target-compartment outline (green) and heatmap peak (star).

Usage: python src/pipeline/plot_gradcam_examples.py
Output: reports/figures/gradcam_val_examples.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple

import numpy as np
import pandas as pd

CONDITION_ORDER = ["Cardiomegaly", "Pleural Effusion", "Enlarged Cardiomediastinum", "Fracture", "Pneumothorax"]


def select_examples(loc: pd.DataFrame) -> List[Tuple[str, str, int]]:
    """(condition, 'hit'|'miss', uid) — lowest-uid hit and miss per condition, conditions with both only."""
    picks = []
    for cond in CONDITION_ORDER:
        rows = loc[loc["condition"] == cond]
        hits, misses = rows[rows["hit"]], rows[~rows["hit"]]
        if len(hits) and len(misses):
            picks.append((cond, "hit", int(hits["uid"].min())))
            picks.append((cond, "miss", int(misses["uid"].min())))
    return picks


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
    from src.pipeline.common import load_benchmark, load_configs, load_image, path
    from src.xai.anatomy import CONDITION_TARGETS, SEG_SIZE, AnatomySegmenter, derive_targets, pad_to_square_gray
    from src.xai.gradcam import gradcam, upsample
    from src.xai.head import PatchTokenHead

    dcfg, ecfg = load_configs()
    loc = pd.read_csv(path(ecfg["paths"]["results_dir"]) / "s2" / "localization_val.csv")
    picks = select_examples(loc)
    feat_dir = path(ecfg["paths"]["features_dir"])
    index = pd.read_csv(feat_dir / "feature_index.csv")
    assert set(index.set_index("uid").loc[[u for _, _, u in picks], "split"]) == {"val"}, "val examples only"
    tokens = np.load(feat_dir / "clip_patch_tokens.f16.npy", mmap_mode="r")
    files = load_benchmark(dcfg).set_index("uid")["primary_image_filename"]

    ckpt = torch.load(path(ecfg["paths"]["checkpoints_dir"]) / "xai_head.pt", map_location="cpu")
    head = PatchTokenHead(in_dim=ckpt["in_dim"], hidden=ecfg["xai"]["head_hidden"], dropout=ecfg["xai"]["head_dropout"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    seg = AnatomySegmenter()

    n_cols = 2
    n_rows = len(picks) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for ax, (cond, kind, uid) in zip(axes.ravel(), picks):
        row = int(np.where(index["uid"] == uid)[0][0])
        img = load_image(dcfg, files[uid])
        square = np.asarray(pad_to_square_gray(img).resize((SEG_SIZE, SEG_SIZE)))
        mask = derive_targets(seg.segment(img), ecfg["xai"]["rim_fraction"])[CONDITION_TARGETS[cond]]
        heat = upsample(gradcam(head, torch.from_numpy(np.asarray(tokens[row], dtype=np.float32)),
                                CHEXBERT_LABELS.index(cond)), SEG_SIZE)
        rec = loc[(loc["uid"] == uid) & (loc["condition"] == cond)].iloc[0]
        ax.imshow(square, cmap="gray")
        ax.imshow(heat, cmap="jet", alpha=0.35)
        ax.contour(mask, colors="lime", linewidths=1.2)
        y, x = np.unravel_index(np.argmax(heat), heat.shape)
        ax.plot(x, y, marker="*", color="white", markersize=14, markeredgecolor="black")
        ax.set_title(f"{cond} — {kind}\nuid {uid}, SMR {rec['smr']:.2f} (chance {rec['mask_area_fraction']:.2f})",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM on frozen LLaVA-Med CLIP features (val). Green: target compartment; star: peak.",
                 fontsize=10)
    fig.tight_layout()
    out = PROJECT_ROOT / "reports" / "figures" / "gradcam_val_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Saved {out} ({len(picks)} panels: {picks})")


if __name__ == "__main__":
    main()
