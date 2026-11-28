"""
External localization test (DR-034): Grad-CAM of the IU-trained S2 head against radiologist-drawn boxes
on a second hospital's images (NIH ChestX-ray14 BBox_List_2017, 984 boxes on 880 images, CC0).

Nothing is tuned here. The head, CAM layers, metrics and chance definition are those of S2 (DR-016,
DR-029); only the targets change from anatomical compartments to drawn lesion boxes. The NIH-to-head
label mapping is fixed below before any result is computed. Two reference heatmaps are scored too:
  uniform : chance = box area fraction (what a uniform heatmap scores on both metrics)
  center  : an isotropic Gaussian at the image centre (sigma = 1/4 side), a content-free anatomy prior

Outputs: results/external/nih_bbox_localization.csv and nih_bbox_localization_summary.json

Usage: python src/pipeline/eval_external_localization.py [--limit 20]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# NIH finding label -> CheXbert class of the S2 head (fixed before evaluation, DR-034)
NIH_TO_HEAD: Dict[str, str] = {
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pneumonia": "Pneumonia",
    "Infiltrate": "Lung Opacity",
    "Mass": "Lung Lesion",
    "Nodule": "Lung Lesion",
}
FRAME = 512  # heatmap / mask frame, as in S2


def box_mask(box: Tuple[float, float, float, float], orig_size: Tuple[int, int], frame: int = FRAME) -> np.ndarray:
    """Box (x, y, w, h) in original pixels -> boolean mask in the padded-square frame used by the encoder."""
    w0, h0 = orig_size
    side = max(w0, h0)
    ox, oy = (side - w0) // 2, (side - h0) // 2       # pad_to_square centres the image
    s = frame / side
    x, y, w, h = box
    x0, y0 = int(np.floor((x + ox) * s)), int(np.floor((y + oy) * s))
    x1, y1 = int(np.ceil((x + w + ox) * s)), int(np.ceil((y + h + oy) * s))
    m = np.zeros((frame, frame), dtype=bool)
    m[max(y0, 0):min(y1, frame), max(x0, 0):min(x1, frame)] = True
    return m


def center_prior(frame: int = FRAME) -> np.ndarray:
    yy, xx = np.mgrid[0:frame, 0:frame]
    c, sigma = (frame - 1) / 2, frame / 4
    return np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2))


def main():
    import torch
    from PIL import Image

    from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
    from src.models.vision_encoder import LlavaMedVisionEncoder
    from src.pipeline.common import load_configs, path, set_seed
    from src.xai.gradcam import gradcam, upsample
    from src.xai.head import PatchTokenHead
    from src.xai.localization import pointing_game_hit, saliency_mass_ratio, summarize

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N images (smoke test)")
    args = ap.parse_args()

    dcfg, ecfg = load_configs()
    set_seed(dcfg["project"]["seed"])
    data_dir = path(dcfg["paths"]["nih_bbox_dir"])
    boxes = pd.read_csv(data_dir / "BBox_List_2017.csv").iloc[:, :6]
    boxes.columns = ["image", "label", "x", "y", "w", "h"]
    boxes = boxes[boxes["label"].isin(NIH_TO_HEAD)]
    images = sorted(boxes["image"].unique())
    if args.limit:
        images = images[: args.limit]

    ckpt = torch.load(path(ecfg["paths"]["checkpoints_dir"]) / "xai_head.pt", map_location="cpu")
    head = PatchTokenHead(in_dim=ckpt["in_dim"], hidden=ecfg["xai"]["head_hidden"], dropout=ecfg["xai"]["head_dropout"])
    head.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head.to(device).eval()
    thresholds = np.asarray(ckpt["thresholds"])
    encoder = LlavaMedVisionEncoder(ecfg["vlm"]["model_id"])
    prior = center_prior()

    records, missing = [], []
    for n, name in enumerate(images):
        f = data_dir / "images" / name
        if not f.exists():
            missing.append(name)
            continue
        img = Image.open(f)
        size = img.size
        tokens = encoder(encoder.preprocess([img]))["patch_tokens"][0].float().cpu()
        with torch.no_grad():
            probs = torch.sigmoid(head(tokens[None].to(device)))[0].cpu().numpy()
        cams = {}
        for _, b in boxes[boxes["image"] == name].iterrows():
            cls = NIH_TO_HEAD[b["label"]]
            c = CHEXBERT_LABELS.index(cls)
            mask = box_mask((b["x"], b["y"], b["w"], b["h"]), size)
            base = {"image": name, "nih_label": b["label"], "condition": b["label"], "head_class": cls,
                    "mask_area_fraction": float(mask.mean()), "head_prob": float(probs[c]),
                    "head_predicted_positive": bool(probs[c] >= thresholds[c])}
            for layer in ("tokens", "hidden"):
                if (layer, c) not in cams:
                    cams[(layer, c)] = upsample(gradcam(head, tokens, c, layer=layer), FRAME)
                heat = cams[(layer, c)]
                records.append({**base, "method": f"gradcam_{layer}", "hit": pointing_game_hit(heat, mask),
                                "smr": saliency_mass_ratio(heat, mask)})
            records.append({**base, "method": "center_prior", "hit": pointing_game_hit(prior, mask),
                            "smr": saliency_mass_ratio(prior, mask)})
        if n % 100 == 0:
            print(f"  {n + 1}/{len(images)} images", flush=True)

    out_dir = path(ecfg["paths"]["results_dir"]) / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{args.limit}" if args.limit else ""
    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"nih_bbox_localization{tag}.csv", index=False)
    summary = {"dataset": "NIH ChestX-ray14 BBox_List_2017 (CC0)", "n_images": len(images) - len(missing),
               "n_missing_images": len(missing), "label_mapping": NIH_TO_HEAD, "frame": FRAME,
               "head": "IU-trained S2 head (train split only), unchanged",
               "chance": "mean box area fraction (uniform heatmap)",
               "methods": {}}
    for m, g in df.groupby("method"):
        recs = g.to_dict("records")
        summary["methods"][m] = {"all_boxes": summarize(recs),
                                 "head_predicted_positive": summarize([r for r in recs if r["head_predicted_positive"]])}
    with open(out_dir / f"nih_bbox_localization{tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    for m, s in summary["methods"].items():
        o = s["all_boxes"]["overall"]
        print(f"{m:16s} PG {o['pointing_game']:.3f} [{o['pointing_game_ci95'][0]:.3f}, {o['pointing_game_ci95'][1]:.3f}] "
              f"SMR {o['smr']:.3f} chance {o['chance_area_fraction']:.3f} (n={o['n']})")


if __name__ == "__main__":
    main()
