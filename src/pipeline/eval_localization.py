"""
S2: Grad-CAM localization evaluation (Pointing Game + Saliency Mass Ratio) (DR-016, DR-019).

For every patient in the split and every CheXbert-positive reference observation that has an
anatomical target (src/xai/anatomy.py), the Grad-CAM heatmap for that class is scored against
the target compartment mask. Heatmaps and masks share the padded-square frame (512 px).

Outputs: results/s2/localization_<split>.csv and results/s2/localization_<split>_summary.json

Usage: python src/pipeline/eval_localization.py --split test [--limit 20 --tag smoke20]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
from src.pipeline.common import load_benchmark, load_configs, load_image, load_reference_labels, path, set_seed
from src.xai.anatomy import CONDITION_TARGETS, SEG_SIZE, AnatomySegmenter, derive_targets
from src.xai.gradcam import gradcam, upsample
from src.xai.head import PatchTokenHead
from src.xai.localization import pointing_game_hit, saliency_mass_ratio, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    dcfg, ecfg = load_configs()
    set_seed(dcfg["project"]["seed"])
    feat_dir = path(ecfg["paths"]["features_dir"])
    index = pd.read_csv(feat_dir / "feature_index.csv")
    tokens = np.load(feat_dir / "clip_patch_tokens.f16.npy", mmap_mode="r")
    ref = load_reference_labels(ecfg).set_index("uid")
    files = load_benchmark(dcfg).set_index("uid")["primary_image_filename"]

    ckpt = torch.load(path(ecfg["paths"]["checkpoints_dir"]) / "xai_head.pt", map_location="cpu")
    head = PatchTokenHead(in_dim=ckpt["in_dim"], hidden=ecfg["xai"]["head_hidden"],
                          dropout=ecfg["xai"]["head_dropout"])
    head.load_state_dict(ckpt["state_dict"])
    head.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    thresholds = np.asarray(ckpt["thresholds"])

    segmenter = AnatomySegmenter()
    rows = np.where(index["split"] == args.split)[0]
    if args.limit:
        rows = rows[: args.limit]

    records = []
    for n, row in enumerate(rows):
        uid = int(index["uid"].iloc[row])
        positives = [c for c in CONDITION_TARGETS if ref.loc[uid, c] == 1]
        if not positives:
            continue
        a = torch.from_numpy(np.asarray(tokens[row], dtype=np.float32))
        with torch.no_grad():
            probs = torch.sigmoid(head(a[None].to(next(head.parameters()).device)))[0].cpu().numpy()
        targets = derive_targets(segmenter.segment(load_image(dcfg, files[uid])), ecfg["xai"]["rim_fraction"])
        for cond in positives:
            c = CHEXBERT_LABELS.index(cond)
            heat = upsample(gradcam(head, a, c), SEG_SIZE)
            mask = targets[CONDITION_TARGETS[cond]]
            records.append({
                "uid": uid, "condition": cond, "target": CONDITION_TARGETS[cond],
                "hit": pointing_game_hit(heat, mask), "smr": saliency_mass_ratio(heat, mask),
                "mask_area_fraction": float(mask.mean()), "head_prob": float(probs[c]),
                "head_predicted_positive": bool(probs[c] >= thresholds[c]),
            })
        if n % 50 == 0:
            print(f"  {n + 1}/{len(rows)} patients, {len(records)} (patient, condition) pairs")

    out_dir = path(ecfg["paths"]["results_dir"]) / "s2"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}{'_' + args.tag if args.tag else ''}"
    pd.DataFrame(records).to_csv(out_dir / f"localization_{suffix}.csv", index=False)
    summary = {"split": args.split, "n_patients_scanned": int(len(rows)),
               "target_definitions": CONDITION_TARGETS, "rim_fraction": ecfg["xai"]["rim_fraction"],
               "gradcam_layer": "vision_tower layer -2 patch tokens (24x24)",
               "all_positive_pairs": summarize(records),
               "head_predicted_positive_pairs": summarize([r for r in records if r["head_predicted_positive"]])}
    with open(out_dir / f"localization_{suffix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    overall = summary["all_positive_pairs"].get("overall", {})
    print(f"Pointing game {overall.get('pointing_game')}, SMR {overall.get('smr')} over {overall.get('n')} pairs")


if __name__ == "__main__":
    main()
