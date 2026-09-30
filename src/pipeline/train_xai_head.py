"""
S2: trains the Grad-CAM classification head on frozen LLaVA-Med CLIP patch tokens (DR-016).

Train split only for fitting; val for epoch selection and per-class thresholds. Test is not read.
Outputs:
  checkpoints/xai_head.pt                (git-ignored)
  results/s2/head_training.json          history, selected epoch, val AUROC/F1, thresholds

Usage: python src/pipeline/train_xai_head.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
from src.pipeline.common import labels_for, load_configs, load_reference_labels, path, set_seed
from src.xai.head import train_head, tune_thresholds


def main():
    dcfg, ecfg = load_configs()
    seed = dcfg["project"]["seed"]
    set_seed(seed)
    xcfg = ecfg["xai"]

    feat_dir = path(ecfg["paths"]["features_dir"])
    index = pd.read_csv(feat_dir / "feature_index.csv")
    tokens = np.load(feat_dir / "clip_patch_tokens.f16.npy", mmap_mode="r")
    ref = load_reference_labels(ecfg)

    tr = np.where(index["split"] == "train")[0]
    va = np.where(index["split"] == "val")[0]
    y_tr = labels_for(ref, index["uid"].values[tr])
    y_va = labels_for(ref, index["uid"].values[va])
    # Load the train/val slices into RAM once (float16: ~3 GB train, ~0.4 GB val).
    x_tr, x_va = np.ascontiguousarray(tokens[tr]), np.ascontiguousarray(tokens[va])
    print(f"Training head on {len(tr)} train patients; selecting on {len(va)} val patients")

    out = train_head(x_tr, y_tr, x_va, y_va, epochs=xcfg["epochs"], lr=xcfg["lr"],
                     weight_decay=xcfg["weight_decay"], batch_size=xcfg["batch_size"], seed=seed)
    p_va = out["predict"](x_va)
    thresholds = tune_thresholds(y_va, p_va)

    per_label = {}
    for c, name in enumerate(CHEXBERT_LABELS):
        has_both = 0 < y_va[:, c].sum() < len(y_va)
        per_label[name] = {
            "train_positives": int(y_tr[:, c].sum()), "val_positives": int(y_va[:, c].sum()),
            "val_auroc": float(roc_auc_score(y_va[:, c], p_va[:, c])) if has_both else None,
            "threshold": float(thresholds[c]),
        }

    ckpt_dir = path(ecfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out["head"].state_dict(), "thresholds": thresholds.tolist(),
                "labels": CHEXBERT_LABELS, "in_dim": int(x_tr.shape[2])}, ckpt_dir / "xai_head.pt")

    res_dir = path(ecfg["paths"]["results_dir"]) / "s2"
    res_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fit_split": "train", "selection_split": "val", "n_train": int(len(tr)), "n_val": int(len(va)),
        "encoder": ecfg["vlm"]["model_id"] + " :: vision_tower layer -2 (frozen)",
        "hyperparameters": xcfg, "seed": seed,
        "best_epoch": out["best_epoch"], "val_macro_auroc": out["best_val_macro_auroc"],
        "val_micro_f1_at_tuned_thresholds": float(f1_score(y_va, p_va >= thresholds, average="micro", zero_division=0)),
        "per_label": per_label, "history": out["history"],
    }
    with open(res_dir / "head_training.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best epoch {out['best_epoch']}: val macro AUROC {out['best_val_macro_auroc']:.4f}. "
          f"Saved {ckpt_dir / 'xai_head.pt'}")


if __name__ == "__main__":
    main()
