"""
Caches frozen-encoder features for every benchmark patient (one pass, reused by S1 and S2):
  - clip_pooled.npy            (N, 1024)       LLaVA-Med CLIP pooled embedding (S1 visual channel)
  - clip_patch_tokens.f16      (N, 576, 1024)  layer -2 patch tokens, float16 memmap (S2 head / Grad-CAM)
  - medcpt_article.npy         (N, 768)        indication + report, article encoder (S1 index; train rows used)
  - medcpt_query.npy           (N, 768)        indication, query encoder (S1 queries)
  - feature_index.csv          uid / split / row order shared by all arrays

Usage: python src/pipeline/extract_features.py [--only clip|medcpt]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import torch

from src.pipeline.common import load_benchmark, load_configs, load_image, path, set_seed

PATCH_SHAPE = (576, 1024)


def extract_clip(df, dcfg, ecfg, out_dir: Path, batch_size: int = 16) -> None:
    from src.models.vision_encoder import LlavaMedVisionEncoder

    encoder = LlavaMedVisionEncoder(ecfg["vlm"]["model_id"])
    n = len(df)
    pooled = np.zeros((n, 1024), dtype=np.float32)
    tokens = np.lib.format.open_memmap(out_dir / "clip_patch_tokens.f16.npy", mode="w+",
                                       dtype=np.float16, shape=(n, *PATCH_SHAPE))
    files = df["primary_image_filename"].tolist()
    for start in range(0, n, batch_size):
        images = [load_image(dcfg, f) for f in files[start: start + batch_size]]
        out = encoder(encoder.preprocess(images))
        pooled[start: start + len(images)] = out["pooled"].float().cpu().numpy()
        tokens[start: start + len(images)] = out["patch_tokens"].to(torch.float16).cpu().numpy()
        if (start // batch_size) % 20 == 0:
            print(f"  CLIP {start + len(images)}/{n}")
    tokens.flush()
    np.save(out_dir / "clip_pooled.npy", pooled)


def extract_medcpt(df, ecfg, out_dir: Path) -> None:
    from src.rag.text_encoder import MedCPTEncoder

    enc = MedCPTEncoder()
    indications = df["clean_indication"].fillna("").tolist()
    np.save(out_dir / "medcpt_article.npy", enc.encode_articles(indications, df["target_report"].fillna("").tolist()))
    np.save(out_dir / "medcpt_query.npy", enc.encode_queries(indications))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["clip", "medcpt"], default=None)
    args = parser.parse_args()

    dcfg, ecfg = load_configs()
    set_seed(dcfg["project"]["seed"])
    df = load_benchmark(dcfg)
    out_dir = path(ecfg["paths"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df[["uid", "split"]].to_csv(out_dir / "feature_index.csv", index=False)
    print(f"Extracting features for {len(df)} benchmark patients -> {out_dir}")

    if args.only in (None, "clip"):
        extract_clip(df, dcfg, ecfg, out_dir)
    if args.only in (None, "medcpt"):
        extract_medcpt(df, ecfg, out_dir)

    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump({"vision_encoder": ecfg["vlm"]["model_id"] + " :: vision_tower (layer -2 patches, pooled CLS)",
                   "text_encoders": [ecfg["rag"]["medcpt_query_encoder"], ecfg["rag"]["medcpt_article_encoder"]],
                   "n_patients": len(df)}, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
