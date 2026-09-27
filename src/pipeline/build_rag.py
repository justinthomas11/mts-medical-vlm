"""
S1 retrieval: builds the train-only hybrid index, tunes the visual/text fusion weight alpha on
VAL, and writes top-k retrieved train reports for val and test queries (DR-018).

Tuning objective (val only): mean example-based F1 between the query's CheXbert reference labels
and the CheXbert labels of each retrieved train report.

Usage: python src/pipeline/build_rag.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd

from src.evaluation.clinical_metrics import example_based_f1
from src.pipeline.common import labels_for, load_benchmark, load_configs, load_reference_labels, path
from src.rag.retriever import HybridRetriever


def retrieval_label_f1(query_labels: np.ndarray, retrieved_uids: np.ndarray, ref: pd.DataFrame) -> float:
    k = retrieved_uids.shape[1]
    scores = [example_based_f1(query_labels, labels_for(ref, retrieved_uids[:, j])) for j in range(k)]
    return float(np.mean(scores))


def main():
    dcfg, ecfg = load_configs()
    feat_dir = path(ecfg["paths"]["features_dir"])
    index = pd.read_csv(feat_dir / "feature_index.csv")
    image_emb = np.load(feat_dir / "clip_pooled.npy")
    article_emb = np.load(feat_dir / "medcpt_article.npy")
    query_emb = np.load(feat_dir / "medcpt_query.npy")
    ref = load_reference_labels(ecfg)
    df = load_benchmark(dcfg)
    assert (df["uid"].values == index["uid"].values).all(), "feature_index.csv is stale — re-run extract_features.py"

    rows = {s: np.where(index["split"] == s)[0] for s in ("train", "val", "test")}
    train_uids = index["uid"].values[rows["train"]]
    retriever = HybridRetriever(train_uids, image_emb[rows["train"]], article_emb[rows["train"]])
    k = ecfg["rag"]["k"]

    # Leakage guard (DR-004): the index holds train patients only.
    held_out = set(index["uid"].values[rows["val"]]) | set(index["uid"].values[rows["test"]])
    assert not held_out & set(retriever.uids.tolist()), "val/test patient found in RAG index"

    val_uids = index["uid"].values[rows["val"]]
    val_labels = labels_for(ref, val_uids)
    tuning = []
    for alpha in ecfg["rag"]["alpha_grid"]:
        uids, _ = retriever.search(image_emb[rows["val"]], query_emb[rows["val"]], k=k, alpha=alpha)
        tuning.append({"alpha": alpha, "val_mean_example_f1": retrieval_label_f1(val_labels, uids, ref)})
        print(f"  alpha={alpha:.2f}  val retrieval label F1={tuning[-1]['val_mean_example_f1']:.4f}")
    best = max(tuning, key=lambda t: t["val_mean_example_f1"])
    retriever.alpha = best["alpha"]

    out_dir = path(ecfg["paths"]["results_dir"]) / "s1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        uids, scores = retriever.search(image_emb[rows[split]], query_emb[rows[split]], k=k)
        records = []
        for q, (r_uids, r_scores) in zip(index["uid"].values[rows[split]], zip(uids, scores)):
            rec = {"uid": int(q)}
            for j in range(k):
                rec[f"retrieved_uid_{j + 1}"] = int(r_uids[j])
                rec[f"score_{j + 1}"] = float(r_scores[j])
            records.append(rec)
        pd.DataFrame(records).to_csv(out_dir / f"retrieval_{split}.csv", index=False)

    with open(out_dir / "retrieval_tuning.json", "w") as f:
        json.dump({"index_split": "train", "n_indexed": int(len(train_uids)), "k": k,
                   "selected_alpha": best["alpha"], "selection_split": "val",
                   "objective": "mean example-based F1, CheXbert(query reference) vs CheXbert(retrieved train report)",
                   "grid": tuning}, f, indent=2)
    print(f"Selected alpha={best['alpha']} on val. Wrote retrieval_val.csv / retrieval_test.csv to {out_dir}")


if __name__ == "__main__":
    main()
