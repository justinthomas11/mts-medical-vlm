"""
Builds the S0–S3 ablation table and MTS (DR-021) from saved per-stage metric files.

Stage composition (text for S2/S3 is the S1 text; DR-016):
  S0 = metrics_<split>.json in results/s0                         (no confidence, no explanation)
  S1 = metrics_<split>.json in results/s1                         (no confidence, no explanation)
  S2 = S1 text metrics + Grad-CAM localization (results/s2/localization_<split>_summary.json)
  S3 = metrics_<split>_s3.json in results/s1 (sample-agreement confidence, review flag) + S2 localization

Outputs: results/ablation/ablation_<split>.csv and ablation_<split>.md

Usage: python src/evaluation/build_ablation.py --split test
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, Optional

import pandas as pd

from src.evaluation.mts import mts, mts_components


def stage_row(stage: str, m: Dict, loc: Optional[Dict], weights: Dict[str, Dict[str, float]]) -> Dict:
    pg = smr = None
    if loc is not None:
        overall = loc["all_positive_pairs"]["overall"]
        pg, smr = overall["pointing_game"], overall["smr"]
    comps = mts_components(m["clinical"]["micro_14"]["f1"], m["radgraph"]["radgraph_f1"], m["ece"],
                           m["hallucination_rate"], pointing_game=pg, smr=smr)
    row = {
        "stage": stage, "n_reports": m["n_reports"],
        "bleu_1": m["lexical"]["bleu_1"], "bleu_4": m["lexical"]["bleu_4"], "rouge_l": m["lexical"]["rouge_l"],
        "chexbert_micro_f1": m["clinical"]["micro_14"]["f1"], "chexbert_macro_f1": m["clinical"]["macro_14"]["f1"],
        "chexbert5_micro_f1": m["clinical"]["micro_5"]["f1"], "radgraph_f1": m["radgraph"]["radgraph_f1"],
        "hallucination_rate": m["hallucination_rate"], "ece": m["ece"],
        "empty_report_rate": m.get("empty_report_rate"),
        "pointing_game": pg, "smr": smr,
        "review_flag_rate": m.get("review_flag", {}).get("flag_rate"),
        "micro_f1_unflagged": m.get("review_flag", {}).get("micro_f1_unflagged"),
        "aurc": m.get("review_flag", {}).get("aurc"),
        "latency_s_mean": m.get("latency_s_mean"),
        **comps,
    }
    for name, w in weights.items():
        row[f"mts_{name}"] = mts(comps, w)
    return row


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Pipe table without the optional `tabulate` dependency; missing values render as '—'."""
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_table(metrics: Dict[str, Dict], localization: Optional[Dict], weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """metrics: {'s0': ..., 's1': ..., 's3': ...} parsed metric dicts; localization: S2 summary dict."""
    rows = [stage_row("S0", metrics["s0"], None, weights),
            stage_row("S1 (+RAG)", metrics["s1"], None, weights)]
    if localization is not None:
        rows.append(stage_row("S2 (+Grad-CAM)", metrics["s1"], localization, weights))
        if "s3" in metrics:
            rows.append(stage_row("S3 (+Uncertainty)", metrics["s3"], localization, weights))
    return pd.DataFrame(rows)


def main():
    from src.pipeline.common import load_configs, path

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--tag", default="", help="suffix used for the generation files, e.g. smoke20")
    args = parser.parse_args()

    _, ecfg = load_configs()
    res = path(ecfg["paths"]["results_dir"])
    suffix = f"{args.split}{'_' + args.tag if args.tag else ''}"

    def read(p: Path) -> Dict:
        if not p.exists():
            raise SystemExit(f"Missing {p} — run the corresponding stage/evaluation first. No values are filled in.")
        with open(p) as f:
            return json.load(f)

    metrics = {"s0": read(res / "s0" / f"metrics_{suffix}.json"), "s1": read(res / "s1" / f"metrics_{suffix}.json")}
    s3_path = res / "s1" / f"metrics_{suffix}_s3.json"
    if s3_path.exists():
        metrics["s3"] = read(s3_path)
    loc_path = res / "s2" / f"localization_{args.split}_summary.json"
    localization = read(loc_path) if loc_path.exists() else None

    weights = {"primary": ecfg["mts"]["weights"], **ecfg["mts"]["sensitivity_weights"]}
    table = build_table(metrics, localization, weights)

    out = res / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"ablation_{suffix}.csv", index=False)
    cols = ["stage", "n_reports", "bleu_4", "rouge_l", "chexbert_micro_f1", "chexbert_macro_f1", "radgraph_f1",
            "hallucination_rate", "ece", "empty_report_rate", "pointing_game", "smr", "review_flag_rate",
            "diagnostic", "reliability", "explainability"] + [c for c in table.columns if c.startswith("mts_")]
    with open(out / f"ablation_{suffix}.md", "w", encoding="utf-8") as f:
        f.write(f"# S0–S3 ablation ({suffix})\n\nGenerated by src/evaluation/build_ablation.py from saved metric files. "
                f"MTS weights (DR-021): {weights['primary']}.\n\n")
        f.write(to_markdown(table[cols]))
        f.write("\n")
    print(table[["stage"] + [c for c in table.columns if c.startswith("mts_")]].to_string(index=False))
    print(f"Saved {out / f'ablation_{suffix}.csv'}")


if __name__ == "__main__":
    main()
