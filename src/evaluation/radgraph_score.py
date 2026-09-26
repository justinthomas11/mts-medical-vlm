"""
Standalone RadGraph F1 scorer, run as a subprocess by src/evaluation/nlg_metrics.py.

RadGraph's bundled AllenNLP code needs transformers < 5, so locally it runs in its own venv
(set RADGRAPH_PYTHON to that interpreter). On Kaggle/Colab (transformers 4.x) it runs directly.

Usage: python src/evaluation/radgraph_score.py <input.json> <output.json>
  input:  {"refs": [...], "hyps": [...], "reward_level": "partial"}
  output: {"radgraph_f1": float, "reward_level": str, "per_report": [...], "transformers": str}
"""

import json
import sys


def main(in_path: str, out_path: str) -> None:
    import transformers
    from radgraph import F1RadGraph

    with open(in_path, encoding="utf-8") as f:
        payload = json.load(f)
    scorer = F1RadGraph(reward_level=payload["reward_level"], model_type="radgraph")
    mean_reward, per_report, _, _ = scorer(hyps=payload["hyps"], refs=payload["refs"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"radgraph_f1": float(mean_reward), "reward_level": payload["reward_level"],
                   "per_report": [float(x) for x in per_report],
                   "transformers": transformers.__version__}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
