"""
Lexical (BLEU-1..4, ROUGE-L) and factual (RadGraph F1) report-generation metrics.

BLEU and ROUGE-L use the `pycocoevalcap` implementations that radiology report-generation
papers report, with lowercase whitespace/punctuation tokenisation applied identically to
reference and generated text.
"""

import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def tokenize_report(text: str) -> str:
    """Lowercases, drops section headers and returns space-joined word/punctuation tokens."""
    text = str(text).lower().replace("findings:", " ").replace("impression:", " ")
    return " ".join(_TOKEN_RE.findall(text))


def lexical_metrics(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge

    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    gts = {i: [tokenize_report(r)] for i, r in enumerate(refs)}
    # pycocoevalcap asserts non-empty hypotheses; an empty generation scores as a single dot.
    res = {i: [tokenize_report(h) or "."] for i, h in enumerate(hyps)}

    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    rouge_l, _ = Rouge().compute_score(gts, res)
    out = {f"bleu_{n + 1}": float(b) for n, b in enumerate(bleu)}
    out["rouge_l"] = float(rouge_l)
    return out


def radgraph_f1(refs: Sequence[str], hyps: Sequence[str], reward_level: str = "partial") -> Dict:
    """RadGraph entity/relation F1 (Delbrouck et al.). Returns mean score and per-report scores.

    Runs src/evaluation/radgraph_score.py in a subprocess using $RADGRAPH_PYTHON (a venv with
    transformers < 5) when set, otherwise the current interpreter.
    """
    import json
    import os
    import subprocess
    import tempfile

    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    python = os.environ.get("RADGRAPH_PYTHON", sys.executable)
    script = PROJECT_ROOT / "src" / "evaluation" / "radgraph_score.py"
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.json", Path(tmp) / "out.json"
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump({"refs": list(refs), "hyps": list(hyps), "reward_level": reward_level}, f)
        proc = subprocess.run([python, str(script), str(in_path), str(out_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"RadGraph scorer failed ({python}): {proc.stderr[-2000:]}")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
