"""
Composite Medical Trustworthiness Score (DR-021). Weights and component definitions are fixed
in configs/experiment_config.yaml before any test-split result is inspected.

  Diagnostic     D = mean(CheXbert micro-F1 over 14 observations, RadGraph F1)
  Reliability    R = mean(1 - label-level ECE, 1 - hallucination rate)
  Explainability E = mean(Pointing Game hit rate, Saliency Mass Ratio); 0 for stages that
                     produce no visual explanation (S0, S1)
  MTS = w_D * D + w_R * R + w_E * E,   w_D + w_R + w_E = 1
"""

import math
from typing import Dict, Optional


def _mean(*xs: float) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    if len(vals) != len(xs):
        raise ValueError(f"MTS component has a missing value: {xs}")
    return sum(vals) / len(vals)


def mts_components(chexbert_micro_f1: float, radgraph_f1: float, ece: float, hallucination_rate: float,
                   pointing_game: Optional[float] = None, smr: Optional[float] = None) -> Dict[str, float]:
    _mean(ece, hallucination_rate)  # raises on a missing value before the 1 - x transform
    diagnostic = _mean(chexbert_micro_f1, radgraph_f1)
    reliability = _mean(1.0 - ece, 1.0 - hallucination_rate)
    explainability = 0.0 if pointing_game is None else _mean(pointing_game, smr)
    return {"diagnostic": diagnostic, "reliability": reliability, "explainability": explainability}


def mts(components: Dict[str, float], weights: Dict[str, float]) -> float:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"MTS weights must sum to 1, got {total}")
    return float(sum(weights[k] * components[k] for k in ("diagnostic", "reliability", "explainability")))
