"""
Prompt construction for S0 (image + indication query) and S1+ (query conditioned on
retrieved train-split reports). Kept model-agnostic: returns the user-turn text only.
"""

from typing import Sequence

RAG_HEADER = (
    "Reports from similar prior chest radiographs are provided for reference only. "
    "They describe other patients; describe only what is visible in this image."
)


def s0_prompt(clinical_query: str) -> str:
    return str(clinical_query).strip()


def rag_prompt(clinical_query: str, retrieved_reports: Sequence[str]) -> str:
    blocks = [f"Reference report {i + 1}:\n{str(r).strip()}" for i, r in enumerate(retrieved_reports)]
    return f"{RAG_HEADER}\n\n" + "\n\n".join(blocks) + f"\n\n{s0_prompt(clinical_query)}"
