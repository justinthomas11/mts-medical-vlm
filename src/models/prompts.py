"""
Prompt construction for S0 (image + indication query) and S1+ (query conditioned on
retrieved train-split reports). Kept model-agnostic: returns the user-turn text only.

Candidate query templates are compared on VAL only (src/pipeline/select_prompt.py, DR-024);
the selected id is set in configs/experiment_config.yaml (vlm.prompt_template_id).
"""

from typing import Sequence

RAG_HEADER = (
    "Reports from similar prior chest radiographs are provided for reference only. "
    "They describe other patients; describe only what is visible in this image."
)

PROMPT_TEMPLATES = {
    # Original roadmap template (configs/data_config.yaml text_processing.query_template).
    "baseline_v1": ("Indication: {indication}. Analyze this chest radiograph and provide detailed findings "
                    "and diagnostic impression."),
    # Explicit radiology-report format with an anatomical checklist.
    "report_v2": ("You are a radiologist. Write the radiology report for this frontal chest X-ray.\n"
                  "Clinical indication: {indication}.\n"
                  "Use exactly two sections:\n"
                  "FINDINGS: describe the heart, mediastinum, lungs, pleura and bones.\n"
                  "IMPRESSION: a one-sentence diagnostic summary.\n"
                  "Be concise and state normal findings when the study is normal."),
    # Minimal explicit-format instruction.
    "report_v3": ("Write a concise chest X-ray radiology report with a FINDINGS section and an IMPRESSION "
                  "section. Indication: {indication}."),
    # Round 2 (DR-024): explicit format without the "state normal findings" cue that made report_v2
    # call almost every study normal; asks for abnormalities and normal statements symmetrically.
    "report_v4": ("You are a radiologist. Write the radiology report for this frontal chest X-ray.\n"
                  "Clinical indication: {indication}.\n"
                  "Use exactly two sections:\n"
                  "FINDINGS: for the heart, mediastinum, lungs, pleura and bones, describe any abnormality you "
                  "see, or state that the structure is normal.\n"
                  "IMPRESSION: the main diagnosis or diagnoses."),
    # Round 2 (DR-024): format only, no checklist and no normality cue.
    "report_v5": ("You are a radiologist. Look carefully at this frontal chest X-ray and write its report.\n"
                  "Clinical indication: {indication}.\n"
                  "Write a FINDINGS section describing what you observe, then an IMPRESSION section with "
                  "your conclusion."),
}

MISSING_INDICATION = "not provided"


def build_query(indication, template_id: str) -> str:
    """Fills a template with the cleaned indication; empty/NaN indications become 'not provided'."""
    text = "" if indication is None else str(indication).strip()
    if text.lower() in ("", "nan", "none", "none."):
        text = MISSING_INDICATION
    return PROMPT_TEMPLATES[template_id].format(indication=text.rstrip("."))


def s0_prompt(clinical_query: str) -> str:
    return str(clinical_query).strip()


def rag_prompt(clinical_query: str, retrieved_reports: Sequence[str]) -> str:
    blocks = [f"Reference report {i + 1}:\n{str(r).strip()}" for i, r in enumerate(retrieved_reports)]
    return f"{RAG_HEADER}\n\n" + "\n\n".join(blocks) + f"\n\n{s0_prompt(clinical_query)}"
