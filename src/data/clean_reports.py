"""
Report text cleaning and standardization module for IU X-Ray dataset.
Handles de-identification tokens (XXXX), whitespace normalization,
section joining, and clinical query generation.
"""

import re
from typing import Dict, Tuple, Optional


# Compiled regex patterns for hospital de-identification normalization
DATE_PATTERN = re.compile(
    r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\b)?\bX+\b(?:\s*[,/-]\s*\bX+\b)+(?:\s*(?:at\s*)?\bX+\s*(?:hours|hrs|am|pm)?)?",
    re.IGNORECASE
)
AGE_PATTERN = re.compile(r"\bX+\s*-\s*year\s*-\s*old\b|\bX+\s*yo\b|\bX+\s*year\s*old\b", re.IGNORECASE)
DOCTOR_PATTERN = re.compile(r"\bDr\.\s*X+\b|\bDoctor\s*X+\b", re.IGNORECASE)
REPEATED_XXXX = re.compile(r"\bX+(?:\s+X+)+\b", re.IGNORECASE)
SINGLE_XXXX = re.compile(r"\bX+\b", re.IGNORECASE)
PUNCT_SPACING = re.compile(r"\s+([.,;:!?])")
MULTIPLE_SPACES = re.compile(r"\s+")


def clean_medical_text(text: Optional[str]) -> str:
    """
    Cleans raw clinical text:
    - Normalizes hospital de-identification tokens (XXXX) into contextual placeholders.
    - Fixes punctuation spacing and normalizes whitespace.
    """
    if text is None or not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    # Normalize contextual de-identification tokens
    cleaned = DATE_PATTERN.sub("[DATE]", cleaned)
    cleaned = AGE_PATTERN.sub("[AGE]", cleaned)
    cleaned = DOCTOR_PATTERN.sub("[DOCTOR]", cleaned)
    cleaned = REPEATED_XXXX.sub("[REDACTED]", cleaned)
    cleaned = SINGLE_XXXX.sub("[REDACTED]", cleaned)

    # Clean whitespace and punctuation
    cleaned = PUNCT_SPACING.sub(r"\1", cleaned)
    cleaned = MULTIPLE_SPACES.sub(" ", cleaned)
    return cleaned.strip()


def build_target_report(findings: str, impression: str) -> Tuple[str, bool]:
    """
    Constructs the generation target report by merging Findings and Impression.
    Returns (target_text, is_valid) where is_valid indicates whether sufficient text exists.
    """
    clean_find = clean_medical_text(findings)
    clean_imp = clean_medical_text(impression)

    if not clean_find and not clean_imp:
        return "", False

    if clean_find and clean_imp:
        target = f"FINDINGS: {clean_find}\nIMPRESSION: {clean_imp}"
    elif clean_find:
        target = f"FINDINGS: {clean_find}"
    else:
        target = f"IMPRESSION: {clean_imp}"

    # Flag records with fewer than 10 characters as invalid targets
    is_valid = len(target.strip()) >= 10
    return target, is_valid


def build_clinical_query(indication: Optional[str], template: str = "Indication: {indication}. Analyze this chest radiograph and provide detailed findings and diagnostic impression.") -> str:
    """
    Constructs a standardized clinical query prompt using patient indication.
    """
    clean_ind = clean_medical_text(indication)
    if not clean_ind or clean_ind.lower() in ["none", "none.", "none available", "none available."]:
        clean_ind = "Routine diagnostic chest examination"
    return template.format(indication=clean_ind)
