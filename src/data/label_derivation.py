"""
Label derivation module for IU X-Ray reports.
Extracts standard 14 thoracic disease categories using clinical rule-based parsing,
negation detection, and corroboration against MeSH / Problems fields.
"""

import re
from typing import Dict, List, Set, Any
import pandas as pd


# Standard thoracic disease categories
TARGET_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Fracture",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumothorax"
]

# Negation prefixes and contexts
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bno\s+evidence\s+of\b",
    r"\bwithout\b",
    r"\bfree\s+of\b",
    r"\bclear\s+of\b",
    r"\bnegative\s+for\b",
    r"\bnot\s+identified\b",
    r"\bunremarkable\b",
    r"\bnormal\b",
    r"\bresolved\b",
    r"\brule\s+out\b",
    r"\br/o\b"
]

# Regex definitions for positive clinical findings
DISEASE_REGEX = {
    "Atelectasis": [
        r"\batelecta(?:sis|tic)\b",
        r"\bcollapse\b",
        r"\bvolume\s+loss\b"
    ],
    "Cardiomegaly": [
        r"\bcardiomegal(?:y|ic)\b",
        r"\benlarged\s+(?:cardiac|heart)\b",
        r"\bheart\s+size\s+is\s+enlarged\b",
        r"\bprominent\s+cardiac\s+silhouette\b"
    ],
    "Consolidation": [
        r"\bconsolidation\b",
        r"\bair\s+space\s+(?:disease|opacity|opacities)\b",
        r"\bairspace\s+(?:disease|opacity|opacities|consolidation)\b",
        r"\blobar\s+opacity\b"
    ],
    "Edema": [
        r"\bedema\b",
        r"\bvascular\s+congestion\b",
        r"\bfluid\s+overload\b",
        r"\bpulmonary\s+congestion\b"
    ],
    "Effusion": [
        r"\beffusion\b",
        r"\bpleural\s+fluid\b",
        r"\bfluid\s+in\s+the\s+pleural\s+space\b",
        r"\bblunting\s+of\s+(?:the\s+)?costophrenic\b"
    ],
    "Emphysema": [
        r"\bemphysema\b",
        r"\bbullous\b",
        r"\bhyperinflat(?:ed|ion)\b",
        r"\bhyperdistent(?:ion|ed)\b"
    ],
    "Fibrosis": [
        r"\bfibro(?:sis|tic)\b",
        r"\binterstitial\s+marking\b",
        r"\breticular\s+opacit(?:y|ies)\b",
        r"\bscarring\b",
        r"\bcicatrix\b"
    ],
    "Fracture": [
        r"\bfracture\b",
        r"\bbroken\s+rib\b",
        r"\bcallus\b"
    ],
    "Hernia": [
        r"\bhernia\b",
        r"\bhiatal\s+hernia\b"
    ],
    "Infiltration": [
        r"\binfiltrat(?:e|es|ion)\b"
    ],
    "Mass": [
        r"\bmass\b",
        r"\bmasses\b",
        r"\bneoplasm\b",
        r"\btumor\b"
    ],
    "Nodule": [
        r"\bnodule\b",
        r"\bnodules\b",
        r"\bgranuloma\b",
        r"\bcalcified\s+granuloma\b"
    ],
    "Pleural_Thickening": [
        r"\bpleural\s+thickening\b",
        r"\bapical\s+thickening\b",
        r"\bpleural\s+calcification\b",
        r"\bfibrothorax\b"
    ],
    "Pneumothorax": [
        r"\bpneumothorax\b",
        r"\bpneumothoraces\b"
    ]
}

# Concept mapping for Problems / MeSH terms
PROBLEMS_MAPPING = {
    "Atelectasis": ["pulmonary atelectasis", "atelectasis", "hypoinflation"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Consolidation": ["consolidation", "pneumonia"],
    "Edema": ["edema", "pulmonary edema", "congestion"],
    "Effusion": ["pleural effusion", "effusion", "pleural fluid"],
    "Emphysema": ["pulmonary disease, chronic obstructive", "emphysema", "bullous emphysema"],
    "Fibrosis": ["pulmonary fibrosis", "fibrosis", "cicatrix"],
    "Fracture": ["fracture", "fractures, bone"],
    "Hernia": ["hernia", "hernia, hiatal", "hiatal hernia"],
    "Infiltration": ["infiltration", "infiltrate"],
    "Mass": ["mass", "neoplasm", "thoracic neoplasm"],
    "Nodule": ["nodule", "calcified granuloma", "granuloma", "multiple pulmonary nodules"],
    "Pleural_Thickening": ["thickening", "pleural thickening", "calcinosis/pleura"],
    "Pneumothorax": ["pneumothorax"]
}


def is_negated_mention(sentence: str, match_start: int) -> bool:
    """
    Checks if a pattern match in a sentence is preceded by a negation within a 40-character window.
    """
    window = sentence[max(0, match_start - 50):match_start].lower()
    for neg in NEGATION_PATTERNS:
        if re.search(neg, window):
            return True
    return False


def extract_labels_from_text(text: str) -> Dict[str, int]:
    """
    Extracts binary disease labels from clinical report text using negation-aware regex.
    """
    labels = {disease: 0 for disease in TARGET_DISEASES}
    if not text or not isinstance(text, str):
        return labels

    sentences = re.split(r"[.!?]\s+|\n+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for disease, patterns in DISEASE_REGEX.items():
            if labels[disease] == 1:
                continue

            for pattern in patterns:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    start_pos = match.start()
                    if not is_negated_mention(sentence, start_pos):
                        labels[disease] = 1
                        break
                if labels[disease] == 1:
                    break

    return labels


def derive_ground_truth_labels(row: pd.Series) -> Dict[str, Any]:
    """
    Derives unified multi-label ground truth by combining text extraction
    with normalized MeSH and Problems fields.
    """
    text = f"{str(row.get('findings', ''))} {str(row.get('impression', ''))}"
    labels = extract_labels_from_text(text)

    problems_text = str(row.get("Problems", "")).lower()
    mesh_text = str(row.get("MeSH", "")).lower()

    # Corroborate with structured fields
    for disease, terms in PROBLEMS_MAPPING.items():
        if labels[disease] == 1:
            continue
        for term in terms:
            if term in problems_text or term in mesh_text:
                labels[disease] = 1
                break

    # Normal vs Abnormal determination
    is_normal_mesh = "normal" in mesh_text and len(mesh_text.split(";")) == 1
    has_positive_disease = any(val == 1 for val in labels.values())

    if has_positive_disease:
        labels["No_Finding"] = 0
        labels["is_abnormal"] = 1
    else:
        labels["No_Finding"] = 1
        # If MeSH or Problems explicitly indicates abnormal findings outside standard 14, mark abnormal
        if (problems_text and "normal" not in problems_text) or (mesh_text and not is_normal_mesh):
            labels["is_abnormal"] = 1
        else:
            labels["is_abnormal"] = 0

    return labels
