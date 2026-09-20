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


POST_NEGATION_PATTERNS = [
    r"\b(?:is\s+|are\s+)?(?:not\s+identified|not\s+seen|absent|ruled\s+out|unremarkable|negative|normal|resolved)\b",
]


def is_negated_mention(sentence: str, match_start: int, match_end: int = None) -> bool:
    """
    Checks if a pattern match in a sentence is preceded by a negation pattern
    (across the full preceding clause) or followed by a post-negation pattern.
    """
    preceding = sentence[:match_start].lower()
    for neg in NEGATION_PATTERNS:
        neg_matches = list(re.finditer(neg, preceding))
        if neg_matches:
            last_neg = neg_matches[-1]
            intervening = preceding[last_neg.end():]
            # Negation does not carry over if an adversative boundary separates them
            if not re.search(r"\b(but|however|except|yet)\b", intervening):
                return True

    # Check post-match negation (e.g. "pleural effusion is not identified")
    if match_end is not None:
        following = sentence[match_end:].lower()
        for post_neg in POST_NEGATION_PATTERNS:
            post_match = re.search(post_neg, following)
            if post_match and post_match.start() < 35:
                intervening = following[:post_match.start()]
                if not re.search(r"\b(but|however|and|with)\b", intervening):
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
                    end_pos = match.end()
                    if not is_negated_mention(sentence, start_pos, end_pos):
                        labels[disease] = 1
                        break
                if labels[disease] == 1:
                    break

    return labels


def derive_ground_truth_labels(row: pd.Series) -> Dict[str, Any]:
    """
    Derives unified multi-label ground truth by combining curated NLM MeSH/Problems tags
    with text extraction.
    Reconciled Consensus Definition:
    A study is Normal (No_Finding=1, is_abnormal=0) if and only if MeSH or Problems
    is explicitly tagged as 'normal'. Otherwise, the study is Abnormal (is_abnormal=1,
    No_Finding=0), with disease labels populated from report text and corroborated tags.
    """
    problems_text = str(row.get("Problems", "")).strip().lower()
    mesh_text = str(row.get("MeSH", "")).strip().lower()

    is_normal_mesh = mesh_text == "normal"
    is_normal_prob = problems_text == "normal"
    is_normal_gold = is_normal_mesh or is_normal_prob

    if is_normal_gold:
        labels = {disease: 0 for disease in TARGET_DISEASES}
        labels["No_Finding"] = 1
        labels["is_abnormal"] = 0
        return labels

    # Abnormal study: extract labels from text
    text = f"{str(row.get('findings', ''))} {str(row.get('impression', ''))}"
    labels = extract_labels_from_text(text)

    # Corroborate with structured fields
    for disease, terms in PROBLEMS_MAPPING.items():
        if labels[disease] == 1:
            continue
        for term in terms:
            if term in problems_text or term in mesh_text:
                labels[disease] = 1
                break

    labels["No_Finding"] = 0
    labels["is_abnormal"] = 1
    return labels
