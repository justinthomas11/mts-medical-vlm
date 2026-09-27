"""
Unit tests for the train-only hybrid retriever (DR-018).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import ExactIPIndex, HybridRetriever, l2_normalize


def test_l2_normalize_unit_rows():
    x = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(x[0]), 1.0)
    assert np.allclose(x[1], 0.0)


def test_exact_index_matches_cosine():
    rng = np.random.default_rng(0)
    base, queries = rng.normal(size=(20, 8)), rng.normal(size=(3, 8))
    expected = l2_normalize(queries) @ l2_normalize(base).T
    assert np.allclose(ExactIPIndex(base).all_scores(queries), expected, atol=1e-5)


def _toy_retriever():
    uids = [10, 20, 30]
    image = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    text = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return HybridRetriever(uids, image, text)


def test_alpha_one_uses_image_only():
    uids, _ = _toy_retriever().search(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), k=1, alpha=1.0)
    assert uids.tolist() == [[10]]


def test_alpha_zero_uses_text_only():
    uids, _ = _toy_retriever().search(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), k=1, alpha=0.0)
    assert uids.tolist() == [[20]]


def test_scores_sorted_descending_and_k_respected():
    uids, scores = _toy_retriever().search(np.array([[1.0, 0.2]]), np.array([[0.3, 1.0]]), k=3, alpha=0.5)
    assert uids.shape == (1, 3)
    assert np.all(np.diff(scores[0]) <= 1e-9)


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError):
        HybridRetriever([1, 2], np.zeros((2, 4)), np.zeros((3, 4)))


def test_retrieval_label_f1_known_value():
    import pandas as pd

    from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
    from src.pipeline.build_rag import retrieval_label_f1

    ref = pd.DataFrame(0, index=[1, 2], columns=CHEXBERT_LABELS)
    ref.loc[1, "Cardiomegaly"] = 1
    ref.loc[2, "No Finding"] = 1
    ref = ref.rename_axis("uid").reset_index()
    query = np.zeros((1, 14), dtype=np.int8)
    query[0, CHEXBERT_LABELS.index("Cardiomegaly")] = 1
    # retrieved: uid 1 (exact match, F1=1) and uid 2 (disjoint, F1=0) -> mean 0.5
    assert retrieval_label_f1(query, np.array([[1, 2]]), ref) == pytest.approx(0.5)


@pytest.mark.parametrize("split", ["val", "test"])
def test_retrieved_reports_come_from_train_only(split):
    """DR-004: every retrieved report must belong to a train-split patient."""
    import pandas as pd

    path = PROJECT_ROOT / "results" / "s1" / f"retrieval_{split}.csv"
    if not path.exists():
        pytest.skip("retrieval not built yet")
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "iu_xray_processed.csv")
    train = set(df.loc[df["split"] == "train", "uid"])
    queries = set(df.loc[df["split"] == split, "uid"])
    ret = pd.read_csv(path)
    retrieved = set(ret.filter(like="retrieved_uid_").to_numpy().ravel())
    assert retrieved <= train
    assert set(ret["uid"]) == queries
