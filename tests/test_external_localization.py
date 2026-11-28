import numpy as np

from src.pipeline.eval_external_localization import NIH_TO_HEAD, box_mask, center_prior


def test_box_mask_square_image_scales_to_frame():
    m = box_mask((256, 512, 256, 128), (1024, 1024), frame=512)
    ys, xs = np.where(m)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (256, 319, 128, 255)
    assert m.mean() == (64 * 128) / 512 ** 2


def test_box_mask_accounts_for_square_padding():
    # 1000 x 500 image is padded vertically by 250 px on top; a box at y=0 starts at frame row 128
    m = box_mask((0, 0, 1000, 10), (1000, 500), frame=512)
    ys, _ = np.where(m)
    assert ys.min() == 128


def test_center_prior_peaks_in_middle_and_mapping_targets_head_classes():
    from src.evaluation.chexbert_scorer import CHEXBERT_LABELS
    p = center_prior(64)
    assert np.unravel_index(p.argmax(), p.shape) in {(31, 31), (31, 32), (32, 31), (32, 32)}
    assert set(NIH_TO_HEAD.values()) <= set(CHEXBERT_LABELS)
