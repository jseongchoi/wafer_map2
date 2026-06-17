import numpy as np

from wafermap.data import PATTERN_CLASSES, SyntheticSample
from wafermap.reporting import summarize_sample_defects


def test_defect_summary_reports_clock_position_and_stby_overlap():
    shape = (24, 24)
    masks = np.zeros((len(PATTERN_CLASSES), *shape), dtype=np.uint8)
    scratch_idx = PATTERN_CLASSES.index("scratch")
    masks[scratch_idx, 1:4, 11:14] = 1
    stby_idx = PATTERN_CLASSES.index("stby_pattern")
    masks[stby_idx, 2:4, 12:14] = 1
    sample = SyntheticSample(
        sample_id="unit",
        severity=np.zeros(shape, dtype=np.uint8),
        wafer_mask=np.ones(shape, dtype=np.uint8),
        valid_test_mask=np.ones(shape, dtype=np.uint8),
        stby_mask=masks[stby_idx].copy(),
        pattern_masks=masks,
        pattern_intensity=masks.astype(np.float32),
        chip_index=np.zeros(shape, dtype=np.int32),
        metadata={"sample_id": "unit"},
    )

    summaries = summarize_sample_defects(sample, min_pixel_ratio=0.0)
    scratch = [item for item in summaries if item.class_name == "scratch"][0]
    assert scratch.location_label == "12:00"
    assert scratch.stby_overlap_ratio > 0
    assert scratch.feature_key == "scratch__1200__middle"
