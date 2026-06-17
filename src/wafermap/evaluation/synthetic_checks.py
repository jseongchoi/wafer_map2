"""Internal consistency checks for synthetic wafer samples."""

from __future__ import annotations

import numpy as np

from wafermap.data import PATTERN_CLASSES, SyntheticSample


def _is_binary(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    return int(values.min()) >= 0 and int(values.max()) <= 1


def validate_synthetic_sample(sample: SyntheticSample) -> list[str]:
    """Return validation errors. Empty list means the sample is internally valid."""

    errors: list[str] = []
    shape = sample.severity.shape
    if sample.wafer_mask.shape != shape:
        errors.append("wafer_mask shape does not match severity")
    if sample.valid_test_mask.shape != shape:
        errors.append("valid_test_mask shape does not match severity")
    if sample.stby_mask.shape != shape:
        errors.append("stby_mask shape does not match severity")
    if sample.pattern_masks.shape != (len(PATTERN_CLASSES), *shape):
        errors.append("pattern_masks shape does not match pattern classes and severity")
    if sample.pattern_intensity.shape != (len(PATTERN_CLASSES), *shape):
        errors.append("pattern_intensity shape does not match pattern classes and severity")
    if sample.chip_index.shape != shape:
        errors.append("chip_index shape does not match severity")
    if not _is_binary(sample.wafer_mask):
        errors.append("wafer_mask must be binary")
    if not _is_binary(sample.valid_test_mask):
        errors.append("valid_test_mask must be binary")
    if not _is_binary(sample.stby_mask):
        errors.append("stby_mask must be binary")
    if sample.pattern_masks.shape == (len(PATTERN_CLASSES), *shape) and not _is_binary(sample.pattern_masks):
        errors.append("pattern_masks must be binary")
    if int(sample.severity.min()) < 0 or int(sample.severity.max()) > 7:
        errors.append("severity must be in grade range 0..7")
    if (sample.severity[sample.wafer_mask == 0] != 0).any():
        errors.append("severity outside wafer_mask must be 0")
    if (sample.stby_mask[sample.wafer_mask == 0] != 0).any():
        errors.append("stby pixels must be inside wafer_mask")
    if (sample.severity[sample.stby_mask > 0] != 0).any():
        errors.append("stby pixels must have severity grade 0 because fail bits are unobserved")
    if (sample.valid_test_mask[sample.stby_mask > 0] != 0).any():
        errors.append("stby pixels must be invalid test pixels")
    if (sample.valid_test_mask[sample.wafer_mask == 0] != 0).any():
        errors.append("pixels outside wafer must be invalid")
    if (sample.chip_index[sample.wafer_mask == 0] != -1).any():
        errors.append("chip_index outside wafer_mask must be -1")
    if list(sample.metadata.get("pattern_classes", [])) != list(PATTERN_CLASSES):
        errors.append("metadata pattern_classes must match PATTERN_CLASSES")
    image_shape = sample.metadata.get("image_shape", {})
    if image_shape:
        if int(image_shape.get("height", -1)) != shape[0] or int(image_shape.get("width", -1)) != shape[1]:
            errors.append("metadata image_shape must match severity shape")
    for item in sample.metadata.get("patterns", []):
        if item.get("type") not in PATTERN_CLASSES:
            errors.append(f"metadata pattern type is not registered: {item.get('type')}")
    return errors
