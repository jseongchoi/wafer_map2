"""Shared data contracts for synthetic wafer map samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

PATTERN_CLASSES: tuple[str, ...] = (
    "scratch",
    "ring",
    "edge",
    "local",
    "random",
    "shot_grid",
    "stby_pattern",
)
GRADE_TO_GRAY: tuple[int, ...] = (0, 31, 151, 175, 191, 207, 223, 255)
GRAY_TO_GRADE: dict[int, int] = {gray: grade for grade, gray in enumerate(GRADE_TO_GRAY)}
STBY_GRAY_VALUE = 255


@dataclass(frozen=True)
class PatternInstance:
    """Metadata for one generated defect instance."""

    pattern_type: str
    instance_id: int
    clock_position: str
    severity: float
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticSample:
    """In-memory representation of one synthetic Fail Bit Map sample."""

    sample_id: str
    severity: NDArray[np.uint8]
    wafer_mask: NDArray[np.uint8]
    valid_test_mask: NDArray[np.uint8]
    stby_mask: NDArray[np.uint8]
    pattern_masks: NDArray[np.uint8]
    pattern_intensity: NDArray[np.float32]
    chip_index: NDArray[np.int32]
    metadata: dict[str, Any]

    def arrays(self) -> dict[str, NDArray[Any]]:
        return {
            "severity": self.severity,
            "wafer_mask": self.wafer_mask,
            "valid_test_mask": self.valid_test_mask,
            "stby_mask": self.stby_mask,
            "pattern_masks": self.pattern_masks,
            "pattern_intensity": self.pattern_intensity,
            "chip_index": self.chip_index,
        }

    @property
    def shape(self) -> tuple[int, int]:
        return self.severity.shape
