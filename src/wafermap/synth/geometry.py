"""Synthetic wafer chip-grid geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Geometry:
    rows: int
    cols: int
    chip_width: int
    chip_height: int
    chip_index: NDArray[np.int32]
    wafer_mask: NDArray[np.uint8]
    chip_centers: NDArray[np.float32]

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows * self.chip_height, self.cols * self.chip_width

    @property
    def net_die(self) -> int:
        return int(self.chip_centers.shape[0])


def make_geometry(target_net_die: int, chip_width: int, chip_height: int) -> Geometry:
    """Create a rectangular chip grid clipped by an elliptical wafer boundary."""

    best: tuple[float, int, int, int] | None = None
    for cols in range(8, 80):
        for rows in range(8, 100):
            count = _count_ellipse_chips(rows, cols)
            aspect = abs((cols * chip_width) / (rows * chip_height) - 1.0)
            score = abs(count - target_net_die) + aspect * 12.0
            if best is None or score < best[0]:
                best = (score, rows, cols, count)

    if best is None:
        raise RuntimeError("Unable to create wafer geometry")
    _, rows, cols, _ = best

    chip_grid = np.full((rows, cols), -1, dtype=np.int32)
    centers: list[tuple[float, float, int, int]] = []
    next_idx = 0
    for row in range(rows):
        for col in range(cols):
            x = ((col + 0.5) / cols - 0.5) * 2.0
            y = ((row + 0.5) / rows - 0.5) * 2.0
            if x * x + y * y <= 1.0:
                chip_grid[row, col] = next_idx
                centers.append((x, y, row, col))
                next_idx += 1

    chip_index = np.repeat(np.repeat(chip_grid, chip_height, axis=0), chip_width, axis=1)
    wafer_mask = (chip_index >= 0).astype(np.uint8)
    chip_centers = np.array(centers, dtype=np.float32)
    return Geometry(rows, cols, chip_width, chip_height, chip_index, wafer_mask, chip_centers)


def _count_ellipse_chips(rows: int, cols: int) -> int:
    count = 0
    for row in range(rows):
        for col in range(cols):
            x = ((col + 0.5) / cols - 0.5) * 2.0
            y = ((row + 0.5) / rows - 0.5) * 2.0
            if x * x + y * y <= 1.0:
                count += 1
    return count
