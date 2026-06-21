"""8-bit grayscale raw PNG parsing for real-unlabeled wafer maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from wafermap.data import GRAY_TO_GRADE, PATTERN_CLASSES, STBY_GRAY_VALUE


def load_png_gray_values(path: str | Path) -> np.ndarray:
    """Load an 8-bit grayscale PNG while rejecting lossy or alpha-masked input."""

    with Image.open(path) as image:
        mode = image.mode
        raw = np.asarray(image)
    if mode == "L" and raw.ndim == 2:
        gray = raw
    elif mode == "LA" and raw.ndim == 3 and raw.shape[2] == 2:
        if not (raw[:, :, 1] == 255).all():
            raise ValueError("png_grayscale_raw requires a fully opaque alpha channel")
        gray = raw[:, :, 0]
    elif mode in ("RGB", "RGBA") and raw.ndim == 3 and raw.shape[2] in (3, 4):
        rgb = raw[:, :, :3]
        if not ((rgb[:, :, 0] == rgb[:, :, 1]).all() and (rgb[:, :, 0] == rgb[:, :, 2]).all()):
            raise ValueError("png_grayscale_raw requires grayscale PNG values; RGB channels differ")
        if raw.shape[2] == 4 and not (raw[:, :, 3] == 255).all():
            raise ValueError("png_grayscale_raw requires a fully opaque alpha channel")
        gray = rgb[:, :, 0]
    else:
        raise ValueError("png_grayscale_raw requires a 2D grayscale image")
    if gray.dtype != np.uint8:
        raise ValueError("png_grayscale_raw requires 8-bit grayscale PNG values")
    return gray.astype(np.uint8, copy=False)


def severity_from_png_gray(raw: np.ndarray) -> np.ndarray:
    """Map exact gray values to grade 0..7 severity."""

    unique_values = {int(value) for value in np.unique(raw)}
    unknown_values = sorted(unique_values - set(GRAY_TO_GRADE))
    if unknown_values:
        raise ValueError(f"Unsupported PNG gray values: {unknown_values}")
    severity = np.zeros(raw.shape, dtype=np.uint8)
    for gray_value, grade in GRAY_TO_GRADE.items():
        severity[raw == gray_value] = grade
    return severity


def metadata_from_png_entry(entry: dict[str, Any], raw: np.ndarray) -> dict[str, Any]:
    """Build normalized metadata for one png_grayscale_raw manifest entry."""

    shape = raw.shape
    chip_width, chip_height, rows, cols = resolve_png_geometry(entry, raw)
    metadata = dict(entry.get("metadata", {}))
    metadata.update(
        {
            "sample_id": str(entry.get("sample_id", metadata.get("sample_id", "real_png_unlabeled"))),
            "actual_net_die": int(entry.get("actual_net_die", metadata.get("actual_net_die", 0))),
            "chip_blocks": {"width": chip_width, "height": chip_height},
            "grid": {"rows": rows, "cols": cols},
            "image_shape": {"height": int(shape[0]), "width": int(shape[1])},
            "pattern_classes": list(PATTERN_CLASSES),
            "patterns": [],
            "source_type": "png_grayscale_raw",
            "gray_value_map": {str(gray): grade for gray, grade in GRAY_TO_GRADE.items()},
            "wafer_mask_strategy": str(entry.get("wafer_mask_strategy", "centered_ellipse_from_png")),
        }
    )
    return metadata


def resolve_png_geometry(entry: dict[str, Any], raw: np.ndarray) -> tuple[int, int, int, int]:
    """Resolve chip width/height and grid rows/cols for an exact raw PNG."""

    height, width = raw.shape
    chip_blocks = entry.get("chip_blocks")
    grid = entry.get("grid")
    if chip_blocks:
        chip_width = int(chip_blocks["width"])
        chip_height = int(chip_blocks["height"])
    elif grid:
        rows = int(grid["rows"])
        cols = int(grid["cols"])
        if rows < 1 or cols < 1 or height % rows != 0 or width % cols != 0:
            raise ValueError("png_grayscale_raw grid must evenly divide the image when chip_blocks is omitted")
        chip_height = height // rows
        chip_width = width // cols
    else:
        if entry.get("allow_geometry_inference") is not True:
            raise ValueError(
                "png_grayscale_raw requires chip_blocks and grid unless allow_geometry_inference=true"
            )
        chip_height, chip_width = infer_chip_blocks_from_stby(raw)

    if chip_width < 1 or chip_height < 1:
        raise ValueError("chip_blocks width/height must be positive")

    if grid:
        rows = int(grid["rows"])
        cols = int(grid["cols"])
    else:
        if height % chip_height != 0 or width % chip_width != 0:
            raise ValueError("png_grayscale_raw chip_blocks must evenly divide the image when grid is omitted")
        rows = height // chip_height
        cols = width // chip_width

    if rows < 1 or cols < 1:
        raise ValueError("grid rows/cols must be positive")
    if rows * chip_height != height or cols * chip_width != width:
        raise ValueError(
            "png_grayscale_raw requires grid rows/cols and chip_blocks width/height to exactly match the image shape"
        )
    return chip_width, chip_height, rows, cols


def infer_chip_blocks_from_stby(raw: np.ndarray) -> tuple[int, int]:
    dims = _filled_rect_component_dims(raw == STBY_GRAY_VALUE)
    if not dims:
        raise ValueError("Cannot infer chip_blocks from PNG: no filled 255 stby rectangles found")
    max_area = max(height * width for height, width in dims)
    dims = [(height, width) for height, width in dims if height * width >= max_area * 0.5]
    heights = [height for height, _ in dims]
    widths = [width for _, width in dims]
    chip_height = _gcd_positive(heights)
    chip_width = _gcd_positive(widths)
    if chip_height < 2 or chip_width < 2:
        raise ValueError("Cannot infer chip_blocks from PNG: inferred stby block size is too small")
    if raw.shape[0] % chip_height != 0 or raw.shape[1] % chip_width != 0:
        raise ValueError(
            "Cannot infer chip_blocks from PNG: inferred stby block size does not divide the image shape"
        )
    if _all_components_match_inferred_size(dims, chip_height, chip_width) and any(
        _has_smaller_tiling_candidate(height, width, raw.shape) for height, width in dims
    ):
        raise ValueError(
            "Cannot infer chip_blocks from PNG: ambiguous 255 stby rectangles may contain adjacent stby chips; "
            "provide chip_blocks/grid explicitly"
        )
    return chip_height, chip_width


def detect_full_gray_stby_blocks(
    raw: np.ndarray,
    chip_height: int,
    chip_width: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Return a binary stby mask for chips whose full raw block is gray 255."""

    stby_mask = np.zeros(raw.shape, dtype=np.uint8)
    for row in range(rows):
        y0 = row * chip_height
        y1 = y0 + chip_height
        for col in range(cols):
            x0 = col * chip_width
            x1 = x0 + chip_width
            chip = raw[y0:y1, x0:x1]
            if chip.size and int(chip.min()) == STBY_GRAY_VALUE and int(chip.max()) == STBY_GRAY_VALUE:
                stby_mask[y0:y1, x0:x1] = 1
    return stby_mask


def infer_png_wafer_mask(metadata: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Infer wafer mask from normalized PNG metadata."""

    strategy = str(metadata.get("wafer_mask_strategy", "centered_ellipse_from_png"))
    if strategy == "full_grid_from_png":
        return np.ones(shape, dtype=np.uint8)
    if strategy != "centered_ellipse_from_png":
        raise ValueError(f"Unsupported png wafer_mask_strategy: {strategy}")

    chip_width = int(metadata["chip_blocks"]["width"])
    chip_height = int(metadata["chip_blocks"]["height"])
    rows = int(metadata["grid"]["rows"])
    cols = int(metadata["grid"]["cols"])
    actual_net_die = int(metadata.get("actual_net_die", 0))
    chip_radius: list[tuple[float, int, int]] = []
    for row in range(rows):
        y = ((row + 0.5) / rows - 0.5) * 2.0
        for col in range(cols):
            x = ((col + 0.5) / cols - 0.5) * 2.0
            chip_radius.append((x * x + y * y, row, col))

    if actual_net_die > 0:
        selected = sorted(chip_radius, key=lambda item: item[0])[:actual_net_die]
        active = {(row, col) for _, row, col in selected}
    else:
        active = {(row, col) for radius_sq, row, col in chip_radius if radius_sq <= 1.0}

    wafer_mask = np.zeros(shape, dtype=np.uint8)
    for row, col in active:
        y0 = row * chip_height
        y1 = y0 + chip_height
        x0 = col * chip_width
        x1 = x0 + chip_width
        wafer_mask[y0:y1, x0:x1] = 1
    return wafer_mask


def _filled_rect_component_dims(mask: np.ndarray) -> list[tuple[int, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    dims: list[tuple[int, int]] = []
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            y_min = y_max = row
            x_min = x_max = col
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                y_min = min(y_min, y)
                y_max = max(y_max, y)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            if area == height * width and area > 1:
                dims.append((height, width))
    return dims


def _gcd_positive(values: list[int]) -> int:
    result = 0
    for value in values:
        result = int(np.gcd(result, int(value)))
    return result


def _all_components_match_inferred_size(
    dims: list[tuple[int, int]],
    chip_height: int,
    chip_width: int,
) -> bool:
    return bool(dims) and all(height == chip_height and width == chip_width for height, width in dims)


def _has_smaller_tiling_candidate(
    component_height: int,
    component_width: int,
    image_shape: tuple[int, int],
) -> bool:
    image_height, image_width = image_shape
    height_candidates = [component_height, *_proper_divisors_at_least_two(component_height)]
    width_candidates = [component_width, *_proper_divisors_at_least_two(component_width)]
    for chip_height in height_candidates:
        if image_height % chip_height != 0:
            continue
        for chip_width in width_candidates:
            if chip_height == component_height and chip_width == component_width:
                continue
            if image_width % chip_width != 0:
                continue
            if component_height % chip_height == 0 and component_width % chip_width == 0:
                return True
    return False


def _proper_divisors_at_least_two(value: int) -> list[int]:
    return [candidate for candidate in range(2, value) if value % candidate == 0]
