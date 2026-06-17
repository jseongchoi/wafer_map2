"""Grid pooling utilities for high-resolution wafer tensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def integral_image(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return a padded integral image for fast rectangular sums."""

    integral = values.cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def rect_sums(
    integral: NDArray[np.float32],
    y_edges: NDArray[np.int32],
    x_edges: NDArray[np.int32],
) -> NDArray[np.float32]:
    y0 = y_edges[:-1]
    y1 = y_edges[1:]
    x0 = x_edges[:-1]
    x1 = x_edges[1:]
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )


def grid_edges(shape: tuple[int, int], grid_size: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    height, width = shape
    y_edges = np.linspace(0, height, grid_size + 1).astype(np.int32)
    x_edges = np.linspace(0, width, grid_size + 1).astype(np.int32)
    return y_edges, x_edges


def pooled_mean(
    values: NDArray[np.float32],
    weights: NDArray[np.float32],
    y_edges: NDArray[np.int32],
    x_edges: NDArray[np.int32],
) -> NDArray[np.float32]:
    numerator = rect_sums(integral_image(values * weights), y_edges, x_edges)
    denominator = rect_sums(integral_image(weights), y_edges, x_edges)
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-6),
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def pooled_occupancy(
    mask: NDArray[np.bool_],
    y_edges: NDArray[np.int32],
    x_edges: NDArray[np.int32],
) -> NDArray[np.float32]:
    sums = rect_sums(integral_image(mask.astype(np.float32)), y_edges, x_edges)
    area = np.diff(y_edges).astype(np.float32)[:, None] * np.diff(x_edges).astype(np.float32)[None, :]
    return sums / np.maximum(area, 1.0)
