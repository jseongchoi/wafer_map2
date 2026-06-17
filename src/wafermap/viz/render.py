"""Rendering utilities for synthetic wafer map previews."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def render_grayscale(
    severity: NDArray[np.uint8],
    wafer_mask: NDArray[np.uint8],
    stby_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Render analysis tensors into the user's current grayscale convention."""

    image = np.zeros(severity.shape, dtype=np.uint8)
    measured = (wafer_mask > 0) & (stby_mask == 0)
    image[measured] = np.round(np.clip(severity[measured], 0, 7) / 7.0 * 230).astype(
        np.uint8
    )
    image[stby_mask > 0] = 255
    return image


def save_preview(
    path: str | Path,
    severity: NDArray[np.uint8],
    wafer_mask: NDArray[np.uint8],
    stby_mask: NDArray[np.uint8],
    title: str | None = None,
) -> None:
    """Save a compact color PNG preview for human expert review."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = severity.astype(np.float32).copy()
    image[wafer_mask == 0] = np.nan
    image[stby_mask > 0] = 7.0
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#2b1038")
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image, cmap=cmap, vmin=0, vmax=7, interpolation="nearest")
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.78, label="Grade")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
