import numpy as np
import pytest
from PIL import Image

from wafermap.real import load_png_gray_values, resolve_png_geometry, severity_from_png_gray


def test_real_png_raw_public_api_loads_exact_grayscale(tmp_path):
    raw = np.array([[0, 31], [151, 255]], dtype=np.uint8)
    path = tmp_path / "raw.png"
    Image.fromarray(raw).save(path)

    loaded = load_png_gray_values(path)
    severity = severity_from_png_gray(loaded)

    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, raw)
    assert severity.tolist() == [[0, 1], [2, 7]]


def test_real_png_raw_public_api_rejects_unknown_gray():
    with pytest.raises(ValueError, match="Unsupported PNG gray values"):
        severity_from_png_gray(np.array([[0, 99]], dtype=np.uint8))


def test_real_png_raw_public_api_infers_geometry_from_stby():
    raw = np.zeros((6, 8), dtype=np.uint8)
    raw[0:3, 2:4] = 255

    chip_width, chip_height, rows, cols = resolve_png_geometry({"allow_geometry_inference": True}, raw)

    assert (chip_width, chip_height, rows, cols) == (2, 3, 2, 4)


def test_real_png_raw_public_api_rejects_ambiguous_adjacent_stby_geometry():
    raw = np.zeros((4, 4), dtype=np.uint8)
    raw[0:2, 0:4] = 255

    with pytest.raises(ValueError, match="ambiguous 255 stby rectangles"):
        resolve_png_geometry({"allow_geometry_inference": True}, raw)
