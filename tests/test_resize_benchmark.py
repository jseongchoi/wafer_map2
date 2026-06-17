import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_resize_benchmark.py"
    spec = importlib.util.spec_from_file_location("evaluate_resize_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_pool_keeps_stby_and_valid_as_separate_channels(tmp_path):
    module = _load_script()
    sample = module.LightSample(
        sample_id="unit",
        sample_dir=tmp_path,
        severity=np.array([[0, 7], [3, 0]], dtype=np.uint8),
        wafer_mask=np.array([[1, 1], [1, 0]], dtype=np.uint8),
        valid_test_mask=np.array([[1, 0], [1, 0]], dtype=np.uint8),
        stby_mask=np.array([[0, 1], [0, 0]], dtype=np.uint8),
        labels=np.zeros(6, dtype=np.int32),
    )

    pooled = module.semantic_pool_vector(sample, grid_size=1)

    assert pooled.shape == (6, 1, 1)
    assert pooled[0, 0, 0] == 3 / 14
    assert pooled[1, 0, 0] == 0.5
    assert pooled[3, 0, 0] == 1 / 3
    assert pooled[4, 0, 0] == 2 / 3
    assert pooled[5, 0, 0] == 3 / 4


def test_nearest_neighbors_uses_matrix_distance_without_broadcast_cube():
    module = _load_script()
    x = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]], dtype=np.float32)

    neighbors, distances = module.nearest_neighbors(x, top_k=1)

    assert neighbors.shape == (3, 1)
    assert neighbors[0, 0] == 1
    assert np.isfinite(distances[0, 1])
