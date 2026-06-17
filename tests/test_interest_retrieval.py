import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_interest_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_interest_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_criterion_features_are_interest_specific_and_observable_only():
    module = _load_script()
    features = [
        "total_fail_density",
        "label_edge",
        "scratch_mask_ratio",
        "edge_density",
        "center_density",
        "radial_zone_04_severity",
        "shot_best_contrast",
        "scratch_angular_peak_contrast",
        "scratch_component_linear_score",
        "polar_r01_a03_severity",
        "stby_polar_r02_a11_ratio",
        "angular_sector_00_severity",
        "local_component_triangle_score",
    ]

    observable = [name for name in features if name not in {"label_edge"} and not name.endswith("_mask_ratio")]
    edge = module.criterion_feature_names(observable, "edge_focus")
    scratch = module.criterion_feature_names(observable, "scratch_focus")
    local = module.criterion_feature_names(observable, "local_focus")
    stby = module.criterion_feature_names(observable, "stby_focus")

    assert "edge_density" in edge
    assert "center_density" in edge
    assert "radial_zone_04_severity" in edge
    assert "label_edge" not in edge
    assert "scratch_mask_ratio" not in scratch
    assert "scratch_angular_peak_contrast" in scratch
    assert "scratch_component_linear_score" in scratch
    assert "angular_sector_00_severity" in scratch
    assert "local_component_triangle_score" in local
    assert "polar_r01_a03_severity" not in scratch
    assert "polar_r01_a03_severity" not in local
    assert "stby_polar_r02_a11_ratio" not in stby


def test_binary_metrics_reports_lift_against_base_rate():
    module = _load_script()
    labels = np.array(
        [
            [1],
            [1],
            [0],
            [0],
        ],
        dtype=np.int32,
    )
    neighbors = np.array(
        [
            [1, 2],
            [0, 2],
            [3, 0],
            [2, 0],
        ],
        dtype=np.int32,
    )

    metrics = module.binary_metrics(labels, neighbors, target_index=0)

    assert metrics["positive_queries"] == 2
    assert metrics["base_rate"] == 0.5
    assert metrics["precision_at_k"] == 0.5
    assert metrics["lift"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
