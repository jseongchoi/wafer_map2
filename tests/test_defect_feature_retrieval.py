import importlib.util
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_defect_feature_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_defect_feature_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_targets_creates_location_aware_feature_targets():
    module = _load_script()
    sample_ids = ["s1", "s2", "s3"]
    defect_rows = [
        {
            "sample_id": "s1",
            "class_name": "scratch",
            "feature_key": "scratch__0300__middle",
            "location_label": "03:00",
            "radial_zone": "middle",
        },
        {
            "sample_id": "s2",
            "class_name": "scratch",
            "feature_key": "scratch__0300__middle",
            "location_label": "03:00",
            "radial_zone": "middle",
        },
        {
            "sample_id": "s3",
            "class_name": "local",
            "feature_key": "local__0600__center",
            "location_label": "06:00",
            "radial_zone": "center",
        },
    ]

    targets = module.build_targets(defect_rows, sample_ids, min_support=2)
    target_ids = {(item["target_kind"], item["target_id"]) for item in targets}

    assert ("class", "scratch") in target_ids
    assert ("class_location", "scratch__03:00") in target_ids
    assert ("class_radial", "scratch__middle") in target_ids
    assert ("feature_key", "scratch__0300__middle") in target_ids
    assert ("class", "local") not in target_ids


def test_evaluate_target_uses_positive_query_precision_against_random_baseline():
    module = _load_script()
    sample_ids = ["s1", "s2", "s3", "s4"]
    positive_ids = {"s1", "s2"}
    neighbors = np.array(
        [
            [1, 2],
            [0, 2],
            [3, 0],
            [2, 0],
        ],
        dtype=np.int32,
    )

    metrics = module.evaluate_target(sample_ids, positive_ids, neighbors, top_k=2)

    assert metrics["support"] == 2
    assert metrics["precision_at_k"] == 0.5
    assert metrics["random_precision"] == 1 / 3
    assert metrics["lift"] == 1.5
    assert metrics["hit_rate_at_k"] == 1.0


def test_class_feature_names_include_spatial_observable_features():
    module = _load_script()
    features = [
        "scratch_component_linear_score",
        "local_component_triangle_score",
        "morph_hot_chip_ratio",
        "polar_r01_a03_severity",
        "polar_r01_a03_fail_density",
        "stby_polar_r02_a11_ratio",
        "stby_ratio",
        "label_scratch",
    ]

    scratch_class = module.class_feature_names(features, "scratch", "class")
    scratch_feature_key = module.class_feature_names(features, "scratch", "feature_key")
    local_feature_key = module.class_feature_names(features, "local", "feature_key")
    stby_feature_key = module.class_feature_names(features, "stby_pattern", "feature_key")

    assert "polar_r01_a03_severity" not in scratch_class
    assert "polar_r01_a03_severity" in scratch_feature_key
    assert "polar_r01_a03_fail_density" in local_feature_key
    assert "stby_polar_r02_a11_ratio" in stby_feature_key
    assert "label_scratch" not in scratch_feature_key
