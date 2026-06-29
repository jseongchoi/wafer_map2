import numpy as np
import pytest

from wafermap.data import PATTERN_CLASSES, SyntheticSample
from wafermap.features import extract_feature_vector, extract_validation_feature_vector
from wafermap.synth import SyntheticConfig, generate_sample


@pytest.mark.slow
def test_extract_feature_vector_contains_observable_analysis_columns():
    sample = generate_sample(
        SyntheticConfig(count=1, target_net_die=80, chip_width=20, chip_height=10, seed=5),
        0,
    )

    features = extract_feature_vector(sample)

    assert "total_fail_density" in features
    assert "stby_ratio" in features
    assert not any(name.endswith("_mask_ratio") for name in features)
    assert "radial_zone_00_severity" in features
    assert "angular_sector_11_severity" in features
    assert "polar_r00_a00_severity" in features
    assert "polar_r02_a11_fail_density" in features
    assert "stby_polar_r02_a11_ratio" in features
    assert "edge_minus_center_density" in features
    assert "edge_chip_outer_minus_inner_density" in features
    assert "shot_best_contrast" in features
    assert "shot_lower_left_contrast" in features
    assert "ring_radial_peak_contrast" in features
    assert "scratch_angular_peak_contrast" in features
    assert "local_hotspot_peak_contrast" in features
    assert "scratch_component_linear_score" in features
    assert "local_component_triangle_score" in features
    assert "edge_sector_peak_contrast" in features
    assert 0.0 <= features["stby_ratio"] <= 1.0
    assert 0.0 <= features["polar_r00_a00_severity"] <= 1.0
    assert 0.0 <= features["polar_r02_a11_fail_density"] <= 1.0
    assert 0.0 <= features["stby_polar_r02_a11_ratio"] <= 1.0
    assert 0.0 <= features["local_hotspot_count_ratio"] <= 1.0
    assert 0.0 <= features["morph_hot_chip_ratio"] <= 1.0
    assert 0.0 <= features["scratch_component_linear_score"] <= 1.0
    assert 0.0 <= features["local_component_triangle_score"] <= 1.0
    assert 0.0 <= features["edge_sector_concentration"] <= 1.0


@pytest.mark.slow
def test_validation_feature_vector_contains_synthetic_oracle_fields():
    sample = generate_sample(
        SyntheticConfig(count=1, target_net_die=80, chip_width=20, chip_height=10, seed=5),
        0,
    )

    features = extract_validation_feature_vector(sample)

    assert "scratch_mask_ratio" in features
    assert "stby_pattern_mask_ratio" in features
    assert all(name.endswith("_mask_ratio") for name in features)


def test_observable_zone_features_ignore_untested_pixels():
    severity = np.array([[7, 0], [0, 0]], dtype=np.uint8)
    sample = SyntheticSample(
        sample_id="valid_mask_unit",
        severity=severity,
        wafer_mask=np.ones((2, 2), dtype=np.uint8),
        valid_test_mask=np.array([[1, 0], [0, 0]], dtype=np.uint8),
        stby_mask=np.array([[0, 1], [1, 1]], dtype=np.uint8),
        pattern_masks=np.zeros((len(PATTERN_CLASSES), 2, 2), dtype=np.uint8),
        pattern_intensity=np.zeros((len(PATTERN_CLASSES), 2, 2), dtype=np.float32),
        chip_index=np.array([[0, 1], [2, 3]], dtype=np.int32),
        metadata={
            "actual_net_die": 4,
            "chip_blocks": {"width": 1, "height": 1},
            "grid": {"rows": 2, "cols": 2},
        },
    )

    features = extract_feature_vector(sample)

    assert features["total_fail_density"] == 1.0
    assert features["grade_weighted_severity"] == 1.0
    radial_values = [features[f"radial_zone_{idx:02d}_severity"] for idx in range(5)]
    assert max(radial_values) == 1.0
