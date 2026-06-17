import numpy as np

from wafermap.evaluation import cross_nearest_neighbor_indices, nearest_neighbor_indices
from wafermap.features import compact_observable_feature_names, observable_feature_names
from wafermap.reporting import clock_position_from_xy
from wafermap.viz import render_grayscale


def test_render_grayscale_keeps_stby_distinct_from_grade():
    severity = np.array([[0, 1, 7]], dtype=np.uint8)
    wafer_mask = np.ones_like(severity, dtype=np.uint8)
    stby_mask = np.array([[0, 0, 1]], dtype=np.uint8)

    rendered = render_grayscale(severity, wafer_mask, stby_mask)

    assert rendered.tolist() == [[0, 33, 255]]


def test_clock_position_uses_ascii_labels():
    assert clock_position_from_xy(1, 0, 0, 0) == "03:00"
    assert clock_position_from_xy(0, -1, 0, 0) == "12:00"
    assert clock_position_from_xy(0, 0, 0, 0) == "center"


def test_compact_observable_feature_names_exclude_labels_masks_and_polar():
    row = {
        "sample_id": "s1",
        "actual_net_die": 600,
        "total_fail_density": 0.1,
        "label_edge": 1,
        "scratch_mask_ratio": 0.2,
        "polar_r00_a00_severity": 3.0,
        "stby_polar_r00_a00_ratio": 1.0,
    }

    assert compact_observable_feature_names(row) == ["total_fail_density"]
    assert observable_feature_names(row, include_location_aware=True) == [
        "total_fail_density",
        "polar_r00_a00_severity",
        "stby_polar_r00_a00_ratio",
    ]


def test_nearest_neighbor_utilities_share_matrix_distance_path():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]], dtype=np.float32)

    neighbors, distances = nearest_neighbor_indices(x, top_k=1)

    assert neighbors.tolist() == [[1], [0], [1]]
    assert np.isfinite(distances[0, 1])


def test_cross_nearest_neighbors_fit_reference_distribution():
    reference = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    query = np.array([[9.0, 0.0]], dtype=np.float32)

    neighbors, distances = cross_nearest_neighbor_indices(query, reference, top_k=1)

    assert neighbors.tolist() == [[1]]
    assert distances.shape == (1, 2)
