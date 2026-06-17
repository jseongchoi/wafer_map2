import numpy as np

from wafermap.data import PATTERN_CLASSES
from wafermap.evaluation import validate_synthetic_sample
from wafermap.synth import SyntheticConfig, generate_sample


def test_generate_small_synthetic_sample_contract():
    config = SyntheticConfig(count=1, target_net_die=80, chip_width=20, chip_height=10, seed=3)

    sample = generate_sample(config, 0)

    assert validate_synthetic_sample(sample) == []
    assert sample.metadata["actual_net_die"] == 80
    assert sample.severity.dtype == np.uint8
    assert sample.pattern_masks.shape == (len(PATTERN_CLASSES), *sample.shape)
    assert sample.valid_test_mask[sample.stby_mask > 0].sum() == 0
    assert sample.severity.max() <= 7


def test_synthetic_validation_rejects_broken_schema_contract():
    config = SyntheticConfig(count=1, target_net_die=80, chip_width=20, chip_height=10, seed=19)
    sample = generate_sample(config, 0)

    y, x = np.argwhere(sample.wafer_mask > 0)[0]
    sample.stby_mask[y, x] = 2
    sample.severity[y, x] = 5
    sample.metadata["pattern_classes"] = ["scratch"]

    errors = validate_synthetic_sample(sample)

    assert "stby_mask must be binary" in errors
    assert "stby pixels must have severity grade 0 because fail bits are unobserved" in errors
    assert "metadata pattern_classes must match PATTERN_CLASSES" in errors


def test_default_geometry_hits_target_net_die():
    config = SyntheticConfig(count=1, target_net_die=600, chip_width=100, chip_height=50, seed=7)

    sample = generate_sample(config, 0)

    assert sample.metadata["actual_net_die"] == 600
    assert sample.shape == (1900, 2000)


def test_stby_probability_creates_chip_level_missing_when_forced():
    config = SyntheticConfig(
        count=1,
        target_net_die=80,
        chip_width=20,
        chip_height=10,
        stby_min_chips=5,
        stby_max_chips=7,
        seed=11,
        pattern_probabilities={
            "scratch": 1.0,
            "ring": 0.0,
            "edge": 0.0,
            "local": 0.0,
            "random": 0.0,
            "stby_pattern": 1.0,
        },
    )

    sample = generate_sample(config, 0)
    chip_area = config.chip_width * config.chip_height
    stby_chips = sample.stby_mask.sum() // chip_area

    assert 5 <= stby_chips <= 7
    assert sample.pattern_masks[PATTERN_CLASSES.index("stby_pattern")].sum() == sample.stby_mask.sum()


def test_stby_can_hide_local_impact_origin_chip():
    config = SyntheticConfig(
        count=1,
        target_net_die=80,
        chip_width=20,
        chip_height=10,
        stby_min_chips=1,
        stby_max_chips=1,
        seed=17,
        pattern_probabilities={
            "scratch": 0.0,
            "ring": 0.0,
            "edge": 0.0,
            "local": 1.0,
            "random": 0.0,
            "stby_pattern": 1.0,
        },
    )

    sample = generate_sample(config, 0)
    local = next(item for item in sample.metadata["patterns"] if item["type"] == "local")
    stby = next(item for item in sample.metadata["patterns"] if item["type"] == "stby_pattern")
    x_norm = float(local["parameters"]["x_norm"])
    y_norm = float(local["parameters"]["y_norm"])
    height, width = sample.shape
    x = int(round((x_norm * (width / 2.0)) + (width - 1) / 2.0))
    y = int(round((y_norm * (height / 2.0)) + (height - 1) / 2.0))
    row = y // config.chip_height
    col = x // config.chip_width
    y0 = row * config.chip_height
    y1 = y0 + config.chip_height
    x0 = col * config.chip_width
    x1 = x0 + config.chip_width

    assert sample.stby_mask[y0:y1, x0:x1].all()
    assert sample.valid_test_mask[y0:y1, x0:x1].sum() == 0
    assert stby["parameters"]["mode"] == "origin_coupled_or_random_chip_missing"
    assert stby["parameters"]["origin_seed_count"] == 1
    assert stby["parameters"]["seeded_stby_chip_count"] == 1
    assert stby["parameters"]["latent_weighted_stby_chip_count_est"] == 0


def test_pattern_classes_exclude_flow_and_include_ring():
    assert "flow" not in PATTERN_CLASSES
    assert "ring" in PATTERN_CLASSES
    assert "shot_grid" in PATTERN_CLASSES


def test_shot_grid_pattern_repeats_relative_shot_region_when_forced():
    config = SyntheticConfig(
        count=1,
        target_net_die=80,
        chip_width=20,
        chip_height=10,
        seed=41,
        pattern_probabilities={
            "scratch": 0.0,
            "ring": 0.0,
            "edge": 0.0,
            "local": 0.0,
            "random": 0.0,
            "shot_grid": 1.0,
            "stby_pattern": 0.0,
        },
    )

    sample = generate_sample(config, 0)
    shot = next(item for item in sample.metadata["patterns"] if item["type"] == "shot_grid")
    params = shot["parameters"]
    affected_slots = {tuple(slot) for slot in params["affected_slots"]}
    shot_signal = sample.pattern_intensity[PATTERN_CLASSES.index("shot_grid")]
    shot_groups: dict[tuple[int, int], dict[tuple[int, int], list[float]]] = {}

    for chip_id in np.unique(sample.chip_index[sample.chip_index >= 0]):
        chip = sample.chip_index == chip_id
        ys, xs = np.where(chip)
        chip_row = int(ys.min() // config.chip_height)
        chip_col = int(xs.min() // config.chip_width)
        shot_key = (
            (chip_row - params["shot_row_offset"]) // params["shot_rows"],
            (chip_col - params["shot_col_offset"]) // params["shot_cols"],
        )
        slot = (
            (chip_row - params["shot_row_offset"]) % params["shot_rows"],
            (chip_col - params["shot_col_offset"]) % params["shot_cols"],
        )
        shot_groups.setdefault(shot_key, {}).setdefault(slot, []).append(float(shot_signal[chip].mean()))

    repeated_region_wins = 0
    usable_shots = 0
    for slots in shot_groups.values():
        affected_values = [value for slot, values in slots.items() if slot in affected_slots for value in values]
        other_values = [value for slot, values in slots.items() if slot not in affected_slots for value in values]
        if not affected_values or not other_values or max(affected_values) <= 0:
            continue
        usable_shots += 1
        if float(np.mean(affected_values)) > float(np.mean(other_values)):
            repeated_region_wins += 1

    assert params["anchor_region"] in {"lower_left", "left_edge", "bottom_edge"}
    assert params["touched_shot_count"] >= 2
    assert repeated_region_wins >= max(2, usable_shots // 2)


def test_local_blob_modes_cycle_for_debug_review():
    config = SyntheticConfig(
        count=3,
        target_net_die=80,
        chip_width=20,
        chip_height=10,
        seed=13,
        pattern_probabilities={
            "scratch": 0.0,
            "ring": 0.0,
            "edge": 0.0,
            "local": 1.0,
            "random": 0.0,
            "stby_pattern": 0.0,
        },
    )

    modes = []
    for idx in range(3):
        sample = generate_sample(config, idx)
        local = [item for item in sample.metadata["patterns"] if item["type"] == "local"]
        modes.append(local[0]["parameters"]["mode"])

    assert modes == ["single_blob", "double_blob", "triple_triangle"]


def test_polar_edge_lift_increases_fail_density_near_edge():
    config = SyntheticConfig(
        count=1,
        target_net_die=80,
        chip_width=20,
        chip_height=10,
        seed=21,
        pattern_probabilities={
            "scratch": 0.0,
            "ring": 0.0,
            "edge": 0.0,
            "local": 0.0,
            "random": 0.0,
            "stby_pattern": 0.0,
        },
    )

    sample = generate_sample(config, 0)
    yy, xx = np.indices(sample.shape)
    cx = (sample.shape[1] - 1) / 2.0
    cy = (sample.shape[0] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = distance / distance[sample.wafer_mask > 0].max()
    valid = sample.valid_test_mask > 0
    center = valid & (radius < 0.55)
    edge = valid & (radius > 0.78)

    assert ((sample.severity > 0) & edge).sum() / edge.sum() > ((sample.severity > 0) & center).sum() / center.sum()


def test_edge_chip_outer_face_has_higher_fail_density_than_inner_face():
    config = SyntheticConfig(
        count=1,
        target_net_die=80,
        chip_width=20,
        chip_height=10,
        seed=31,
        pattern_probabilities={
            "scratch": 0.0,
            "ring": 0.0,
            "edge": 0.0,
            "local": 0.0,
            "random": 0.0,
            "stby_pattern": 0.0,
        },
    )

    sample = generate_sample(config, 0)
    yy, xx = np.indices(sample.shape)
    cx = (sample.shape[1] - 1) / 2.0
    cy = (sample.shape[0] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = distance / distance[sample.wafer_mask > 0].max()
    valid = sample.valid_test_mask > 0
    inner_face = np.zeros(sample.shape, dtype=bool)
    outer_face = np.zeros(sample.shape, dtype=bool)

    for chip_id in np.unique(sample.chip_index[sample.chip_index >= 0]):
        chip = sample.chip_index == chip_id
        chip_radius = radius[chip]
        if chip_radius.max() < 0.78:
            continue
        radius_min = chip_radius.min()
        radius_max = chip_radius.max()
        if radius_max <= radius_min:
            continue
        local_rank = (radius - radius_min) / (radius_max - radius_min)
        inner_face |= chip & valid & (local_rank < 0.34)
        outer_face |= chip & valid & (local_rank > 0.66)

    inner_density = ((sample.severity > 0) & inner_face).sum() / inner_face.sum()
    outer_density = ((sample.severity > 0) & outer_face).sum() / outer_face.sum()

    assert outer_density > inner_density
