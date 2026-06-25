import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wafermap.data import PATTERN_CLASSES, SyntheticSample


def _load_real_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_png_batch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_png_raw_folders.py"
    spec = importlib.util.spec_from_file_location("analyze_png_raw_folders", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample() -> SyntheticSample:
    severity = np.zeros((4, 4), dtype=np.uint8)
    severity[0:2, 0:2] = 2
    wafer_mask = np.ones((4, 4), dtype=np.uint8)
    valid_test_mask = np.ones((4, 4), dtype=np.uint8)
    stby_mask = np.zeros((4, 4), dtype=np.uint8)
    stby_mask[2:4, 2:4] = 1
    valid_test_mask[2:4, 2:4] = 0
    chip_index = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    return SyntheticSample(
        sample_id="real_like_unit",
        severity=severity,
        wafer_mask=wafer_mask,
        valid_test_mask=valid_test_mask,
        stby_mask=stby_mask,
        pattern_masks=np.zeros((len(PATTERN_CLASSES), 4, 4), dtype=np.uint8),
        pattern_intensity=np.zeros((len(PATTERN_CLASSES), 4, 4), dtype=np.float32),
        chip_index=chip_index,
        metadata={
            "actual_net_die": 4,
            "chip_blocks": {"width": 2, "height": 2},
            "grid": {"rows": 2, "cols": 2},
        },
    )


def _manifest_entry(npz_path: Path) -> dict:
    return {
        "sample_id": "product_aaaaaaaaaa_wbbbbbbbbbb",
        "source_type": "npz_semantic_arrays",
        "arrays_npz": str(npz_path),
        "parser_name": "unit_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "chip_blocks": {"width": 2, "height": 2},
        "grid": {"rows": 2, "cols": 2},
        "actual_net_die": 4,
    }


def _npz_arrays() -> dict[str, np.ndarray]:
    severity = np.zeros((4, 4), dtype=np.uint8)
    severity[0:2, 0:2] = 2
    wafer_mask = np.ones((4, 4), dtype=np.uint8)
    valid_test_mask = np.ones((4, 4), dtype=np.uint8)
    stby_mask = np.zeros((4, 4), dtype=np.uint8)
    stby_mask[2:4, 2:4] = 1
    valid_test_mask[2:4, 2:4] = 0
    chip_index = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    return {
        "severity": severity,
        "wafer_mask": wafer_mask,
        "valid_test_mask": valid_test_mask,
        "stby_mask": stby_mask,
        "chip_index": chip_index,
    }


def test_real_like_validator_accepts_semantic_stby_contract():
    module = _load_real_script()
    sample = _sample()

    errors, warnings = module.validate_real_like_sample(sample)

    assert errors == []
    assert warnings == []


def test_real_like_validator_rejects_stby_as_grade7():
    module = _load_real_script()
    sample = _sample()
    sample.severity[sample.stby_mask > 0] = 7

    errors, _ = module.validate_real_like_sample(sample)

    assert "stby pixels must have severity grade 0; stby is unobserved, not Grade 7" in errors


def test_manifest_requires_known_schema_versions_and_source_type():
    module = _load_real_script()
    base = {
        "schema_version": "real_unlabeled_manifest/v1",
        "feature_schema_version": "observable_fbm_features/v1",
        "samples": [{"sample_id": "x"}],
    }

    with pytest.raises(ValueError, match="requires source_type"):
        module.validate_manifest(base)

    bad_version = dict(base)
    bad_version["schema_version"] = "v0"
    with pytest.raises(ValueError, match="schema_version=real_unlabeled_manifest/v1"):
        module.validate_manifest(bad_version)


def test_manifest_rejects_duplicate_sample_ids():
    module = _load_real_script()
    manifest = {
        "schema_version": "real_unlabeled_manifest/v1",
        "feature_schema_version": "observable_fbm_features/v1",
        "samples": [
            {"sample_id": "dup", "source_type": "synthetic_sample_dir", "sample_dir": "a"},
            {"sample_id": "dup", "source_type": "synthetic_sample_dir", "sample_dir": "b"},
        ],
    }

    with pytest.raises(ValueError, match="duplicate sample_id"):
        module.validate_manifest(manifest)


def test_real_unlabeled_manifest_templates_are_valid():
    module = _load_real_script()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "real_unlabeled_manifest_template_standard.json",
        "real_unlabeled_manifest_template_keymap.json",
        "real_unlabeled_manifest_template_png.json",
    ):
        manifest = json.loads((root / "configs" / "eval" / name).read_text(encoding="utf-8"))

        module.validate_manifest(manifest)


def test_real_unlabeled_cli_top_k_requires_positive_integer():
    module = _load_real_script()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--top-k",
                "0",
            ]
        )


def test_load_npz_semantic_arrays_with_key_mapping(tmp_path):
    module = _load_real_script()
    arrays = _npz_arrays()
    npz_path = tmp_path / "arrays.npz"
    np.savez(
        npz_path,
        grade=arrays["severity"],
        in_wafer=arrays["wafer_mask"],
        valid=arrays["valid_test_mask"],
        stby=arrays["stby_mask"],
        die_id=arrays["chip_index"],
    )
    entry = _manifest_entry(npz_path)
    entry["array_keys"] = {
        "severity": "grade",
        "wafer_mask": "in_wafer",
        "valid_test_mask": "valid",
        "stby_mask": "stby",
        "chip_index": "die_id",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": [entry]}), encoding="utf-8")

    sample = module.load_real_like_sample(entry, manifest_path)
    errors, warnings = module.validate_real_like_sample(sample)

    assert sample.sample_id == "product_aaaaaaaaaa_wbbbbbbbbbb"
    assert sample.severity.dtype == np.uint8
    assert errors == []
    assert warnings == []


def test_load_png_grayscale_raw_maps_grades_and_stby_chip(tmp_path):
    module = _load_real_script()
    raw = np.array(
        [
            [255, 255, 0, 31],
            [255, 255, 151, 175],
            [191, 207, 223, 255],
            [0, 31, 151, 175],
        ],
        dtype=np.uint8,
    )
    png_path = tmp_path / "wafer.png"
    Image.fromarray(raw).save(png_path)
    entry = {
        "sample_id": "product_aaaaaaaaaa_wcccccccccc",
        "source_type": "png_grayscale_raw",
        "png_path": str(png_path),
        "parser_name": "unit_png_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "chip_blocks": {"width": 2, "height": 2},
        "grid": {"rows": 2, "cols": 2},
    }

    sample = module.load_real_like_sample(entry, tmp_path / "manifest.json")
    errors, warnings = module.validate_real_like_sample(sample)

    assert sample.sample_id == "product_aaaaaaaaaa_wcccccccccc"
    assert sample.severity.dtype == np.uint8
    assert sample.stby_mask[0:2, 0:2].all()
    assert sample.valid_test_mask[0:2, 0:2].sum() == 0
    assert sample.severity[0:2, 0:2].sum() == 0
    assert sample.severity[2, 3] == 7
    assert sample.valid_test_mask[2, 3] == 1
    assert errors == []
    assert any("centered-ellipse wafer_mask" in warning for warning in warnings)


def test_load_png_grayscale_raw_infers_chip_blocks_from_stby(tmp_path):
    module = _load_real_script()
    raw = np.zeros((6, 8), dtype=np.uint8)
    raw[0:3, 2:4] = 255
    png_path = tmp_path / "wafer.png"
    Image.fromarray(raw).save(png_path)
    entry = {
        "sample_id": "product_aaaaaaaaaa_wdddddddddd",
        "source_type": "png_grayscale_raw",
        "png_path": str(png_path),
        "parser_name": "unit_png_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "allow_geometry_inference": True,
    }

    sample = module.load_real_like_sample(entry, tmp_path / "manifest.json")

    assert sample.metadata["chip_blocks"] == {"width": 2, "height": 3}
    assert sample.metadata["grid"] == {"rows": 2, "cols": 4}
    assert sample.stby_mask.sum() == 6


def test_load_png_grayscale_raw_rejects_unknown_gray_values(tmp_path):
    module = _load_real_script()
    raw = np.array([[0, 31], [151, 99]], dtype=np.uint8)
    png_path = tmp_path / "wafer.png"
    Image.fromarray(raw).save(png_path)
    entry = {
        "sample_id": "product_aaaaaaaaaa_weeeeeeeeee",
        "source_type": "png_grayscale_raw",
        "png_path": str(png_path),
        "parser_name": "unit_png_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "chip_blocks": {"width": 1, "height": 1},
        "grid": {"rows": 2, "cols": 2},
    }

    with pytest.raises(ValueError, match="Unsupported PNG gray values"):
        module.load_real_like_sample(entry, tmp_path / "manifest.json")


def test_png_raw_folder_batch_builds_manifest_with_inferred_product_geometry(tmp_path):
    module = _load_png_batch_script()
    real_module = _load_real_script()
    product_dir = tmp_path / "raw" / "Product A"
    product_dir.mkdir(parents=True)
    raw = np.array(
        [
            [255, 255, 0, 31],
            [255, 255, 151, 175],
            [191, 207, 223, 0],
            [0, 31, 151, 175],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(raw).save(product_dir / "wafer_001.png")

    groups = module.discover_products(
        tmp_path / "raw",
        glob_pattern="*.png",
        recursive=True,
        limit_per_product=None,
    )
    manifest = module.build_manifest(
        groups,
        geometry_by_product={},
        wafer_mask_strategy="centered_ellipse_from_png",
        orientation="not_rotated",
    )

    real_module.validate_manifest(manifest)
    entry = manifest["samples"][0]
    assert entry["sample_id"] == "Product_A_wafer_001"
    assert entry["chip_blocks"] == {"width": 2, "height": 2}
    assert entry["grid"] == {"rows": 2, "cols": 2}
    assert entry["source_type"] == "png_grayscale_raw"


def test_png_raw_folder_batch_rejects_mixed_inferred_geometry(tmp_path):
    module = _load_png_batch_script()
    product_dir = tmp_path / "raw" / "prod_mixed"
    product_dir.mkdir(parents=True)
    raw_a = np.zeros((4, 4), dtype=np.uint8)
    raw_a[0:2, 0:2] = 255
    raw_b = np.zeros((6, 4), dtype=np.uint8)
    raw_b[0:3, 0:2] = 255
    Image.fromarray(raw_a).save(product_dir / "wafer_a.png")
    Image.fromarray(raw_b).save(product_dir / "wafer_b.png")
    groups = module.discover_products(
        tmp_path / "raw",
        glob_pattern="*.png",
        recursive=True,
        limit_per_product=None,
    )

    with pytest.raises(ValueError, match="Inconsistent inferred chip geometry"):
        module.build_manifest(
            groups,
            geometry_by_product={},
            wafer_mask_strategy="centered_ellipse_from_png",
            orientation="not_rotated",
        )


def test_png_raw_folder_batch_accepts_explicit_geometry_when_no_stby(tmp_path):
    module = _load_png_batch_script()
    product_dir = tmp_path / "raw" / "prod_b"
    product_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(product_dir / "wafer_001.png")
    groups = module.discover_products(
        tmp_path / "raw",
        glob_pattern="*.png",
        recursive=True,
        limit_per_product=None,
    )

    manifest = module.build_manifest(
        groups,
        geometry_by_product={"prod_b": {"chip_blocks": {"width": 2, "height": 2}, "grid": {"rows": 2, "cols": 2}}},
        wafer_mask_strategy="centered_ellipse_from_png",
        orientation="not_rotated",
    )

    entry = manifest["samples"][0]
    assert entry["sample_id"] == "prod_b_wafer_001"
    assert entry["chip_blocks"] == {"width": 2, "height": 2}


def test_png_raw_folder_batch_uses_readable_folder_and_file_sample_ids(tmp_path):
    module = _load_png_batch_script()
    product_dir = tmp_path / "raw" / "Product Name"
    product_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(product_dir / "wafer_001.png")

    groups = module.discover_products(
        tmp_path / "raw",
        glob_pattern="*.png",
        recursive=True,
        limit_per_product=None,
    )

    assert groups[0].alias == "Product_Name"
    assert module.stable_wafer_alias(groups[0], product_dir / "wafer_001.png") == "wafer_001"


def test_png_raw_folder_batch_metadata_keeps_manifest_path(tmp_path, monkeypatch):
    module = _load_png_batch_script()
    repo = tmp_path / "repo"
    product_dir = tmp_path / "raw" / "Product Name"
    product_dir.mkdir(parents=True)
    png_path = product_dir / "wafer_001.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(png_path)
    monkeypatch.setattr(module, "ROOT", repo)
    args = module.parse_args(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--geometry-json",
            "product_geometry.json",
            "--out-dir",
            "outputs/reports/real_png_batch",
            "--cpu-model",
            "outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz",
        ]
    )
    group = module.ProductGroup("Product Name", "Product_Name", product_dir, [png_path])

    metadata = module.build_batch_metadata(
        args,
        groups=[group],
        manifest_path=repo / "outputs" / "manifests" / "real_png_batch_manifest.json",
        geometry_by_product={
            "Product Name": {
                "chip_blocks": {"width": 2, "height": 2},
                "grid": {"rows": 2, "cols": 2},
                "actual_net_die": 4,
            }
        },
    )

    assert metadata["geometry_contract"] == "explicit"
    assert metadata["manifest_path"].endswith("outputs\\manifests\\real_png_batch_manifest.json") or metadata[
        "manifest_path"
    ].endswith("outputs/manifests/real_png_batch_manifest.json")
    assert metadata["cpu_model_scoring"] is True


def test_png_raw_folder_batch_rejects_invalid_explicit_geometry():
    module = _load_png_batch_script()

    with pytest.raises(ValueError, match="width/height must be positive"):
        module.normalize_geometry({"chip_blocks": {"width": 0, "height": 2}, "grid": {"rows": 2, "cols": 2}})

    with pytest.raises(ValueError, match="cannot exceed grid"):
        module.normalize_geometry(
            {
                "chip_blocks": {"width": 2, "height": 2},
                "grid": {"rows": 2, "cols": 2},
                "actual_net_die": 5,
            }
        )


def test_load_npz_rejects_invalid_values_before_cast(tmp_path):
    module = _load_real_script()
    arrays = _npz_arrays()
    arrays["wafer_mask"] = arrays["wafer_mask"].astype(np.float32)
    arrays["wafer_mask"][0, 0] = 0.5
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, **arrays)
    entry = _manifest_entry(npz_path)

    with pytest.raises(ValueError, match="wafer_mask must contain binary 0/1 values"):
        module.load_real_like_sample(entry, tmp_path / "manifest.json")


def test_load_npz_rejects_non_integer_severity_before_cast(tmp_path):
    module = _load_real_script()
    arrays = _npz_arrays()
    arrays["severity"] = arrays["severity"].astype(np.float32)
    arrays["severity"][0, 0] = 1.5
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, **arrays)
    entry = _manifest_entry(npz_path)

    with pytest.raises(ValueError, match="severity must contain integer grade values"):
        module.load_real_like_sample(entry, tmp_path / "manifest.json")


def test_real_input_inside_workspace_is_allowed(tmp_path, monkeypatch):
    module = _load_real_script()
    monkeypatch.setattr(module, "ROOT", tmp_path.parent)
    arrays = _npz_arrays()
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, **arrays)
    entry = _manifest_entry(npz_path)

    sample = module.load_real_like_sample(entry, tmp_path / "manifest.json")

    assert sample.sample_id == entry["sample_id"]


def test_output_paths_can_be_anywhere(tmp_path, monkeypatch):
    module = _load_real_script()

    allowed = module.resolve_output_path(tmp_path / "outputs" / "report.json")

    assert allowed == (tmp_path / "outputs" / "report.json").resolve()
    assert module.resolve_output_path(tmp_path / "leak.json") == (tmp_path / "leak.json").resolve()


def test_invalid_real_samples_write_sanity_under_output_root(tmp_path, monkeypatch):
    module = _load_real_script()
    output_root = tmp_path / "outputs"
    arrays = _npz_arrays()
    arrays["valid_test_mask"] = np.zeros((4, 4), dtype=np.uint8)
    arrays["stby_mask"] = np.zeros((4, 4), dtype=np.uint8)
    arrays["severity"] = np.zeros((4, 4), dtype=np.uint8)
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, **arrays)
    entry = _manifest_entry(npz_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "real_unlabeled_manifest/v1",
                "feature_schema_version": "observable_fbm_features/v1",
                "samples": [entry],
            }
        ),
        encoding="utf-8",
    )
    sanity_out = output_root / "bad_sanity.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_real_unlabeled_features.py",
            "--manifest",
            str(manifest_path),
            "--features-out",
            str(output_root / "features.csv"),
            "--sanity-out",
            str(sanity_out),
            "--report-out",
            str(output_root / "report.html"),
            "--neighbors-out",
            str(output_root / "neighbors.csv"),
            "--review-template-out",
            str(output_root / "review.csv"),
        ],
    )

    with pytest.raises(SystemExit, match="No valid samples"):
        module.main()

    payload = json.loads(sanity_out.read_text(encoding="utf-8"))
    assert payload["samples"][0]["errors"] == ["sample has no valid tested pixels"]


def test_neighbor_rows_do_not_copy_reference_labels_by_default():
    module = _load_real_script()
    query_rows = [{"sample_id": "q", "total_fail_density": 0.1, "stby_ratio": 0.0}]
    reference_rows = [
        {"sample_id": "r1", "total_fail_density": "0.1", "stby_ratio": "0.0", "label_edge": "1"},
        {"sample_id": "r2", "total_fail_density": "0.9", "stby_ratio": "0.0", "label_edge": "0"},
    ]

    rows = module.nearest_neighbor_rows(query_rows, reference_rows, top_k=1, include_reference_labels=False)

    assert rows[0]["neighbor_sample_id"] == "r1"
    assert "label_edge" not in rows[0]


def test_real_unlabeled_global_neighbors_exclude_location_aware_polar_features():
    module = _load_real_script()
    query_rows = [
        {
            "sample_id": "q",
            "total_fail_density": 0.1,
            "stby_ratio": 0.0,
            "polar_r00_a00_severity": 100.0,
            "stby_polar_r00_a00_ratio": 100.0,
        }
    ]
    reference_rows = [
        {
            "sample_id": "compact_match",
            "total_fail_density": "0.1",
            "stby_ratio": "0.0",
            "polar_r00_a00_severity": "0.0",
            "stby_polar_r00_a00_ratio": "0.0",
        },
        {
            "sample_id": "polar_match_only",
            "total_fail_density": "0.9",
            "stby_ratio": "0.8",
            "polar_r00_a00_severity": "100.0",
            "stby_polar_r00_a00_ratio": "100.0",
        },
    ]

    rows = module.nearest_neighbor_rows(query_rows, reference_rows, top_k=1, include_reference_labels=False)

    assert rows[0]["neighbor_sample_id"] == "compact_match"


def test_feature_drift_summary_uses_compact_observable_features_only():
    module = _load_real_script()
    query_rows = [
        {
            "sample_id": "q",
            "total_fail_density": 0.4,
            "stby_ratio": 0.0,
            "polar_r00_a00_severity": 99.0,
            "label_edge": 1,
        }
    ]
    reference_rows = [
        {
            "sample_id": "r1",
            "total_fail_density": "0.1",
            "stby_ratio": "0.0",
            "polar_r00_a00_severity": "99.0",
            "label_edge": "1",
        },
        {
            "sample_id": "r2",
            "total_fail_density": "0.2",
            "stby_ratio": "0.0",
            "polar_r00_a00_severity": "0.0",
            "label_edge": "0",
        },
    ]

    summary = module.feature_drift_summary(query_rows, reference_rows)
    features = {item["feature"] for item in summary["top_shifted_features"]}

    assert summary["compared_feature_count"] == 2
    assert "total_fail_density" in features
    assert "polar_r00_a00_severity" not in features
    assert "label_edge" not in features

