import importlib.util
import json
import re
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
        "pseudonymized": True,
        "parser_name": "unit_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "chip_blocks": {"width": 2, "height": 2},
        "grid": {"rows": 2, "cols": 2},
        "actual_net_die": 4,
        "allow_workspace_input": True,
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
        "pseudonymized": True,
        "parser_name": "unit_png_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "chip_blocks": {"width": 2, "height": 2},
        "grid": {"rows": 2, "cols": 2},
        "allow_workspace_input": True,
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
        "pseudonymized": True,
        "parser_name": "unit_png_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "allow_geometry_inference": True,
        "allow_workspace_input": True,
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
        "pseudonymized": True,
        "parser_name": "unit_png_parser",
        "parser_version": "0.1.0",
        "orientation": "not_rotated",
        "chip_blocks": {"width": 1, "height": 1},
        "grid": {"rows": 2, "cols": 2},
        "allow_workspace_input": True,
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
        use_folder_names=False,
    )
    manifest = module.build_manifest(
        groups,
        geometry_by_product={},
        wafer_mask_strategy="centered_ellipse_from_png",
        orientation="not_rotated",
        allow_workspace_input=True,
    )

    real_module.validate_manifest(manifest)
    entry = manifest["samples"][0]
    assert re.fullmatch(r"product_[0-9a-f]{10}_w[0-9a-f]{10}", entry["sample_id"])
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
        use_folder_names=False,
    )

    with pytest.raises(ValueError, match="Inconsistent inferred chip geometry"):
        module.build_manifest(
            groups,
            geometry_by_product={},
            wafer_mask_strategy="centered_ellipse_from_png",
            orientation="not_rotated",
            allow_workspace_input=True,
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
        use_folder_names=False,
    )

    manifest = module.build_manifest(
        groups,
        geometry_by_product={"prod_b": {"chip_blocks": {"width": 2, "height": 2}, "grid": {"rows": 2, "cols": 2}}},
        wafer_mask_strategy="centered_ellipse_from_png",
        orientation="not_rotated",
        allow_workspace_input=True,
    )

    entry = manifest["samples"][0]
    assert re.fullmatch(r"product_[0-9a-f]{10}_w[0-9a-f]{10}", entry["sample_id"])
    assert entry["chip_blocks"] == {"width": 2, "height": 2}


def test_png_raw_folder_batch_rejects_folder_name_sample_ids(tmp_path):
    module = _load_png_batch_script()
    product_dir = tmp_path / "raw" / "Sensitive Product Name"
    product_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(product_dir / "wafer_001.png")

    with pytest.raises(ValueError, match="sample_id must stay opaque"):
        module.discover_products(
            tmp_path / "raw",
            glob_pattern="*.png",
            recursive=True,
            limit_per_product=None,
            use_folder_names=True,
        )


def test_png_raw_folder_batch_restricts_workspace_manifest_output():
    module = _load_png_batch_script()
    leaky_path = module.ROOT / "outputs" / "reports" / "real_paths_manifest.json"
    private_path = module.ROOT / "outputs" / "private" / "real_paths_manifest.json"

    with pytest.raises(ValueError, match="outputs/private"):
        module.validate_manifest_output_path(leaky_path, allow_workspace_manifest_output=False)

    module.validate_manifest_output_path(private_path, allow_workspace_manifest_output=False)
    module.validate_manifest_output_path(leaky_path, allow_workspace_manifest_output=True)


def test_png_raw_folder_batch_production_run_requires_explicit_geometry(tmp_path, monkeypatch):
    module = _load_png_batch_script()
    repo = tmp_path / "repo"
    raw_root = tmp_path / "secure_raw"
    product_dir = raw_root / "prod_a"
    product_dir.mkdir(parents=True)
    png_path = product_dir / "wafer_001.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(png_path)
    monkeypatch.setattr(module, "ROOT", repo)
    args = module.parse_args(
        [
            "--raw-root",
            str(raw_root),
            "--production-run",
            "--out-dir",
            "outputs/reports/real_png_batch",
            "--reference-features",
            "outputs/pre_real_readiness/reports/synthetic_reference_features.csv",
        ]
    )
    group = module.ProductGroup("prod_a", "product_aaaaaaaaaa", product_dir, [png_path])

    with pytest.raises(ValueError, match="requires --geometry-json"):
        module.validate_production_run(
            args,
            raw_root=raw_root,
            out_dir=repo / "outputs" / "reports" / "real_png_batch",
            manifest_path=repo / "outputs" / "private" / "real_png_batch_manifest.json",
            groups=[group],
            geometry_by_product={},
        )


def test_png_raw_folder_batch_production_run_requires_positive_actual_net_die(tmp_path, monkeypatch):
    module = _load_png_batch_script()
    repo = tmp_path / "repo"
    raw_root = tmp_path / "secure_raw"
    product_dir = raw_root / "prod_a"
    product_dir.mkdir(parents=True)
    png_path = product_dir / "wafer_001.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(png_path)
    monkeypatch.setattr(module, "ROOT", repo)
    args = module.parse_args(
        [
            "--raw-root",
            str(raw_root),
            "--production-run",
            "--geometry-json",
            "secure_geometry.json",
            "--out-dir",
            "outputs/reports/real_png_batch",
            "--reference-features",
            "outputs/pre_real_readiness/reports/synthetic_reference_features.csv",
        ]
    )
    group = module.ProductGroup("prod_a", "product_aaaaaaaaaa", product_dir, [png_path])

    with pytest.raises(ValueError, match="positive actual_net_die"):
        module.validate_production_run(
            args,
            raw_root=raw_root,
            out_dir=repo / "outputs" / "reports" / "real_png_batch",
            manifest_path=repo / "outputs" / "private" / "real_png_batch_manifest.json",
            groups=[group],
            geometry_by_product={
                "prod_a": {
                    "chip_blocks": {"width": 2, "height": 2},
                    "grid": {"rows": 2, "cols": 2},
                    "actual_net_die": 0,
                }
            },
        )


def test_png_raw_folder_batch_production_run_accepts_guarded_contract(tmp_path, monkeypatch):
    module = _load_png_batch_script()
    repo = tmp_path / "repo"
    raw_root = tmp_path / "secure_raw"
    product_dir = raw_root / "prod_a"
    product_dir.mkdir(parents=True)
    png_path = product_dir / "wafer_001.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(png_path)
    monkeypatch.setattr(module, "ROOT", repo)
    args = module.parse_args(
        [
            "--raw-root",
            str(raw_root),
            "--production-run",
            "--geometry-json",
            "secure_geometry.json",
            "--out-dir",
            "outputs/reports/real_png_batch",
            "--reference-features",
            "outputs/pre_real_readiness/reports/synthetic_reference_features.csv",
        ]
    )
    group = module.ProductGroup("prod_a", "product_aaaaaaaaaa", product_dir, [png_path])

    module.validate_production_run(
        args,
        raw_root=raw_root,
        out_dir=repo / "outputs" / "reports" / "real_png_batch",
        manifest_path=repo / "outputs" / "private" / "real_png_batch_manifest.json",
        groups=[group],
        geometry_by_product={
            "prod_a": {
                "chip_blocks": {"width": 2, "height": 2},
                "grid": {"rows": 2, "cols": 2},
                "actual_net_die": 4,
            }
        },
    )


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


def test_png_raw_folder_batch_metadata_is_shareable_without_product_names(tmp_path, monkeypatch):
    module = _load_png_batch_script()
    repo = tmp_path / "repo"
    raw_root = tmp_path / "secure_raw"
    product_dir = raw_root / "Secret Product Name"
    product_dir.mkdir(parents=True)
    png_path = product_dir / "wafer_001.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(png_path)
    monkeypatch.setattr(module, "ROOT", repo)
    args = module.parse_args(
        [
            "--raw-root",
            str(raw_root),
            "--production-run",
            "--geometry-json",
            "secure_geometry.json",
            "--out-dir",
            "outputs/reports/real_png_batch",
            "--reference-features",
            "outputs/pre_real_readiness/reports/synthetic_reference_features.csv",
            "--cpu-model",
            "outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz",
        ]
    )
    group = module.ProductGroup("Secret Product Name", "product_aaaaaaaaaa", product_dir, [png_path])

    metadata = module.build_batch_metadata(
        args,
        groups=[group],
        manifest_path=repo / "outputs" / "private" / "real_png_batch_manifest.json",
        geometry_by_product={
            "Secret Product Name": {
                "chip_blocks": {"width": 2, "height": 2},
                "grid": {"rows": 2, "cols": 2},
                "actual_net_die": 4,
            }
        },
    )

    text = json.dumps(metadata, ensure_ascii=False)
    assert metadata["production_run"] is True
    assert metadata["geometry_contract"] == "explicit"
    assert metadata["manifest_location"] == "outputs/private"
    assert metadata["reference_features"] is True
    assert metadata["cpu_model_scoring"] is True
    assert "Secret Product Name" not in text
    assert str(raw_root) not in text


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


def test_real_input_inside_workspace_requires_explicit_allow_flag(tmp_path, monkeypatch):
    module = _load_real_script()
    monkeypatch.setattr(module, "ROOT", tmp_path.parent)
    arrays = _npz_arrays()
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, **arrays)
    entry = _manifest_entry(npz_path)
    entry.pop("allow_workspace_input")

    with pytest.raises(ValueError, match="arrays_npz must live outside the workspace"):
        module.load_real_like_sample(entry, tmp_path / "manifest.json")


def test_output_paths_are_restricted_to_output_root(tmp_path, monkeypatch):
    module = _load_real_script()
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "outputs")

    allowed = module.ensure_output_path(tmp_path / "outputs" / "report.json")

    assert allowed == (tmp_path / "outputs" / "report.json").resolve()
    with pytest.raises(ValueError, match="Output path must be under"):
        module.ensure_output_path(tmp_path / "leak.json")
    assert module.ensure_output_path(tmp_path / "leak.json", allow_outside_root=True) == (tmp_path / "leak.json").resolve()


def test_invalid_real_samples_write_sanity_under_output_root(tmp_path, monkeypatch):
    module = _load_real_script()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
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
