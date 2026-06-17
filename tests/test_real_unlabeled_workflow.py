import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wafermap.data import PATTERN_CLASSES, SyntheticSample


def _load_real_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
        "sample_id": "real_like_npz",
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

    assert sample.sample_id == "real_like_npz"
    assert sample.severity.dtype == np.uint8
    assert errors == []
    assert warnings == []


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
