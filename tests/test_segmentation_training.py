import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wafermap.assets import load_prediction_masks, mask_to_rle
from wafermap.data import PATTERN_CLASSES
from wafermap.training.segmentation import INPUT_CHANNELS, TARGET_CHANNELS, sample_to_input_tensor, sample_to_target_tensor
from wafermap.synth import SyntheticConfig, generate_sample, save_sample


ROOT = Path(__file__).resolve().parents[1]


def test_segmentation_input_tensor_keeps_mean_max_and_density_signals():
    severity = np.zeros((4, 4), dtype=np.uint8)
    severity[1, 2] = 7
    sample = SimpleNamespace(
        shape=severity.shape,
        severity=severity,
        wafer_mask=np.ones_like(severity, dtype=np.uint8),
        valid_test_mask=np.ones_like(severity, dtype=np.uint8),
        stby_mask=np.zeros_like(severity, dtype=np.uint8),
    )

    tensor = sample_to_input_tensor(sample, output_size=1)

    assert INPUT_CHANNELS[:3] == ("severity_mean", "severity_max", "fail_density")
    assert tensor[INPUT_CHANNELS.index("severity_mean"), 0, 0] == pytest.approx(1.0 / 16.0)
    assert tensor[INPUT_CHANNELS.index("severity_max"), 0, 0] == pytest.approx(1.0)
    assert tensor[INPUT_CHANNELS.index("fail_density"), 0, 0] == pytest.approx(1.0 / 16.0)
    assert INPUT_CHANNELS[3] == "wafer_mask"


def test_segmentation_target_tensor_excludes_invalid_test_pixels():
    severity = np.zeros((2, 2), dtype=np.uint8)
    pattern_masks = np.zeros((len(PATTERN_CLASSES), 2, 2), dtype=np.uint8)
    local_pattern_idx = PATTERN_CLASSES.index("local")
    local_target_idx = TARGET_CHANNELS.index("local")
    pattern_masks[local_pattern_idx, 0, 0] = 1
    pattern_masks[local_pattern_idx, 1, 1] = 1
    sample = SimpleNamespace(
        shape=severity.shape,
        severity=severity,
        wafer_mask=np.ones_like(severity, dtype=np.uint8),
        valid_test_mask=np.array([[0, 1], [1, 1]], dtype=np.uint8),
        stby_mask=np.array([[1, 0], [0, 0]], dtype=np.uint8),
        pattern_masks=pattern_masks,
    )

    target = sample_to_target_tensor(sample, output_size=2)

    assert target[local_target_idx, 0, 0] == 0.0
    assert target[local_target_idx, 1, 1] == 1.0


@pytest.mark.slow
def test_segmentation_input_tensor_includes_position_channels():
    sample = generate_sample(
        SyntheticConfig(count=1, target_net_die=36, chip_width=8, chip_height=6, seed=83),
        0,
    )

    tensor = sample_to_input_tensor(sample, output_size=24)

    assert tensor.shape == (len(INPUT_CHANNELS), 24, 24)
    for channel in ("x_norm", "y_norm", "radial_norm", "angle_sin", "angle_cos", "edge_distance_norm"):
        assert channel in INPUT_CHANNELS
    assert tensor[INPUT_CHANNELS.index("radial_norm")].max() <= 1.0
    assert tensor[INPUT_CHANNELS.index("edge_distance_norm")].min() >= 0.0


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_segmentation_smoke_training_runs_from_manifest(tmp_path):
    readiness = _load_script("build_segmentation_readiness.py")
    smoke = _load_script("train_segmentation_smoke.py")
    data_root = tmp_path / "synthetic"
    for idx in range(4):
        sample = generate_sample(
            SyntheticConfig(count=4, target_net_die=36, chip_width=8, chip_height=6, seed=91),
            idx,
        )
        save_sample(sample, data_root / sample.sample_id)

    manifest = tmp_path / "manifest.csv"
    readiness.main(
        [
            "--data",
            str(data_root),
            "--out",
            str(tmp_path / "readiness.html"),
            "--metrics",
            str(tmp_path / "readiness.json"),
            "--manifest",
            str(manifest),
            "--gallery",
            str(tmp_path / "readiness.png"),
            "--max-gallery-rows",
            "2",
        ]
    )
    report = tmp_path / "smoke.html"
    metrics = tmp_path / "smoke.json"
    smoke.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(report),
            "--metrics",
            str(metrics),
            "--figure",
            str(tmp_path / "loss.png"),
            "--output-size",
            "24",
            "--max-train-samples",
            "3",
            "--max-val-samples",
            "1",
            "--steps",
            "2",
        ]
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["train_samples"] >= 1
    assert payload["loss"]["final"] == payload["loss"]["final"]
    assert report.exists()


@pytest.mark.slow
def test_embedding_smoke_training_runs_from_manifest(tmp_path):
    readiness = _load_script("build_segmentation_readiness.py")
    smoke = _load_script("train_embedding_smoke.py")
    data_root = tmp_path / "synthetic"
    for idx in range(6):
        sample = generate_sample(
            SyntheticConfig(count=6, target_net_die=36, chip_width=8, chip_height=6, seed=117),
            idx,
        )
        save_sample(sample, data_root / sample.sample_id)

    manifest = tmp_path / "manifest.csv"
    readiness.main(
        [
            "--data",
            str(data_root),
            "--out",
            str(tmp_path / "readiness.html"),
            "--metrics",
            str(tmp_path / "readiness.json"),
            "--manifest",
            str(manifest),
            "--gallery",
            str(tmp_path / "readiness.png"),
            "--val-fraction",
            "0.5",
            "--max-gallery-rows",
            "2",
        ]
    )
    report = tmp_path / "embedding.html"
    metrics = tmp_path / "embedding.json"
    embeddings = tmp_path / "embedding.csv"
    smoke.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(report),
            "--metrics",
            str(metrics),
            "--embeddings-out",
            str(embeddings),
            "--output-size",
            "16",
            "--embedding-dim",
            "4",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "2",
            "--top-k",
            "2",
        ]
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["train_samples"] >= 1
    assert payload["effective_embedding_dim"] >= 1
    assert payload["retrieval"]["top1_mean_jaccard"] == payload["retrieval"]["top1_mean_jaccard"]
    assert report.exists()
    assert embeddings.exists()


def test_unet_training_dependency_check_writes_report(tmp_path):
    unet = _load_script("train_unet_segmentation.py")
    report = tmp_path / "unet.html"
    metrics = tmp_path / "unet.json"

    unet.main(
        [
            "--out",
            str(report),
            "--metrics",
            str(metrics),
            "--check-deps",
        ]
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["status"] == "dependency_check"
    assert "torch_available" in payload
    assert payload["manifest_target_coverage"]["status"] in {"MISSING", "PASS", "CHECK"}
    assert report.exists()


def test_unet_prediction_export_resize_nearest_expands_source_cells():
    export = _load_script("export_unet_predictions.py")
    mask = np.array([[1, 0], [0, 1]], dtype=bool)

    resized = export.resize_mask_nearest(mask, (4, 4))

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(resized, expected)


def test_unet_prediction_export_schema_round_trips_to_tool_loader(tmp_path):
    export = _load_script("export_unet_predictions.py")
    local = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)

    record = export.prediction_record("wafer_a", {"local": local})
    payload = export.prediction_payload(
        [record],
        model_path=Path("outputs/models/asset_unet_segmentation.pt"),
        threshold=0.35,
    )
    prediction_path = tmp_path / "fbm_prediction_masks.json"
    prediction_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = load_prediction_masks(prediction_path, "wafer_a", local.shape)

    assert loaded["local"] == mask_to_rle(local)
    for family in TARGET_CHANNELS:
        assert family in loaded


def test_unet_prediction_export_dependency_check_writes_status(tmp_path):
    export = _load_script("export_unet_predictions.py")
    out = tmp_path / "dependency_status.json"

    export.main(["--out", str(out), "--check-deps"])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "dependency_check"
    assert payload["prediction_schema"] == "fbm_prediction_masks/v1"
    assert payload["target_channels"] == list(TARGET_CHANNELS)


def test_unet_manifest_target_coverage_flags_missing_train_classes(tmp_path):
    unet = _load_script("train_unet_segmentation.py")
    manifest = tmp_path / "manifest.csv"
    fieldnames = ["sample_id", "split", *[f"has_{name}" for name in TARGET_CHANNELS]]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "only_local",
                "split": "train",
                **{f"has_{name}": int(name == "local") for name in TARGET_CHANNELS},
            }
        )
        writer.writerow(
            {
                "sample_id": "val_local",
                "split": "val",
                **{f"has_{name}": int(name == "local") for name in TARGET_CHANNELS},
            }
        )

    coverage = unet.manifest_target_coverage(manifest, min_positive_samples_per_class=1)

    assert coverage["status"] == "CHECK"
    assert "local" not in coverage["missing_classes"]
    assert "scratch" in coverage["missing_classes"]
    with pytest.raises(ValueError, match="positive train samples"):
        unet.require_manifest_target_coverage(
            SimpleNamespace(
                manifest=str(manifest),
                min_positive_samples_per_class=1,
                allow_incomplete_target_coverage=False,
            )
        )
    allowed = unet.require_manifest_target_coverage(
        SimpleNamespace(
            manifest=str(manifest),
            min_positive_samples_per_class=1,
            allow_incomplete_target_coverage=True,
        )
    )
    assert allowed["status"] == "CHECK"


def test_unet_manifest_target_coverage_reports_missing_validation_without_blocking(tmp_path):
    unet = _load_script("train_unet_segmentation.py")
    manifest = tmp_path / "manifest.csv"
    fieldnames = ["sample_id", "split", *[f"has_{name}" for name in TARGET_CHANNELS]]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "train_all",
                "split": "train",
                **{f"has_{name}": 1 for name in TARGET_CHANNELS},
            }
        )
        writer.writerow(
            {
                "sample_id": "val_local_only",
                "split": "val",
                **{f"has_{name}": int(name == "local") for name in TARGET_CHANNELS},
            }
        )

    coverage = unet.require_manifest_target_coverage(
        SimpleNamespace(
            manifest=str(manifest),
            min_positive_samples_per_class=1,
            allow_incomplete_target_coverage=False,
        )
    )

    assert coverage["status"] == "CHECK"
    assert coverage["train_status"] == "PASS"
    assert coverage["validation_status"] == "CHECK"
    assert coverage["blocking_issues"] == []
    assert "scratch" in coverage["missing_val_classes"]
