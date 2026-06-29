import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wafermap.data import PATTERN_CLASSES, SyntheticSample
from wafermap.synth import SyntheticConfig, generate_sample, save_sample
from wafermap.training.cpu_encoder import load_cpu_encoder_model


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_cpu_encoder_train_and_score_pipeline(tmp_path):
    readiness = _load_script("build_segmentation_readiness.py")
    train_cpu = _load_script("train_cpu_encoder_model.py")
    score_cpu = _load_script("score_unlabeled_cpu_encoder.py")
    data_root = tmp_path / "synthetic"
    sample_dirs = []
    for idx in range(6):
        sample = generate_sample(
            SyntheticConfig(count=6, target_net_die=36, chip_width=8, chip_height=6, seed=411),
            idx,
        )
        sample_dir = data_root / sample.sample_id
        save_sample(sample, sample_dir)
        sample_dirs.append(sample_dir)
    invalid = SyntheticSample(
        sample_id="invalid_unlabeled",
        severity=np.zeros_like(sample.severity),
        wafer_mask=np.ones_like(sample.wafer_mask),
        valid_test_mask=np.zeros_like(sample.valid_test_mask),
        stby_mask=np.zeros_like(sample.stby_mask),
        pattern_masks=np.zeros((len(PATTERN_CLASSES), *sample.shape), dtype=np.uint8),
        pattern_intensity=np.zeros((len(PATTERN_CLASSES), *sample.shape), dtype=np.float32),
        chip_index=np.where(sample.wafer_mask > 0, sample.chip_index, -1).astype(np.int32),
        metadata=sample.metadata | {"sample_id": "invalid_unlabeled"},
    )
    invalid_dir = data_root / "invalid_unlabeled"
    save_sample(invalid, invalid_dir)

    manifest = tmp_path / "segmentation_manifest.csv"
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
            "0.35",
            "--max-gallery-rows",
            "2",
        ]
    )

    model = tmp_path / "cpu_model.npz"
    metrics = tmp_path / "cpu_metrics.json"
    train_cpu.main(
        [
            "--manifest",
            str(manifest),
            "--model-out",
            str(model),
            "--out",
            str(tmp_path / "cpu_report.html"),
            "--metrics",
            str(metrics),
            "--predictions-out",
            str(tmp_path / "cpu_val_predictions.csv"),
            "--output-size",
            "12",
            "--hidden-dim",
            "8",
            "--embedding-dim",
            "4",
            "--epochs",
            "3",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "2",
            "--top-k",
            "2",
            "--learning-rate",
            "0.04",
        ]
    )

    score_manifest = tmp_path / "score_manifest.json"
    score_manifest.write_text(
        json.dumps(
            {
                "schema_version": "real_unlabeled_manifest/v1",
                "feature_schema_version": "observable_fbm_features/v1",
                "samples": [
                    {
                        "sample_id": "score_000",
                        "source_type": "synthetic_sample_dir",
                        "sample_dir": str(sample_dirs[0]),
                    },
                    {
                        "sample_id": "score_001",
                        "source_type": "synthetic_sample_dir",
                        "sample_dir": str(sample_dirs[1]),
                    },
                    {
                        "sample_id": "score_invalid",
                        "source_type": "synthetic_sample_dir",
                        "sample_dir": str(invalid_dir),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "score_predictions.csv"
    neighbors = tmp_path / "score_neighbors.csv"
    score_cpu.main(
        [
            "--model",
            str(model),
            "--manifest",
            str(score_manifest),
            "--predictions-out",
            str(predictions),
            "--neighbors-out",
            str(neighbors),
            "--sanity-out",
            str(tmp_path / "score_sanity.json"),
            "--report-out",
            str(tmp_path / "score_report.html"),
            "--top-k",
            "2",
        ]
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["model_version"] == "cpu_shared_encoder/v1"
    assert payload["validation"]["bce"] == payload["validation"]["bce"]
    assert payload["readiness_gate"]["status"] in {"PASS", "CHECK"}
    assert model.exists()
    assert predictions.exists()
    assert neighbors.exists()
    with predictions.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    invalid_rows = [row for row in rows if row["sample_id"] == "score_invalid"]
    assert invalid_rows[0]["scoring_status"] == "SKIPPED"
    assert invalid_rows[0]["top_synthetic_label_hint"] == ""


def test_cpu_encoder_model_rejects_bad_version(tmp_path):
    bad_model = tmp_path / "bad_model.npz"
    np.savez_compressed(
        bad_model,
        metadata_json=np.array(
            json.dumps(
                {
                    "model_version": "cpu_shared_encoder/v0",
                    "output_size": 12,
                    "input_dim": 4,
                    "hidden_dim": 2,
                    "embedding_dim": 2,
                    "label_names": ["scratch"],
                }
            ),
            dtype=np.str_,
        ),
        mean=np.zeros(4, dtype=np.float32),
        scale=np.ones(4, dtype=np.float32),
        w1=np.zeros((4, 2), dtype=np.float32),
        b1=np.zeros(2, dtype=np.float32),
        w2=np.zeros((2, 2), dtype=np.float32),
        b2=np.zeros(2, dtype=np.float32),
        wc=np.zeros((2, 1), dtype=np.float32),
        bc=np.zeros(1, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Unsupported CPU encoder model_version"):
        load_cpu_encoder_model(bad_model)
