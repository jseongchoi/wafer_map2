import importlib.util
import json
from pathlib import Path

from wafermap.synth import SyntheticConfig, generate_sample, save_sample


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
