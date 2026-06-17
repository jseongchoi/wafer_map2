import importlib.util
import json
from pathlib import Path

from wafermap.synth import SyntheticConfig, generate_sample, save_sample


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "build_segmentation_readiness.py"
    spec = importlib.util.spec_from_file_location("build_segmentation_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_segmentation_readiness_outputs_manifest_and_metrics(tmp_path):
    module = _load_script()
    data_root = tmp_path / "synthetic"
    for idx in range(3):
        sample = generate_sample(
            SyntheticConfig(count=3, target_net_die=30, chip_width=8, chip_height=6, seed=31),
            idx,
        )
        save_sample(sample, data_root / sample.sample_id)

    report = tmp_path / "report.html"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "manifest.csv"
    gallery = tmp_path / "gallery.png"
    module.main(
        [
            "--data",
            str(data_root),
            "--out",
            str(report),
            "--metrics",
            str(metrics),
            "--manifest",
            str(manifest),
            "--gallery",
            str(gallery),
            "--max-gallery-rows",
            "2",
        ]
    )

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 3
    assert "scratch" in payload["target_channels"]
    assert any(row["class"] == "scratch" for row in payload["class_summary"])
    assert report.exists()
    assert manifest.exists()
    assert gallery.exists()
