import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run_pre_real_readiness.py"
    spec = importlib.util.spec_from_file_location("run_pre_real_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_pre_real_readiness_pipeline_runs_end_to_end(tmp_path):
    module = _load_script()
    config = tmp_path / "small_synth.json"
    config.write_text(
        json.dumps(
            {
                "seed": 29,
                "count": 5,
                "target_net_die": 36,
                "chip_blocks": {"width": 8, "height": 6},
                "stby_chips": {"min": 2, "max": 4},
                "pattern_probabilities": {
                    "scratch": 0.5,
                    "ring": 0.4,
                    "edge": 0.5,
                    "local": 1.0,
                    "random": 1.0,
                    "shot_grid": 0.5,
                    "stby_pattern": 1.0,
                },
                "grade_thresholds": [0.055, 0.115, 0.205, 0.340, 0.530, 0.760, 1.050],
            }
        ),
        encoding="utf-8",
    )

    out_root = tmp_path / "readiness"
    stale_dir = out_root / "synthetic" / "synth_stale"
    stale_dir.mkdir(parents=True)
    (stale_dir / "raw_grayscale.png").write_bytes(b"stale")

    module.main(
        [
            "--config",
            str(config),
            "--out-root",
            str(out_root),
            "--count",
            "5",
            "--output-size",
            "12",
            "--hidden-dim",
            "8",
            "--embedding-dim",
            "4",
            "--epochs",
            "2",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "2",
            "--score-sample-count",
            "2",
            "--top-k",
            "2",
        ]
    )

    summary_path = out_root / "reports" / "pre_real_readiness_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "CHECK"
    assert any("synthetic sample count is small" in issue for issue in payload["readiness_issues"])
    assert payload["total_elapsed_seconds"] > 0
    assert any(step["name"] == "analyze synthetic raw png batch" for step in payload["steps"])
    assert all(item["exists"] for item in payload["output_checks"])
    assert any(item["name"] == "synthetic_unlabeled_predictions" and item["row_count"] == 2 for item in payload["output_checks"])
    assert any(item["name"] == "synthetic_png_batch_features" and item["row_count"] == 5 for item in payload["output_checks"])
    assert Path(payload["outputs"]["cpu_encoder_model"]).exists()
    assert Path(payload["outputs"]["synthetic_unlabeled_predictions"]).exists()
    assert Path(payload["outputs"]["synthetic_unlabeled_neighbors"]).exists()
    assert Path(payload["outputs"]["synthetic_png_batch_report"]).exists()
    assert "data/raw" in payload["real_png_batch_command"]
    assert "data/raw/product_geometry.json" in payload["real_png_batch_command"]
