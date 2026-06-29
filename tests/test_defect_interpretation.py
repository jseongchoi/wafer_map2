import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from wafermap.reporting import score_feature_row
from wafermap.synth import SyntheticConfig, generate_sample, save_sample


def _load_interpret_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "interpret_fbm.py"
    spec = importlib.util.spec_from_file_location("interpret_fbm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_feature_row_prioritizes_structured_edge_signal():
    row = {
        "sample_id": "wafer_edge",
        "edge_density": 0.30,
        "edge_minus_center_density": 0.16,
        "edge_chip_outer_minus_inner_density": 0.18,
        "edge_chip_peak_contrast": 0.25,
        "edge_sector_peak_contrast": 0.18,
        "total_fail_density": 0.22,
        "grade_weighted_severity": 0.12,
        "polar_r02_a03_fail_density": 0.75,
    }

    scores = score_feature_row(row)

    assert scores[0].defect_family == "edge"
    assert scores[0].score >= 70
    assert scores[0].confidence == "high"
    assert "edge_density" in scores[0].evidence
    assert "edge near" in scores[0].location


def test_center_polar_location_does_not_invent_clock_position():
    row = {
        "sample_id": "wafer_local",
        "local_hotspot_peak_contrast": 0.30,
        "local_hotspot_top3_mean_contrast": 0.20,
        "local_component_compactness": 0.60,
        "polar_r00_a00_severity": 0.80,
    }

    scores = score_feature_row(row)
    local = [item for item in scores if item.defect_family == "local"][0]

    assert local.location == "hotspot near center / radial zone 0"


def test_interpret_fbm_existing_feature_csv_writes_compact_user_outputs(tmp_path):
    module = _load_interpret_script()
    features = tmp_path / "features.csv"
    sanity = tmp_path / "sanity.json"
    out = tmp_path / "interpretation"
    with features.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "edge_density",
                "edge_minus_center_density",
                "edge_chip_outer_minus_inner_density",
                "edge_chip_peak_contrast",
                "edge_sector_peak_contrast",
                "ring_radial_peak_contrast",
                "ring_radial_peak_width_ratio",
                "local_hotspot_peak_contrast",
                "local_hotspot_top3_mean_contrast",
                "scratch_component_linear_score",
                "shot_best_contrast",
                "stby_ratio",
                "total_fail_density",
                "grade_weighted_severity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "wafer_001",
                "edge_density": 0.25,
                "edge_minus_center_density": 0.14,
                "edge_chip_outer_minus_inner_density": 0.14,
                "edge_chip_peak_contrast": 0.20,
                "edge_sector_peak_contrast": 0.18,
                "ring_radial_peak_contrast": 0.04,
                "ring_radial_peak_width_ratio": 0.10,
                "local_hotspot_peak_contrast": 0.02,
                "local_hotspot_top3_mean_contrast": 0.01,
                "scratch_component_linear_score": 0.0,
                "shot_best_contrast": 0.0,
                "stby_ratio": 0.0,
                "total_fail_density": 0.20,
                "grade_weighted_severity": 0.11,
            }
        )
    sanity.write_text(
        json.dumps({"samples": [{"sample_id": "wafer_001", "errors": [], "warnings": []}]}),
        encoding="utf-8",
    )

    module.main(
        [
            "--features-csv",
            str(features),
            "--sanity-json",
            str(sanity),
            "--out",
            str(out),
        ]
    )

    assert (out / "defect_scores.csv").exists()
    assert (out / "wafer_interpretation_report.html").exists()
    assert (out / "sanity_summary.json").exists()
    assert (out / "similar_wafers.csv").exists()
    rows = list(csv.DictReader((out / "defect_scores.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["sample_id"] == "wafer_001"
    assert rows[0]["defect_family"] == "edge"
    summary = json.loads((out / "sanity_summary.json").read_text(encoding="utf-8"))
    assert summary["overall_status"] == "PASS"
    report_text = (out / "wafer_interpretation_report.html").read_text(encoding="utf-8")
    assert "FBM One-Wafer Defect Review" in report_text
    assert "Focus Wafer" in report_text
    assert "Focus Wafer Score Table" in report_text


@pytest.mark.slow
def test_render_wafer_previews_from_manifest(tmp_path):
    module = _load_interpret_script()
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=3), 0)
    sample_dir = tmp_path / "sample"
    save_sample(sample, sample_dir)
    manifest = {
        "schema_version": "real_unlabeled_manifest/v1",
        "feature_schema_version": "observable_fbm_features/v1",
        "samples": [
            {
                "sample_id": "preview_unit",
                "source_type": "synthetic_sample_dir",
                "sample_dir": str(sample_dir),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "out"

    samples = module.load_manifest_samples(manifest_path)
    image_map = module.render_wafer_previews_from_samples(samples, out)

    image_path = image_map["preview_unit"]
    assert image_path.exists()
    assert image_path.parent.name == "wafer_images"
    with Image.open(image_path) as image:
        assert image.width > 0
        assert image.height > 0


@pytest.mark.slow
def test_render_annotated_preview_marks_local_candidate(tmp_path):
    module = _load_interpret_script()
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=3), 0)
    sample.severity[:] = 0
    sample.valid_test_mask[:] = 1
    sample.stby_mask[:] = 0
    sample.severity[-6:, sample.shape[1] // 2 - 2 : sample.shape[1] // 2 + 2] = 7
    rows = [
        {
            "sample_id": sample.sample_id,
            "defect_family": "local",
            "score": 88.0,
            "confidence": "high",
            "location": "hotspot",
            "evidence": "unit-test",
        }
    ]
    out = tmp_path / "out"

    masks = module.defect_overlay_masks(sample, rows)
    image_map = module.render_annotated_previews_from_samples(
        [sample],
        rows,
        out,
        focus_sample_id=sample.sample_id,
    )

    assert "local" in masks
    assert masks["local"].any()
    image_path = image_map[sample.sample_id]
    assert image_path.exists()
    assert image_path.parent.name == "annotated_images"
    with Image.open(image_path) as image:
        assert image.width > 0
        assert image.height > 0
