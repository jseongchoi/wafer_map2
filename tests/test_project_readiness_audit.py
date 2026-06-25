import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "audit_project_readiness.py"
    spec = importlib.util.spec_from_file_location("audit_project_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_readiness_audit_writes_stage_report(tmp_path):
    module = _load_script()
    pre_real_summary = tmp_path / "pre_real_readiness_summary.json"
    pre_real_summary.write_text(
        json.dumps({"status": "PASS", "output_checks": [{"name": "features", "exists": True}]}),
        encoding="utf-8",
    )

    out = tmp_path / "audit.json"
    html_out = tmp_path / "audit.html"
    module.main(
        [
            "--pre-real-summary",
            str(pre_real_summary),
            "--out",
            str(out),
            "--html-out",
            str(html_out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    stages = {stage["stage_id"]: stage for stage in payload["stages"]}
    assert "2_raw_png_ingestion" in stages
    assert "5_cpu_ai_baseline" in stages
    assert "7_real_go_live_preflight" in stages
    assert "8_real_data_actual_batch" in stages
    assert stages["6_pre_real_readiness"]["status"] in {"PASS", "CHECK"}
    assert html_out.exists()


def test_project_readiness_audit_marks_missing_pre_real_summary_as_check():
    module = _load_script()

    audit = module.build_audit(Path("missing_pre_real_summary.json"))
    stage = next(stage for stage in audit["stages"] if stage["stage_id"] == "6_pre_real_readiness")

    assert stage["status"] == "CHECK"
    assert stage["missing_evidence"]


def test_project_readiness_audit_distinguishes_pre_real_gate_from_missing_file(tmp_path):
    module = _load_script()
    summary = tmp_path / "pre_real_readiness_summary.json"
    summary.write_text(json.dumps({"status": "CHECK", "output_checks": []}), encoding="utf-8")

    audit = module.build_audit(summary)
    stage = next(stage for stage in audit["stages"] if stage["stage_id"] == "6_pre_real_readiness")

    assert stage["status"] == "CHECK"
    assert any("PASS 기준 미달" in item for item in stage["missing_evidence"])


def test_project_readiness_audit_marks_stale_pre_real_summary_as_check(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PRE_REAL_FRESHNESS_INPUTS", ("scripts/pipeline.py",))
    summary = tmp_path / "pre_real_readiness_summary.json"
    summary.write_text(json.dumps({"status": "PASS", "output_checks": []}), encoding="utf-8")
    pipeline = tmp_path / "scripts" / "pipeline.py"
    pipeline.parent.mkdir()
    pipeline.write_text("print('changed')\n", encoding="utf-8")
    os.utime(summary, (100.0, 100.0))
    os.utime(pipeline, (200.0, 200.0))

    readiness = module.read_pre_real_summary(summary)

    assert readiness["status"] == "CHECK"
    assert any("실제 데이터 전 준비 요약이 실행 입력보다 오래됨" in note for note in readiness["notes"])


def test_project_readiness_audit_marks_pre_real_provenance_hash_mismatch(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PRE_REAL_FRESHNESS_INPUTS", ("scripts/pipeline.py",))
    pipeline = tmp_path / "scripts" / "pipeline.py"
    pipeline.parent.mkdir()
    pipeline.write_text("print('current')\n", encoding="utf-8")
    summary = tmp_path / "pre_real_readiness_summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "PASS",
                "output_checks": [],
                "provenance": {
                    "schema_version": "pre_real_readiness_provenance/v1",
                    "config": {"path": "config.json", "sha256": "abc"},
                    "pipeline_inputs": [{"path": "scripts/pipeline.py", "sha256": "wrong"}],
                    "git": {"commit": "abc123", "dirty": False},
                },
            }
        ),
        encoding="utf-8",
    )

    readiness = module.read_pre_real_summary(summary)

    assert readiness["status"] == "CHECK"
    assert any("hash mismatch" in note for note in readiness["notes"])


def test_project_readiness_audit_marks_pre_real_artifact_hash_mismatch(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PRE_REAL_REQUIRED_OUTPUTS", ("reference_features",))
    artifact = tmp_path / "outputs" / "pre_real_readiness" / "reports" / "features.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("sample_id,total_fail_density\nx,0.1\n", encoding="utf-8")
    summary = tmp_path / "pre_real_readiness_summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "PASS",
                "output_checks": [
                    {
                        "name": "reference_features",
                        "path": "outputs/pre_real_readiness/reports/features.csv",
                        "exists": True,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": "wrong",
                    }
                ],
                "provenance": {
                    "schema_version": "pre_real_readiness_provenance/v1",
                    "config": {"path": "config.json", "sha256": "abc"},
                    "pipeline_inputs": [],
                    "git": {"commit": "abc123", "dirty": False},
                },
            }
        ),
        encoding="utf-8",
    )

    readiness = module.read_pre_real_summary(summary)

    assert readiness["status"] == "CHECK"
    assert any("output sha256 mismatch" in note for note in readiness["notes"])


def test_project_readiness_audit_accepts_real_batch_without_production_metadata(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "analyze_png_raw_folders.py").write_text("# script\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real_unlabeled_workflow.md").write_text("# workflow\n", encoding="utf-8")
    manifest_dir = tmp_path / "outputs" / "manifests"
    report_dir = tmp_path / "outputs" / "reports" / "real_png_batch"
    manifest_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    (manifest_dir / "real_png_batch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "real_unlabeled_manifest/v1",
                "feature_schema_version": "observable_fbm_features/v1",
                "samples": [
                    {
                        "sample_id": "q",
                        "source_type": "png_grayscale_raw",
                        "png_path": "data/raw/product_a/wafer.png",
                        "parser_name": "png_raw_folder_batch",
                        "parser_version": "0.1.0",
                        "orientation": "not_rotated",
                        "chip_blocks": {"width": 2, "height": 2},
                        "grid": {"rows": 2, "cols": 2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "features.csv").write_text("sample_id,total_fail_density\nq,0\n", encoding="utf-8")
    (report_dir / "sanity.json").write_text(json.dumps({"samples": [{"sample_id": "q", "errors": []}]}), encoding="utf-8")
    (report_dir / "batch_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "png_raw_batch_metadata/v1",
                "production_run": False,
                "product_count": 1,
                "png_sample_count": 1,
                "geometry_contract": "inferred_from_stby_smoke",
                "reference_features": True,
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (report_dir / "neighbors.csv").write_text("query_sample_id,rank\nq,1\n", encoding="utf-8")
    (report_dir / "review_template.csv").write_text("query_sample_id,reviewer_decision\nq,\n", encoding="utf-8")

    audit = module.build_audit(tmp_path / "missing_pre_real_summary.json")
    stage8 = next(stage for stage in audit["stages"] if stage["stage_id"] == "8_real_data_actual_batch")

    assert stage8["status"] == "PASS"


def test_project_readiness_audit_checks_real_batch_sample_counts(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    report_dir = tmp_path / "outputs" / "reports" / "real_png_batch"
    report_dir.mkdir(parents=True)
    (report_dir / "features.csv").write_text("sample_id,total_fail_density\n", encoding="utf-8")
    (report_dir / "neighbors.csv").write_text("query_sample_id,rank,neighbor_sample_id,distance\nq,1,r,0.1\n", encoding="utf-8")
    (report_dir / "review_template.csv").write_text(
        "review_case_id,query_sample_id,rank,neighbor_sample_id,distance\ncase,q,1,r,0.1\n",
        encoding="utf-8",
    )
    (report_dir / "cpu_encoder_predictions.csv").write_text("sample_id,predicted_label\nq,label\n", encoding="utf-8")
    (report_dir / "sanity.json").write_text(
        json.dumps({"samples": [{"sample_id": "q", "errors": []}]}),
        encoding="utf-8",
    )
    (report_dir / "batch_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "png_raw_batch_metadata/v1",
                "production_run": True,
                "product_count": 1,
                "explicit_geometry_product_count": 1,
                "actual_net_die_product_count": 1,
                "png_sample_count": 2,
                "geometry_contract": "explicit",
                "reference_features": True,
                "cpu_model_scoring": True,
            }
        ),
        encoding="utf-8",
    )

    notes = module.inspect_real_go_live_outputs()

    assert any("features.csv row count mismatch" in note for note in notes)
    assert any("neighbors.csv row count is too small" in note for note in notes)
    assert any("review_template.csv row count is too small" in note for note in notes)
    assert any("sanity sample count mismatch" in note for note in notes)
    assert any("cpu_encoder_predictions.csv row count mismatch" in note for note in notes)
