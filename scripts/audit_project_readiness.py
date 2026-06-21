"""Audit project readiness by implementation stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.real import SOURCE_TYPE_PNG_GRAYSCALE_RAW, validate_manifest

PRE_REAL_REQUIRED_OUTPUTS = (
    "reference_features",
    "cpu_encoder_model",
    "cpu_encoder_metrics",
    "synthetic_unlabeled_predictions",
    "synthetic_png_batch_features",
    "synthetic_png_batch_sanity",
    "synthetic_png_batch_report",
)
PRE_REAL_FRESHNESS_INPUTS = (
    "scripts/run_pre_real_readiness.py",
    "scripts/generate_synthetic.py",
    "scripts/validate_synthetic.py",
    "scripts/extract_features.py",
    "scripts/build_segmentation_readiness.py",
    "scripts/train_embedding_smoke.py",
    "scripts/train_cpu_encoder_model.py",
    "scripts/score_unlabeled_cpu_encoder.py",
    "scripts/analyze_png_raw_folders.py",
    "scripts/extract_real_unlabeled_features.py",
)
SHAREABLE_SCAN_ROOTS = (
    "outputs/reports",
    "outputs/pre_real_readiness/reports",
)
SHAREABLE_SUFFIXES = {".csv", ".html", ".json", ".log", ".txt"}
SENSITIVE_OUTPUT_PATTERNS = (
    ("private manifest key", re.compile(r'"(?:png_path|arrays_npz|metadata_json)"\s*:')),
    ("workspace override flag", re.compile(r'"allow_workspace_input"\s*:\s*true')),
    ("absolute Windows path", re.compile(r"[A-Za-z]:(?:\\\\|/)")),
    ("absolute user path", re.compile(r"(?:/home/|/Users/)")),
)


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    title: str
    goal: str
    required_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    next_actions: tuple[str, ...]


STAGES: tuple[StageDefinition, ...] = (
    StageDefinition(
        stage_id="0_problem_schema_contract",
        title="문제와 스키마 기준",
        goal="Wafer map 의미, 보안 경계, 입력/출력 기준을 정의한다.",
        required_paths=(
            "docs/project_overview.md",
            "docs/data_schema.md",
            "docs/validation_protocol.md",
            "docs/roadmap.md",
            "src/wafermap/data/schema.py",
            "src/wafermap/real/manifest.py",
        ),
        evidence_paths=("tests/test_documentation_quality.py", "tests/test_real_manifest.py"),
        next_actions=("Schema/version 변경은 src/wafermap/real/manifest.py 중심으로 관리한다.",),
    ),
    StageDefinition(
        stage_id="1_synthetic_data",
        title="합성 데이터 생성기",
        goal="실제 데이터 전 검증을 위해 raw grayscale PNG와 같은 기준의 합성 FBM sample을 만든다.",
        required_paths=(
            "scripts/generate_synthetic.py",
            "scripts/validate_synthetic.py",
            "configs/synth/debug.json",
            "src/wafermap/synth/generator.py",
            "src/wafermap/viz/render.py",
        ),
        evidence_paths=("tests/test_synthetic_generator.py",),
        next_actions=("합성 데이터 결과는 방법 확인 근거로만 쓰고 실제 wafer 성능 근거로 주장하지 않는다.",),
    ),
    StageDefinition(
        stage_id="2_raw_png_ingestion",
        title="Raw PNG 읽기",
        goal="제품별 폴더의 raw PNG를 읽고 exact gray value와 chip geometry를 검증한다.",
        required_paths=(
            "scripts/analyze_png_raw_folders.py",
            "configs/eval/real_unlabeled_manifest_template_png.json",
            "src/wafermap/real/png_raw.py",
        ),
        evidence_paths=("tests/test_real_png_raw.py", "tests/test_real_unlabeled_workflow.py"),
        next_actions=("보안 폴더 하나를 실행해 geometry, gray value, stby 기본 검사 결과를 확인한다.",),
    ),
    StageDefinition(
        stage_id="3_observable_features",
        title="관측 가능한 Feature 추출",
        goal="Oracle 정보 없이 실제 wafer에서 계산 가능한 feature와 nearest-neighbor 리뷰 산출물을 만든다.",
        required_paths=(
            "scripts/extract_features.py",
            "scripts/extract_real_unlabeled_features.py",
            "src/wafermap/features/wafer_vector.py",
            "src/wafermap/features/selection.py",
            "src/wafermap/evaluation/nearest.py",
        ),
        evidence_paths=("tests/test_features.py", "tests/test_global_feature_contract.py"),
        next_actions=("실제 PNG 실행 후 기본 검사와 drift 요약을 synthetic reference feature와 비교한다.",),
    ),
    StageDefinition(
        stage_id="4_expert_review_loop",
        title="전문가 리뷰 흐름",
        goal="Feature와 neighbor 결과를 reviewer 판단, 실패 유형, 다음 작업 목록으로 연결한다.",
        required_paths=(
            "scripts/make_expert_review_template.py",
            "scripts/summarize_expert_review.py",
            "src/wafermap/reporting/expert_review.py",
            "docs/expert_review_protocol.md",
            "docs/real_wafer_review_checklist.md",
        ),
        evidence_paths=("tests/test_expert_review_protocol.py",),
        next_actions=("첫 실제 batch report에서 최소 20~50개의 neighbor 쌍 리뷰를 수집한다.",),
    ),
    StageDefinition(
        stage_id="5_cpu_ai_baseline",
        title="CPU AI 기준선",
        goal="실제 label 전에도 CPU에서 끝까지 실행 가능한 embedding/classification 기준선을 제공한다.",
        required_paths=(
            "scripts/train_embedding_smoke.py",
            "scripts/train_cpu_encoder_model.py",
            "scripts/score_unlabeled_cpu_encoder.py",
            "src/wafermap/training/embedding.py",
            "src/wafermap/training/cpu_encoder.py",
        ),
        evidence_paths=("tests/test_cpu_encoder_model.py", "tests/test_segmentation_training.py"),
        next_actions=("실제 리뷰 label이 생기기 전까지 CPU 확률은 리뷰 우선순위 참고값으로만 쓴다.",),
    ),
    StageDefinition(
        stage_id="6_pre_real_readiness",
        title="실제 데이터 전 준비",
        goal="합성 데이터 생성, 검증, feature 추출, CPU 학습, 라벨 없는 scoring을 한 번에 실행한다.",
        required_paths=("scripts/run_pre_real_readiness.py",),
        evidence_paths=("tests/test_pre_real_readiness.py",),
        next_actions=("큰 pipeline 변경 후와 첫 실제 batch 전에는 준비 실행을 다시 돌린다.",),
    ),
    StageDefinition(
        stage_id="7_real_go_live_preflight",
        title="실제 실행 전 최종 점검",
        goal="실제 데이터가 오기 전에 batch 명령, private manifest, PNG 읽기, 리포트, CPU scoring 경로를 확인한다.",
        required_paths=("scripts/analyze_png_raw_folders.py", "scripts/run_pre_real_readiness.py", "docs/real_unlabeled_workflow.md"),
        evidence_paths=("tests/test_pre_real_readiness.py",),
        next_actions=(
            "큰 pipeline 변경 후에는 outputs/pre_real_readiness를 최신 상태로 갱신한다.",
            "생성된 real_png_batch_command를 보안 환경 실제 데이터 실행 기준으로 쓴다.",
        ),
    ),
    StageDefinition(
        stage_id="8_real_data_actual_batch",
        title="실제 데이터 Batch",
        goal="보안 제품별 폴더의 실제 raw PNG를 실행하고 원본 노출 없이 파생 리포트를 확인한다.",
        required_paths=("scripts/analyze_png_raw_folders.py", "docs/real_unlabeled_workflow.md"),
        evidence_paths=(
            "outputs/private/real_png_batch_manifest.json",
            "outputs/reports/real_png_batch/features.csv",
            "outputs/reports/real_png_batch/sanity.json",
            "outputs/reports/real_png_batch/batch_metadata.json",
            "outputs/reports/real_png_batch/report.html",
            "outputs/reports/real_png_batch/neighbors.csv",
            "outputs/reports/real_png_batch/review_template.csv",
        ),
        next_actions=(
            "보안 제품별 폴더에 raw PNG를 넣고 scripts/analyze_png_raw_folders.py를 실행한다.",
            "Manifest와 실제 경로를 제외한 파생 리포트만 공유한다.",
        ),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/reports/project_readiness_audit.json")
    parser.add_argument("--html-out", default="outputs/reports/project_readiness_audit.html")
    parser.add_argument(
        "--pre-real-summary",
        default="outputs/pre_real_readiness/reports/pre_real_readiness_summary.json",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args(argv)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def path_record(path_text: str) -> dict[str, Any]:
    path = ROOT / path_text
    return {"path": path_text, "exists": path.exists(), "size_bytes": int(path.stat().st_size) if path.exists() else 0}


def audit_stage(stage: StageDefinition, pre_real_summary: Path) -> dict[str, Any]:
    required = [path_record(path) for path in stage.required_paths]
    evidence = [path_record(path) for path in stage.evidence_paths]
    missing_required = [item["path"] for item in required if not item["exists"]]
    missing_evidence = [item["path"] for item in evidence if not item["exists"]]
    notes: list[str] = []

    if stage.stage_id == "6_pre_real_readiness":
        readiness = read_pre_real_summary(pre_real_summary)
        notes.extend(readiness["notes"])
        if readiness["status"] != "PASS":
            missing_evidence.append(pre_real_evidence_gap(pre_real_summary, readiness))
    if stage.stage_id == "7_real_go_live_preflight":
        readiness = read_pre_real_summary(pre_real_summary)
        notes.extend(readiness["notes"])
        if readiness["status"] != "PASS":
            missing_evidence.append(pre_real_evidence_gap(pre_real_summary, readiness))
        leakage_findings = scan_shareable_outputs()
        if leakage_findings:
            notes.extend(leakage_findings[:10])
            if len(leakage_findings) > 10:
                notes.append(f"Additional shareable output leakage findings: {len(leakage_findings) - 10}")
            missing_evidence.append("shareable output leakage scan")

    if stage.stage_id == "8_real_data_actual_batch":
        notes.extend(inspect_real_go_live_outputs())

    if stage.stage_id == "8_real_data_actual_batch" and missing_evidence:
        status = "PENDING"
    elif missing_required:
        status = "PENDING"
    elif stage.stage_id == "8_real_data_actual_batch" and notes:
        status = "CHECK"
    elif missing_evidence:
        status = "CHECK"
    else:
        status = "PASS"

    return {
        "stage_id": stage.stage_id,
        "title": stage.title,
        "goal": stage.goal,
        "status": status,
        "required": required,
        "evidence": evidence,
        "missing_required": missing_required,
        "missing_evidence": missing_evidence,
        "notes": notes,
        "next_actions": list(stage.next_actions),
    }


def pre_real_evidence_gap(path: Path, readiness: dict[str, Any]) -> str:
    if readiness.get("status") == "MISSING":
        return relative(path)
    return f"{relative(path)} PASS 기준 미달"


def read_pre_real_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "notes": [f"Pre-real summary not found: {relative(path)}"]}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "notes": [f"Pre-real summary is not valid JSON: {exc}"]}
    status = str(payload.get("status", "UNKNOWN"))
    notes = [f"최신 실제 데이터 전 준비 상태: {status}"]
    stale_inputs = stale_pre_real_inputs(path)
    if stale_inputs:
        notes.append(f"실제 데이터 전 준비 요약이 실행 입력보다 오래됨: {', '.join(stale_inputs[:5])}")
        if len(stale_inputs) > 5:
            notes.append(f"추가로 오래된 실행 입력 수: {len(stale_inputs) - 5}")
        status = "CHECK"
    provenance_notes = inspect_pre_real_provenance(payload)
    if provenance_notes:
        notes.extend(provenance_notes)
        status = "CHECK"
    output_checks = payload.get("output_checks", [])
    if isinstance(output_checks, list):
        checks_by_name = {str(item.get("name")): item for item in output_checks if isinstance(item, dict)}
        missing_required_outputs = [
            name
            for name in PRE_REAL_REQUIRED_OUTPUTS
            if checks_by_name.get(name, {}).get("exists") is not True
        ]
        if missing_required_outputs:
            notes.append(f"Missing required pre-real outputs: {', '.join(missing_required_outputs)}")
            status = "CHECK"
        missing_outputs = [
            str(item.get("name", item.get("path", "unknown")))
            for item in output_checks
            if isinstance(item, dict) and item.get("exists") is not True
        ]
        if missing_outputs:
            notes.append(f"Missing pre-real outputs: {', '.join(missing_outputs)}")
            status = "CHECK"
        artifact_hash_notes = inspect_pre_real_artifact_hashes(output_checks)
        if artifact_hash_notes:
            notes.extend(artifact_hash_notes)
            status = "CHECK"
    outputs = payload.get("outputs", {})
    if isinstance(outputs, dict) and "cpu_encoder_metrics" in outputs:
        cpu_metrics_path = ROOT / str(outputs["cpu_encoder_metrics"])
        try:
            cpu_metrics = json.loads(cpu_metrics_path.read_text(encoding="utf-8"))
            cpu_gate = cpu_metrics.get("readiness_gate", {})
            if cpu_gate.get("status") != "PASS":
                notes.append(f"CPU encoder 준비 기준 상태: {cpu_gate.get('status', 'MISSING')}")
                status = "CHECK"
        except Exception as exc:  # noqa: BLE001 - audit reports the problem.
            notes.append(f"Could not inspect CPU encoder metrics: {exc}")
            status = "CHECK"
    return {"status": status, "notes": notes}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pre_real_provenance(payload: dict[str, Any]) -> list[str]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["Pre-real summary is missing provenance"]
    if provenance.get("schema_version") != "pre_real_readiness_provenance/v1":
        return ["Pre-real summary provenance schema is missing or unsupported"]

    notes = []
    config = provenance.get("config", {})
    if not isinstance(config, dict) or not config.get("sha256"):
        notes.append("Pre-real summary provenance is missing config sha256")
    pipeline_inputs = provenance.get("pipeline_inputs", [])
    if not isinstance(pipeline_inputs, list):
        return ["Pre-real summary provenance pipeline_inputs is invalid"]
    recorded = {
        str(item.get("path")): str(item.get("sha256"))
        for item in pipeline_inputs
        if isinstance(item, dict) and item.get("path") and item.get("sha256")
    }
    missing = [path for path in PRE_REAL_FRESHNESS_INPUTS if (ROOT / path).exists() and path not in recorded]
    if missing:
        notes.append(f"Pre-real provenance missing pipeline inputs: {', '.join(missing[:5])}")
    mismatched = []
    for raw_path, expected_sha in recorded.items():
        path = ROOT / raw_path
        if path.exists() and sha256_file(path) != expected_sha:
            mismatched.append(raw_path)
    if mismatched:
        notes.append(f"Pre-real provenance hash mismatch: {', '.join(mismatched[:5])}")
    git_info = provenance.get("git", {})
    if not isinstance(git_info, dict) or "commit" not in git_info or "dirty" not in git_info:
        notes.append("Pre-real provenance is missing git commit/dirty fields")
    return notes


def inspect_pre_real_artifact_hashes(output_checks: list[Any]) -> list[str]:
    notes = []
    for item in output_checks:
        if not isinstance(item, dict) or item.get("exists") is not True:
            continue
        name = str(item.get("name", item.get("path", "unknown")))
        if name not in PRE_REAL_REQUIRED_OUTPUTS:
            continue
        expected_sha = item.get("sha256")
        if not expected_sha:
            notes.append(f"Pre-real output is missing sha256: {name}")
            continue
        raw_path = item.get("path")
        if not raw_path:
            notes.append(f"Pre-real output is missing path for sha256 check: {name}")
            continue
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            notes.append(f"Pre-real output disappeared before sha256 check: {name}")
            continue
        if sha256_file(resolved) != str(expected_sha):
            notes.append(f"Pre-real output sha256 mismatch: {name}")
    return notes


def stale_pre_real_inputs(summary_path: Path) -> list[str]:
    try:
        summary_mtime = summary_path.stat().st_mtime
    except OSError:
        return []
    stale = []
    for raw_path in PRE_REAL_FRESHNESS_INPUTS:
        path = ROOT / raw_path
        if path.exists() and path.stat().st_mtime > summary_mtime + 1.0:
            stale.append(raw_path)
    return stale


def inspect_real_go_live_outputs() -> list[str]:
    notes: list[str] = []
    expected_sample_count: int | None = None
    cpu_model_scoring = False
    sanity_samples: list[Any] | None = None
    manifest_path = ROOT / "outputs" / "private" / "real_png_batch_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            validate_manifest(manifest)
            samples = manifest.get("samples", [])
            if any(sample.get("source_type") != SOURCE_TYPE_PNG_GRAYSCALE_RAW for sample in samples):
                notes.append("Real go-live manifest must contain only png_grayscale_raw samples.")
            if any(sample.get("allow_workspace_input") is True for sample in samples):
                notes.append("Real go-live manifest must not set allow_workspace_input=true.")
        except Exception as exc:  # noqa: BLE001 - audit should report contract issues, not crash.
            notes.append(f"Real go-live manifest 기준 검사 실패: {exc}")

    sanity_path = ROOT / "outputs" / "reports" / "real_png_batch" / "sanity.json"
    if sanity_path.exists():
        try:
            payload = json.loads(sanity_path.read_text(encoding="utf-8"))
            samples = payload.get("samples", payload if isinstance(payload, list) else [])
            sanity_samples = samples if isinstance(samples, list) else None
            error_count = sum(len(sample.get("errors", [])) for sample in samples if isinstance(sample, dict))
            if error_count:
                notes.append(f"Real go-live sanity has {error_count} errors; inspect sanity.json before sharing reports.")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Real go-live sanity check failed: {exc}")
    metadata_path = ROOT / "outputs" / "reports" / "real_png_batch" / "batch_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != "png_raw_batch_metadata/v1":
                notes.append("Real go-live batch metadata schema is missing or unsupported.")
            production_run = metadata.get("production_run") is True
            if not production_run:
                notes.append("Real go-live batch must be produced with --production-run.")
            if metadata.get("geometry_contract") != "explicit":
                notes.append("Real go-live batch must use explicit product geometry.")
            if metadata.get("manifest_location") != "outputs/private":
                notes.append("Real go-live batch manifest must be stored under outputs/private.")
            if metadata.get("reference_features") is not True:
                notes.append("Real go-live batch must include reference features for reviewer output.")
            expected_sample_count = int(metadata.get("png_sample_count", 0))
            cpu_model_scoring = metadata.get("cpu_model_scoring") is True
            if expected_sample_count <= 0:
                notes.append("Real go-live batch metadata has no PNG samples.")
                expected_sample_count = None
            if production_run:
                product_count = int(metadata.get("product_count", 0))
                explicit_count = int(metadata.get("explicit_geometry_product_count", 0))
                actual_net_die_count = int(metadata.get("actual_net_die_product_count", 0))
                if product_count <= 0:
                    notes.append("Real go-live batch metadata has no products.")
                elif explicit_count != product_count:
                    notes.append(
                        "Real go-live explicit geometry count mismatch: "
                        f"{explicit_count} of {product_count} products."
                    )
                elif actual_net_die_count != product_count:
                    notes.append(
                        "Real go-live positive actual_net_die count mismatch: "
                        f"{actual_net_die_count} of {product_count} products."
                    )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Real go-live batch metadata check failed: {exc}")
    if expected_sample_count is not None:
        report_dir = ROOT / "outputs" / "reports" / "real_png_batch"
        _append_exact_csv_count_issue(notes, report_dir / "features.csv", "features.csv", expected_sample_count)
        _append_min_csv_count_issue(notes, report_dir / "neighbors.csv", "neighbors.csv", expected_sample_count)
        _append_min_csv_count_issue(notes, report_dir / "review_template.csv", "review_template.csv", expected_sample_count)
        if sanity_samples is not None and len(sanity_samples) != expected_sample_count:
            notes.append(
                f"Real go-live sanity sample count mismatch: sanity.json has {len(sanity_samples)}, "
                f"batch metadata expects {expected_sample_count}."
            )
        if cpu_model_scoring:
            _append_exact_csv_count_issue(
                notes,
                report_dir / "cpu_encoder_predictions.csv",
                "cpu_encoder_predictions.csv",
                expected_sample_count,
            )
    return notes


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def _append_exact_csv_count_issue(notes: list[str], path: Path, label: str, expected: int) -> None:
    if not path.exists():
        notes.append(f"Real go-live {label} is missing.")
        return
    try:
        row_count = csv_row_count(path)
    except Exception as exc:  # noqa: BLE001 - audit reports the malformed artifact.
        notes.append(f"Real go-live {label} row count check failed: {exc}")
        return
    if row_count != expected:
        notes.append(f"Real go-live {label} row count mismatch: found {row_count}, expected {expected}.")


def _append_min_csv_count_issue(notes: list[str], path: Path, label: str, minimum: int) -> None:
    if not path.exists():
        notes.append(f"Real go-live {label} is missing.")
        return
    try:
        row_count = csv_row_count(path)
    except Exception as exc:  # noqa: BLE001 - audit reports the malformed artifact.
        notes.append(f"Real go-live {label} row count check failed: {exc}")
        return
    if row_count < minimum:
        notes.append(f"Real go-live {label} row count is too small: found {row_count}, expected at least {minimum}.")


def scan_shareable_outputs(roots: tuple[str, ...] = SHAREABLE_SCAN_ROOTS) -> list[str]:
    findings: list[str] = []
    for root_text in roots:
        root = ROOT / root_text
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SHAREABLE_SUFFIXES):
            if _is_under_private_output(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                findings.append(f"Could not scan shareable output {relative(path)}: {exc}")
                continue
            for label, pattern in SENSITIVE_OUTPUT_PATTERNS:
                if pattern.search(text):
                    findings.append(f"Shareable output contains {label}: {relative(path)}")
                    break
    return findings


def _is_under_private_output(path: Path) -> bool:
    try:
        path.resolve().relative_to((ROOT / "outputs" / "private").resolve())
        return True
    except ValueError:
        return False


def build_audit(pre_real_summary: Path) -> dict[str, Any]:
    stages = [audit_stage(stage, pre_real_summary) for stage in STAGES]
    counts = {status: sum(1 for stage in stages if stage["status"] == status) for status in ("PASS", "CHECK", "PENDING")}
    overall = "PASS"
    if counts["PENDING"] > 0:
        overall = "PENDING"
    elif counts["CHECK"] > 0:
        overall = "CHECK"
    return {
        "generated_at": utc_now_iso(),
        "overall_status": overall,
        "status_counts": counts,
        "stages": stages,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(stage['stage_id'])}</td>"
        f"<td>{html.escape(stage['title'])}</td>"
        f"<td>{html.escape(stage['status'])}</td>"
        f"<td>{html.escape(stage['goal'])}</td>"
        f"<td>{html.escape('; '.join(stage['missing_required']) or '-')}</td>"
        f"<td>{html.escape('; '.join(stage['missing_evidence']) or '-')}</td>"
        f"<td>{html.escape('; '.join(stage['next_actions']))}</td>"
        "</tr>"
        for stage in audit["stages"]
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WaferMap Project Readiness Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #dadce0; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ background: #f1f3f4; text-align: left; }}
    .status {{ font-weight: 700; }}
  </style>
</head>
<body>
  <h1>WaferMap Project Readiness Audit</h1>
  <p class="status">Overall: {html.escape(audit["overall_status"])}</p>
  <p>Generated at {html.escape(audit["generated_at"])}</p>
  <table>
    <thead>
      <tr>
        <th>단계</th><th>제목</th><th>상태</th><th>목표</th>
        <th>빠진 필수 항목</th><th>빠진 근거</th><th>다음 작업</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_audit(ROOT / args.pre_real_summary)
    write_json(ROOT / args.out, audit)
    write_html(ROOT / args.html_out, audit)
    print(f"overall_status={audit['overall_status']}")
    print(f"json={args.out}")
    print(f"html={args.html_out}")
    if args.fail_on_missing and audit["overall_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
