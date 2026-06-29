"""Run the full pre-real-data readiness pipeline on synthetic wafer maps."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.real import manifest_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synth/debug.json")
    parser.add_argument("--out-root", default="outputs/pre_real_readiness")
    parser.add_argument("--data-dir", help="Synthetic sample directory. Defaults to <out-root>/synthetic.")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--output-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-sample-count", type=int, default=6)
    return parser.parse_args(argv)


def relpath(target: Path, base_file: Path) -> str:
    return os.path.relpath(target.resolve(), base_file.resolve().parent).replace("\\", "/")


def repo_path(target: str | Path) -> str:
    path = Path(target)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(target)


def display_path(target: str | Path) -> str:
    path = Path(target)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path.resolve())


def display_command(command: list[str]) -> list[str]:
    displayed = []
    for idx, value in enumerate(command):
        if idx == 0 and Path(value).name.lower().startswith("python"):
            displayed.append("python")
        else:
            displayed.append(repo_path(value))
    return displayed


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_step(name: str, command: list[str], log_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_index = len(list(log_dir.glob("*.log"))) + 1
    log_path = log_dir / f"{log_index:02d}_{safe_name(name)}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    record = {
        "name": name,
        "command": display_command(command),
        "returncode": int(result.returncode),
        "elapsed_seconds": round(elapsed, 3),
        "log": repo_path(log_path),
    }
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}. See {log_path}")
    return record


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "step"


def reset_default_synthetic_dir(data_dir: Path, out_root: Path, explicit_data_dir: bool) -> None:
    if explicit_data_dir or not data_dir.exists():
        return
    resolved_data = data_dir.resolve()
    resolved_root = out_root.resolve()
    if resolved_data.name != "synthetic":
        raise RuntimeError(f"Refusing to clean non-default synthetic directory: {resolved_data}")
    try:
        resolved_data.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to clean synthetic directory outside out-root: {resolved_data}") from exc
    shutil.rmtree(resolved_data)


def write_score_manifest(data_dir: Path, path: Path, limit: int) -> None:
    sample_dirs = sorted(sample_dir for sample_dir in data_dir.glob("synth_*") if (sample_dir / "arrays.npz").exists())
    if not sample_dirs:
        raise RuntimeError(f"No synthetic samples available for unlabeled scoring manifest under {data_dir}")
    selected = sample_dirs[: max(1, min(limit, len(sample_dirs)))]
    manifest = manifest_payload(
        [
            {
                "sample_id": f"pre_real_score_{idx:06d}",
                "source_type": "synthetic_sample_dir",
                "sample_dir": str(sample_dir.resolve()),
            }
            for idx, sample_dir in enumerate(selected)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_png_geometry_json(data_dir: Path, path: Path) -> None:
    sample_dirs = sorted(sample_dir for sample_dir in data_dir.glob("synth_*") if (sample_dir / "raw_grayscale.png").exists())
    if not sample_dirs:
        raise RuntimeError(f"No synthetic raw_grayscale.png files available under {data_dir}")
    geometry_by_product: dict[str, dict[str, Any]] = {}
    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing synthetic metadata for PNG geometry smoke: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        geometry: dict[str, Any] = {
            "chip_blocks": metadata["chip_blocks"],
            "grid": metadata["grid"],
        }
        if "actual_net_die" in metadata:
            geometry["actual_net_die"] = metadata["actual_net_die"]
        geometry_by_product[sample_dir.name] = geometry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry_by_product, ensure_ascii=False, indent=2), encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def build_output_checks(outputs: dict[str, str]) -> list[dict[str, Any]]:
    checks = []
    for name, raw_path in outputs.items():
        path = Path(raw_path)
        resolved = path if path.is_absolute() else ROOT / path
        record: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "exists": resolved.exists(),
            "size_bytes": int(resolved.stat().st_size) if resolved.exists() else 0,
        }
        if resolved.exists() and resolved.suffix.lower() == ".csv":
            record["row_count"] = count_csv_rows(resolved)
        checks.append(record)
    return checks


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def readiness_issues(args: argparse.Namespace, outputs: dict[str, str], output_checks: list[dict[str, Any]]) -> list[str]:
    issues = []
    if not all(item["exists"] and item["size_bytes"] > 0 for item in output_checks):
        issues.append("one or more expected output artifacts are missing or empty")
    if int(args.count) < 20:
        issues.append(f"synthetic sample count is small for enterprise readiness: {args.count} < 20")
    cpu_metrics = read_json_if_exists(ROOT / outputs["cpu_encoder_metrics"])
    cpu_gate = cpu_metrics.get("readiness_gate", {})
    if cpu_gate.get("status") != "PASS":
        issue_text = "; ".join(str(item) for item in cpu_gate.get("issues", [])) or "CPU encoder 준비 기준이 PASS가 아님"
        issues.append(f"cpu_encoder readiness_gate={cpu_gate.get('status', 'MISSING')}: {issue_text}")
    return issues


def html_report(summary: dict[str, Any], out: Path) -> str:
    step_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(step['name'])}</td>"
        f"<td>{step['elapsed_seconds']:.2f}s</td>"
        f"<td><code>{html.escape(relpath(Path(step['log']), out))}</code></td>"
        "</tr>"
        for step in summary["steps"]
    )
    output_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td><code>{html.escape(relpath(Path(path), out))}</code></td>"
        "</tr>"
        for name, path in summary["outputs"].items()
    )
    check_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(check['name'])}</td>"
        f"<td>{'PASS' if check['exists'] and check['size_bytes'] > 0 else 'CHECK'}</td>"
        f"<td>{check['size_bytes']}</td>"
        f"<td>{check.get('row_count', '-')}</td>"
        "</tr>"
        for check in summary["output_checks"]
    )
    command = " ".join(summary["real_png_batch_command"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pre-Real WaferMap Readiness</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; line-height: 1.55; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #eef6ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Pre-Real WaferMap Readiness</h1>
  <div class="note">This report proves the synthetic-to-real pipeline can run end to end before real raw PNG data is available.</div>
  <h2>Status: {html.escape(summary['status'])}</h2>
  <ul>
    <li>Synthetic samples: {summary['synthetic_count']}</li>
    <li>Output root: <code>{html.escape(str(summary['out_root']))}</code></li>
  </ul>
  <h2>Steps</h2>
  <table>
    <tr><th>Step</th><th>Elapsed</th><th>Log</th></tr>
    {step_rows}
  </table>
  <h2>Key Outputs</h2>
  <table>
    <tr><th>Name</th><th>Path</th></tr>
    {output_rows}
  </table>
  <h2>Output Checks</h2>
  <table>
    <tr><th>Name</th><th>Status</th><th>Size Bytes</th><th>CSV Rows</th></tr>
    {check_rows}
  </table>
  <h2>Real PNG Batch Command</h2>
  <p><code>{html.escape(command)}</code></p>
</body>
</html>
"""


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now_iso()
    started = time.perf_counter()
    out_root = Path(args.out_root).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else out_root / "synthetic"
    reset_default_synthetic_dir(data_dir, out_root, explicit_data_dir=bool(args.data_dir))
    reports = out_root / "reports"
    figures = out_root / "figures"
    models = out_root / "models"
    manifests = out_root / "manifests"
    logs = out_root / "logs"
    for path in (reports, figures, models, manifests, logs):
        path.mkdir(parents=True, exist_ok=True)

    reference_features = reports / "synthetic_reference_features.csv"
    segmentation_manifest = reports / "segmentation_manifest.csv"
    model_path = models / "fbm_cpu_encoder_model.npz"
    score_manifest = manifests / "synthetic_unlabeled_score_manifest.json"
    score_predictions = reports / "synthetic_unlabeled_cpu_encoder_predictions.csv"
    score_neighbors = reports / "synthetic_unlabeled_cpu_encoder_neighbors.csv"
    score_sanity = reports / "synthetic_unlabeled_cpu_encoder_sanity.json"
    score_report = reports / "synthetic_unlabeled_cpu_encoder_report.html"
    png_geometry = reports / "synthetic_png_geometry.json"
    png_batch_dir = reports / "synthetic_png_batch"
    png_batch_manifest = manifests / "synthetic_png_batch_manifest.json"
    summary_json = reports / "pre_real_readiness_summary.json"
    summary_html = reports / "pre_real_readiness_report.html"
    legacy_report_score_manifest = reports / "synthetic_unlabeled_score_manifest.json"
    if legacy_report_score_manifest.exists():
        legacy_report_score_manifest.unlink()

    py = sys.executable
    steps = []
    steps.append(
        run_step(
            "generate synthetic raw grayscale",
            [
                py,
                str(ROOT / "scripts" / "generate_synthetic.py"),
                "--config",
                str(Path(args.config).resolve()),
                "--out",
                str(data_dir),
                "--count",
                str(args.count),
                "--no-preview",
            ],
            logs,
        )
    )
    steps.append(run_step("validate synthetic", [py, str(ROOT / "scripts" / "validate_synthetic.py"), "--data", str(data_dir)], logs))
    steps.append(
        run_step(
            "extract synthetic reference features",
            [
                py,
                str(ROOT / "scripts" / "extract_features.py"),
                "--data",
                str(data_dir),
                "--out",
                str(reference_features),
                "--include-validation-fields",
            ],
            logs,
        )
    )
    steps.append(
        run_step(
            "build segmentation readiness",
            [
                py,
                str(ROOT / "scripts" / "build_segmentation_readiness.py"),
                "--data",
                str(data_dir),
                "--out",
                str(reports / "segmentation_readiness_report.html"),
                "--metrics",
                str(reports / "segmentation_readiness_metrics.json"),
                "--manifest",
                str(segmentation_manifest),
                "--gallery",
                str(figures / "segmentation_readiness_gallery.png"),
            ],
            logs,
        )
    )
    steps.append(
        run_step(
            "train embedding smoke",
            [
                py,
                str(ROOT / "scripts" / "train_embedding_smoke.py"),
                "--manifest",
                str(segmentation_manifest),
                "--out",
                str(reports / "embedding_smoke_report.html"),
                "--metrics",
                str(reports / "embedding_smoke_metrics.json"),
                "--embeddings-out",
                str(reports / "embedding_smoke_embeddings.csv"),
                "--output-size",
                str(args.output_size),
                "--embedding-dim",
                str(args.embedding_dim),
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-val-samples",
                str(args.max_val_samples),
                "--top-k",
                str(args.top_k),
            ],
            logs,
        )
    )
    steps.append(
        run_step(
            "train cpu shared encoder",
            [
                py,
                str(ROOT / "scripts" / "train_cpu_encoder_model.py"),
                "--manifest",
                str(segmentation_manifest),
                "--model-out",
                str(model_path),
                "--out",
                str(reports / "cpu_encoder_report.html"),
                "--metrics",
                str(reports / "cpu_encoder_metrics.json"),
                "--predictions-out",
                str(reports / "cpu_encoder_val_predictions.csv"),
                "--output-size",
                str(args.output_size),
                "--hidden-dim",
                str(args.hidden_dim),
                "--embedding-dim",
                str(args.embedding_dim),
                "--epochs",
                str(args.epochs),
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-val-samples",
                str(args.max_val_samples),
                "--top-k",
                str(args.top_k),
            ],
            logs,
        )
    )
    write_score_manifest(data_dir, score_manifest, args.score_sample_count)
    steps.append(
        run_step(
            "score synthetic as unlabeled",
            [
                py,
                str(ROOT / "scripts" / "score_unlabeled_cpu_encoder.py"),
                "--model",
                str(model_path),
                "--manifest",
                str(score_manifest),
                "--predictions-out",
                str(score_predictions),
                "--neighbors-out",
                str(score_neighbors),
                "--sanity-out",
                str(score_sanity),
                "--report-out",
                str(score_report),
                "--top-k",
                str(args.top_k),
            ],
            logs,
        )
    )
    write_png_geometry_json(data_dir, png_geometry)
    steps.append(
        run_step(
            "analyze synthetic raw png batch",
            [
                py,
                str(ROOT / "scripts" / "analyze_png_raw_folders.py"),
                "--raw-root",
                str(data_dir),
                "--out-dir",
                str(png_batch_dir),
                "--manifest-out",
                str(png_batch_manifest),
                "--glob",
                "raw_grayscale.png",
                "--geometry-json",
                str(png_geometry),
                "--reference-features",
                str(reference_features),
                "--cpu-model",
                str(model_path),
                "--top-k",
                str(args.top_k),
            ],
            logs,
        )
    )

    real_png_command = [
        "python",
        "scripts/analyze_png_raw_folders.py",
        "--raw-root",
        "data/raw",
        "--geometry-json",
        "data/raw/product_geometry.json",
        "--out-dir",
        "outputs/reports/real_png_batch",
        "--manifest-out",
        "outputs/manifests/real_png_batch_manifest.json",
        "--reference-features",
        repo_path(reference_features),
        "--cpu-model",
        repo_path(model_path),
    ]
    outputs = {
        "reference_features": repo_path(reference_features),
        "segmentation_manifest": repo_path(segmentation_manifest),
        "embedding_smoke_report": repo_path(reports / "embedding_smoke_report.html"),
        "cpu_encoder_model": repo_path(model_path),
        "cpu_encoder_report": repo_path(reports / "cpu_encoder_report.html"),
        "cpu_encoder_metrics": repo_path(reports / "cpu_encoder_metrics.json"),
        "synthetic_unlabeled_predictions": repo_path(score_predictions),
        "synthetic_unlabeled_neighbors": repo_path(score_neighbors),
        "synthetic_unlabeled_report": repo_path(score_report),
        "synthetic_png_geometry": repo_path(png_geometry),
        "synthetic_png_batch_manifest": repo_path(png_batch_manifest),
        "synthetic_png_batch_features": repo_path(png_batch_dir / "features.csv"),
        "synthetic_png_batch_sanity": repo_path(png_batch_dir / "sanity.json"),
        "synthetic_png_batch_report": repo_path(png_batch_dir / "report.html"),
        "synthetic_png_batch_cpu_predictions": repo_path(png_batch_dir / "cpu_encoder_predictions.csv"),
    }
    output_checks = build_output_checks(outputs)
    gate_issues = readiness_issues(args, outputs, output_checks)
    output_status = "PASS" if not gate_issues else "CHECK"
    summary = {
        "status": output_status,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "synthetic_count": int(args.count),
        "out_root": repo_path(out_root),
        "steps": steps,
        "outputs": outputs,
        "output_checks": output_checks,
        "readiness_issues": gate_issues,
        "real_png_batch_command": real_png_command,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["outputs"]["summary_json"] = repo_path(summary_json)
    summary_html.write_text(html_report(summary, summary_html), encoding="utf-8")
    summary["outputs"]["summary_html"] = repo_path(summary_html)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_pipeline(args)
    print(f"Pre-real readiness: {summary['status']}")
    print(f"Summary JSON: {summary['outputs']['summary_json']}")
    print(f"Summary HTML: {summary['outputs']['summary_html']}")


if __name__ == "__main__":
    main()
