"""Create a compact FBM defect interpretation package.

This is the user-facing entry point. It hides the larger experiment pipeline
behind four outputs:

  defect_scores.csv
  wafer_interpretation_report.html
  sanity_summary.json
  similar_wafers.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.cli import positive_int
from wafermap.reporting import score_feature_rows
from wafermap.reporting.files import read_csv_rows, read_json_if_exists, write_csv_rows

os.environ.setdefault("MPLBACKEND", "Agg")
from wafermap.reporting.interpretation_report import html_report
from wafermap.reporting.overlays import (
    apply_overlay_location_labels,
    defect_overlay_masks,
    render_annotated_previews_from_samples,
)
from wafermap.reporting.previews import render_wafer_previews_from_samples

SIMILAR_COLUMNS = ("query_sample_id", "rank", "neighbor_sample_id", "distance")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Product-folder raw PNG root, e.g. data/raw or D:/fbm/raw_png.")
    source.add_argument("--manifest", help="real_unlabeled_manifest/v1 JSON path.")
    source.add_argument("--features-csv", help="Existing feature CSV to interpret without re-running extraction.")
    parser.add_argument("--sanity-json", help="Existing sanity JSON, required only with --features-csv when available.")
    parser.add_argument("--neighbors-csv", help="Existing nearest-neighbor CSV, optional with --features-csv.")
    parser.add_argument("--out", default="outputs/interpretation", help="Output directory for the compact package.")
    parser.add_argument("--geometry-json", help="Product geometry JSON for --input.")
    parser.add_argument("--reference-features", help="Optional reference feature CSV for similar wafer search.")
    parser.add_argument("--focus-sample", help="Sample ID to show as the one-wafer detailed review target.")
    parser.add_argument("--top-k", type=positive_int, default=5)
    return parser.parse_args(argv)


def output_dir(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_raw_png_pipeline(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    internal = out_dir / "_internal"
    manifest_path = (ROOT / "outputs" / "manifests" / f"{out_dir.name}_interpret_manifest.json").resolve()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_png_raw_folders.py"),
        "--raw-root",
        str(Path(args.input).resolve()),
        "--out-dir",
        str(internal),
        "--manifest-out",
        str(manifest_path),
        "--top-k",
        str(args.top_k),
    ]
    if args.geometry_json:
        command.extend(["--geometry-json", args.geometry_json])
    if args.reference_features:
        command.extend(["--reference-features", args.reference_features])
    subprocess.run(command, check=True)
    return (
        internal / "features.csv",
        internal / "sanity.json",
        _maybe_path(internal / "neighbors.csv"),
        internal / "batch_metadata.json",
        manifest_path,
    )


def run_manifest_pipeline(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    internal = out_dir / "_internal"
    internal.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "extract_real_unlabeled_features.py"),
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--features-out",
        str(internal / "features.csv"),
        "--sanity-out",
        str(internal / "sanity.json"),
        "--report-out",
        str(internal / "feature_report.html"),
        "--neighbors-out",
        str(internal / "neighbors.csv"),
        "--review-template-out",
        str(internal / "review_template.csv"),
        "--top-k",
        str(args.top_k),
    ]
    if args.reference_features:
        command.extend(["--reference-features", args.reference_features])
    subprocess.run(command, check=True)
    return internal / "features.csv", internal / "sanity.json", _maybe_path(internal / "neighbors.csv"), None, Path(args.manifest).resolve()


def existing_feature_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path | None, Path | None]:
    features = Path(args.features_csv).resolve()
    if not features.exists():
        raise ValueError(f"--features-csv does not exist: {features}")
    sanity = Path(args.sanity_json).resolve() if args.sanity_json else None
    neighbors = Path(args.neighbors_csv).resolve() if args.neighbors_csv else None
    return features, _maybe_path(sanity), _maybe_path(neighbors), None, None


def _maybe_path(path: Path | None) -> Path | None:
    if path is not None and path.exists() and path.stat().st_size > 0:
        return path
    return None


def build_sanity_summary(
    sanity_payload: dict[str, Any] | None,
    *,
    feature_rows: list[dict[str, str]],
    batch_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    samples = sanity_payload.get("samples", []) if sanity_payload else []
    sample_status = []
    pass_count = 0
    check_count = 0
    fail_count = 0
    if samples:
        for item in samples:
            errors = list(item.get("errors", []))
            warnings = list(item.get("warnings", []))
            if errors:
                status = "FAIL"
                fail_count += 1
            elif warnings:
                status = "CHECK"
                check_count += 1
            else:
                status = "PASS"
                pass_count += 1
            sample_status.append(
                {
                    "sample_id": item.get("sample_id", ""),
                    "status": status,
                    "errors": errors,
                    "warnings": warnings,
                }
            )
    else:
        pass_count = len(feature_rows)
        sample_status = [{"sample_id": row.get("sample_id", ""), "status": "PASS", "errors": [], "warnings": []} for row in feature_rows]
    overall = "FAIL" if fail_count else "CHECK" if check_count else "PASS"
    return {
        "schema_version": "fbm_interpretation_sanity/v1",
        "overall_status": overall,
        "sample_count": len(sample_status),
        "status_counts": {"PASS": pass_count, "CHECK": check_count, "FAIL": fail_count},
        "samples": sample_status,
        "batch_metadata": batch_metadata or {},
    }


def copy_similar_wafers(source: Path | None, out_path: Path) -> None:
    if source is None:
        write_csv_rows(out_path, [], SIMILAR_COLUMNS)
        return
    rows = read_csv_rows(source)
    if rows:
        write_csv_rows(out_path, rows)
    else:
        write_csv_rows(out_path, [], SIMILAR_COLUMNS)


def load_manifest_samples(manifest_path: Path | None) -> list[Any]:
    if manifest_path is None:
        return []
    module = _load_real_feature_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    module.validate_manifest(manifest)
    return [module.load_real_like_sample(entry, manifest_path) for entry in manifest.get("samples", [])]


def _load_real_feature_module() -> Any:
    path = ROOT / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features_for_interpret", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load real feature extraction module: {path}")
    spec.loader.exec_module(module)
    return module


def choose_focus_sample_id(feature_rows: list[dict[str, str]], requested: str | None) -> str:
    sample_ids = [str(row.get("sample_id", "")) for row in feature_rows]
    if requested:
        if requested not in sample_ids:
            raise ValueError(f"--focus-sample not found in feature rows: {requested}")
        return requested
    return sample_ids[0]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = output_dir(args.out)

    if args.input:
        features_path, source_sanity_path, source_neighbors_path, batch_metadata_path, manifest_path = run_raw_png_pipeline(args, out_dir)
    elif args.manifest:
        features_path, source_sanity_path, source_neighbors_path, batch_metadata_path, manifest_path = run_manifest_pipeline(args, out_dir)
    else:
        features_path, source_sanity_path, source_neighbors_path, batch_metadata_path, manifest_path = existing_feature_inputs(args)

    feature_rows = read_csv_rows(features_path)
    if not feature_rows:
        raise SystemExit(f"No feature rows found: {features_path}")
    defect_rows = score_feature_rows(feature_rows)
    focus_sample_id = choose_focus_sample_id(feature_rows, args.focus_sample)
    samples = load_manifest_samples(manifest_path)
    image_map = render_wafer_previews_from_samples(samples, out_dir)
    annotated_image_map = render_annotated_previews_from_samples(
        samples,
        defect_rows,
        out_dir,
        focus_sample_id=focus_sample_id,
    )
    apply_overlay_location_labels(defect_rows, annotated_image_map, samples)

    defect_scores_path = out_dir / "defect_scores.csv"
    report_path = out_dir / "wafer_interpretation_report.html"
    sanity_summary_path = out_dir / "sanity_summary.json"
    similar_path = out_dir / "similar_wafers.csv"

    write_csv_rows(defect_scores_path, defect_rows)
    sanity_payload = read_json_if_exists(source_sanity_path)
    batch_metadata = read_json_if_exists(batch_metadata_path)
    sanity_summary = build_sanity_summary(sanity_payload, feature_rows=feature_rows, batch_metadata=batch_metadata)
    sanity_summary_path.write_text(json.dumps(sanity_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_similar_wafers(source_neighbors_path, similar_path)
    report_path.write_text(
        html_report(
            feature_rows=feature_rows,
            defect_rows=defect_rows,
            sanity_summary=sanity_summary,
            image_map=image_map,
            annotated_image_map=annotated_image_map,
            focus_sample_id=focus_sample_id,
            defect_scores_path=defect_scores_path,
            sanity_path=sanity_summary_path,
            similar_path=similar_path,
            report_path=report_path,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(features_path, out_dir / "source_features.csv")
    print(f"Wrote defect scores: {defect_scores_path}")
    print(f"Wrote interpretation report: {report_path}")
    print(f"Wrote sanity summary: {sanity_summary_path}")
    print(f"Wrote similar wafer CSV: {similar_path}")


if __name__ == "__main__":
    main()
