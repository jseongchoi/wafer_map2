"""Extract observable features from real-like unlabeled FBM samples.

The input manifest must point to local files outside git-tracked real data.
Only derived feature/report outputs should be written by this script.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.data import PATTERN_CLASSES, SyntheticSample, load_metadata, load_sample
from wafermap.evaluation import cross_nearest_neighbor_indices
from wafermap.features import (
    compact_observable_feature_names,
    extract_feature_vector,
    feature_matrix,
    shared_observable_feature_names as shared_compact_feature_names,
)
from wafermap.reporting import build_template_rows, write_template_csv

LABEL_PREFIX = "label_"
OUTPUT_ROOT = ROOT / "outputs"
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON manifest listing real-like samples.")
    parser.add_argument("--features-out", default="outputs/reports/real_unlabeled_features.csv")
    parser.add_argument("--sanity-out", default="outputs/reports/real_unlabeled_sanity.json")
    parser.add_argument("--report-out", default="outputs/reports/real_unlabeled_report.html")
    parser.add_argument("--reference-features", help="Optional reference feature CSV for nearest-neighbor review.")
    parser.add_argument("--neighbors-out", default="outputs/reports/real_unlabeled_neighbors.csv")
    parser.add_argument("--review-template-out", default="outputs/reports/real_unlabeled_expert_review_template.csv")
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument(
        "--include-reference-labels",
        action="store_true",
        help="Copy reference label_* columns into the neighbor CSV. Use only with approved synthetic references.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_path(path: str | Path, manifest_path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "real_unlabeled_manifest/v1":
        raise ValueError("Manifest requires schema_version=real_unlabeled_manifest/v1")
    if manifest.get("feature_schema_version") != "observable_fbm_features/v1":
        raise ValueError("Manifest requires feature_schema_version=observable_fbm_features/v1")
    entries = manifest.get("samples", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain at least one sample entry")
    sample_ids: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"samples[{idx}] must be an object")
        if "source_type" not in entry:
            raise ValueError(f"samples[{idx}] requires source_type")
        source_type = str(entry["source_type"])
        sample_id = str(entry.get("sample_id", ""))
        if not sample_id or not SAMPLE_ID_RE.match(sample_id):
            raise ValueError(f"samples[{idx}] requires pseudonymized sample_id using only letters/numbers/._-")
        if sample_id in sample_ids:
            raise ValueError(f"samples[{idx}] duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if source_type == "npz_semantic_arrays":
            required = ("arrays_npz", "chip_blocks", "grid", "parser_name", "parser_version", "orientation")
            missing = [name for name in required if name not in entry]
            if missing:
                raise ValueError(f"samples[{idx}] missing required npz_semantic_arrays fields: {missing}")
            if entry.get("pseudonymized") is not True:
                raise ValueError(f"samples[{idx}] must set pseudonymized=true for real semantic arrays")
        elif source_type == "synthetic_sample_dir":
            if "sample_dir" not in entry:
                raise ValueError(f"samples[{idx}] missing sample_dir")
        else:
            raise ValueError(f"samples[{idx}] unsupported source_type: {source_type}")


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _ensure_real_input_outside_workspace(path: Path, entry: dict[str, Any], label: str) -> None:
    if _is_inside(path, ROOT) and entry.get("allow_workspace_input") is not True:
        raise ValueError(f"{label} must live outside the workspace for real inputs: {path}")


def ensure_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if not _is_inside(resolved, OUTPUT_ROOT):
        raise ValueError(f"Output path must be under {OUTPUT_ROOT}: {path}")
    return resolved


def load_real_like_sample(entry: dict[str, Any], manifest_path: Path) -> SyntheticSample:
    source_type = str(entry.get("source_type", "npz_semantic_arrays"))
    if source_type == "synthetic_sample_dir":
        sample_dir = resolve_manifest_path(entry["sample_dir"], manifest_path)
        sample = load_sample(sample_dir)
        if "sample_id" in entry:
            sample.sample_id = str(entry["sample_id"])
        return sample
    if source_type != "npz_semantic_arrays":
        raise ValueError(f"Unsupported source_type: {source_type}")

    arrays_path = resolve_manifest_path(entry["arrays_npz"], manifest_path)
    _ensure_real_input_outside_workspace(arrays_path, entry, "arrays_npz")
    if entry.get("metadata_json"):
        _ensure_real_input_outside_workspace(resolve_manifest_path(entry["metadata_json"], manifest_path), entry, "metadata_json")
    key_map = {
        "severity": "severity",
        "wafer_mask": "wafer_mask",
        "valid_test_mask": "valid_test_mask",
        "stby_mask": "stby_mask",
        "chip_index": "chip_index",
        **entry.get("array_keys", {}),
    }
    with np.load(arrays_path, allow_pickle=False) as arrays:
        severity = _load_severity_array(arrays, key_map["severity"])
        shape = severity.shape
        wafer_mask = _load_binary_array_or_default(
            arrays,
            key_map.get("wafer_mask"),
            "wafer_mask",
            shape,
            np.ones(shape, dtype=np.uint8),
            allow_missing=entry.get("allow_missing_masks_for_test") is True,
        )
        stby_mask = _load_binary_array_or_default(
            arrays,
            key_map.get("stby_mask"),
            "stby_mask",
            shape,
            np.zeros(shape, dtype=np.uint8),
            allow_missing=entry.get("allow_missing_masks_for_test") is True,
        )
        valid_default = ((wafer_mask > 0) & (stby_mask == 0)).astype(np.uint8)
        valid_test_mask = _load_binary_array_or_default(
            arrays,
            key_map.get("valid_test_mask"),
            "valid_test_mask",
            shape,
            valid_default,
            allow_missing=entry.get("allow_missing_masks_for_test") is True,
        )
        metadata = _metadata_from_entry(entry, manifest_path, shape)
        chip_index = _load_chip_index_or_default(
            arrays,
            key_map.get("chip_index"),
            shape,
            _infer_chip_index(metadata, shape, wafer_mask),
        )
    pattern_masks = np.zeros((len(PATTERN_CLASSES), *shape), dtype=np.uint8)
    pattern_intensity = np.zeros((len(PATTERN_CLASSES), *shape), dtype=np.float32)
    return SyntheticSample(
        sample_id=str(entry.get("sample_id", metadata.get("sample_id", arrays_path.stem))),
        severity=severity,
        wafer_mask=wafer_mask,
        valid_test_mask=valid_test_mask,
        stby_mask=stby_mask,
        pattern_masks=pattern_masks,
        pattern_intensity=pattern_intensity,
        chip_index=chip_index,
        metadata=metadata,
    )


def _array_or_default(arrays: Any, key: str | None, default: np.ndarray) -> np.ndarray:
    if key and key in arrays.files:
        return arrays[key]
    return default


def _require_array(arrays: Any, key: str, logical_name: str) -> np.ndarray:
    if key not in arrays.files:
        raise ValueError(f"Real semantic arrays require {logical_name} array key '{key}'")
    return arrays[key]


def _is_integer_like(values: np.ndarray) -> bool:
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        return True
    if not np.issubdtype(values.dtype, np.floating):
        return False
    return bool(np.all(np.isfinite(values)) and np.all(values == np.floor(values)))


def _load_severity_array(arrays: Any, key: str) -> np.ndarray:
    raw = _require_array(arrays, key, "severity")
    if raw.ndim != 2:
        raise ValueError("severity must be a 2D array")
    if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
        raise ValueError("severity contains NaN or inf before semantic casting")
    if not _is_integer_like(raw):
        raise ValueError("severity must contain integer grade values before semantic casting")
    raw_int = raw.astype(np.int16, copy=False)
    if raw_int.min(initial=0) < 0 or raw_int.max(initial=0) > 7:
        raise ValueError("severity must be in grade range 0..7 before semantic casting")
    return raw_int.astype(np.uint8)


def _load_binary_array_or_default(
    arrays: Any,
    key: str | None,
    logical_name: str,
    shape: tuple[int, int],
    default: np.ndarray,
    *,
    allow_missing: bool,
) -> np.ndarray:
    if not key or key not in arrays.files:
        if allow_missing:
            return default.astype(np.uint8)
        raise ValueError(f"Real semantic arrays require {logical_name} array key '{key}'")
    raw = arrays[key]
    if raw.shape != shape:
        raise ValueError(f"{logical_name} shape must match severity")
    if raw.ndim != 2:
        raise ValueError(f"{logical_name} must be a 2D array")
    if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
        raise ValueError(f"{logical_name} contains NaN or inf before semantic casting")
    if not _is_integer_like(raw):
        raise ValueError(f"{logical_name} must contain binary 0/1 values before semantic casting")
    raw_int = raw.astype(np.int16, copy=False)
    if raw_int.min(initial=0) < 0 or raw_int.max(initial=0) > 1:
        raise ValueError(f"{logical_name} must contain binary 0/1 values before semantic casting")
    return raw_int.astype(np.uint8)


def _load_chip_index_or_default(
    arrays: Any,
    key: str | None,
    shape: tuple[int, int],
    default: np.ndarray,
) -> np.ndarray:
    if not key or key not in arrays.files:
        return default.astype(np.int32)
    raw = arrays[key]
    if raw.shape != shape:
        raise ValueError("chip_index shape must match severity")
    if raw.ndim != 2:
        raise ValueError("chip_index must be a 2D array")
    if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
        raise ValueError("chip_index contains NaN or inf before semantic casting")
    if not _is_integer_like(raw):
        raise ValueError("chip_index must contain integer values before semantic casting")
    return raw.astype(np.int32)


def _metadata_from_entry(entry: dict[str, Any], manifest_path: Path, shape: tuple[int, int]) -> dict[str, Any]:
    metadata_path = entry.get("metadata_json")
    metadata: dict[str, Any] = {}
    if metadata_path:
        metadata = load_metadata(resolve_manifest_path(metadata_path, manifest_path))
    metadata.update(entry.get("metadata", {}))
    chip_blocks = entry.get("chip_blocks", metadata.get("chip_blocks", {}))
    grid = entry.get("grid", metadata.get("grid", {}))
    if not chip_blocks or not grid:
        raise ValueError("npz_semantic_arrays entries require chip_blocks and grid metadata")
    metadata.update(
        {
            "sample_id": str(entry.get("sample_id", metadata.get("sample_id", "real_unlabeled"))),
            "actual_net_die": int(entry.get("actual_net_die", metadata.get("actual_net_die", 0))),
            "chip_blocks": {"width": int(chip_blocks["width"]), "height": int(chip_blocks["height"])},
            "grid": {"rows": int(grid["rows"]), "cols": int(grid["cols"])},
            "image_shape": {"height": int(shape[0]), "width": int(shape[1])},
            "pattern_classes": list(PATTERN_CLASSES),
            "patterns": [],
        }
    )
    return metadata


def _infer_chip_index(metadata: dict[str, Any], shape: tuple[int, int], wafer_mask: np.ndarray) -> np.ndarray:
    chip_width = int(metadata["chip_blocks"]["width"])
    chip_height = int(metadata["chip_blocks"]["height"])
    rows = int(metadata["grid"]["rows"])
    cols = int(metadata["grid"]["cols"])
    chip_index = np.full(shape, -1, dtype=np.int32)
    chip_id = 0
    for row in range(rows):
        y0 = row * chip_height
        y1 = min(y0 + chip_height, shape[0])
        for col in range(cols):
            x0 = col * chip_width
            x1 = min(x0 + chip_width, shape[1])
            chip = wafer_mask[y0:y1, x0:x1] > 0
            block = chip_index[y0:y1, x0:x1]
            block[chip] = chip_id
            chip_index[y0:y1, x0:x1] = block
            chip_id += 1
    return chip_index


def validate_real_like_sample(sample: SyntheticSample) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    shape = sample.severity.shape
    arrays = {
        "wafer_mask": sample.wafer_mask,
        "valid_test_mask": sample.valid_test_mask,
        "stby_mask": sample.stby_mask,
        "chip_index": sample.chip_index,
    }
    for name, values in arrays.items():
        if values.shape != shape:
            errors.append(f"{name} shape does not match severity")
    if not np.isfinite(sample.severity.astype(np.float32)).all():
        errors.append("severity contains NaN or inf")
    if int(sample.severity.min()) < 0 or int(sample.severity.max()) > 7:
        errors.append("severity must be in grade range 0..7 after semantic parsing")
    for name in ("wafer_mask", "valid_test_mask", "stby_mask"):
        values = arrays[name]
        if values.size and (int(values.min()) < 0 or int(values.max()) > 1):
            errors.append(f"{name} must be binary")
    if (sample.severity[sample.wafer_mask == 0] != 0).any():
        errors.append("severity outside wafer_mask must be 0")
    if (sample.severity[sample.valid_test_mask == 0] != 0).any():
        errors.append("severity on invalid-test pixels must be 0")
    if (sample.stby_mask[sample.wafer_mask == 0] != 0).any():
        errors.append("stby pixels must be inside wafer_mask")
    if (sample.valid_test_mask[sample.wafer_mask == 0] != 0).any():
        errors.append("pixels outside wafer must be invalid")
    if (sample.valid_test_mask[sample.stby_mask > 0] != 0).any():
        errors.append("stby pixels must be invalid test pixels")
    if (sample.severity[sample.stby_mask > 0] != 0).any():
        errors.append("stby pixels must have severity grade 0; stby is unobserved, not Grade 7")
    if (sample.chip_index[sample.wafer_mask == 0] != -1).any():
        errors.append("chip_index outside wafer_mask must be -1")
    if (sample.chip_index[sample.wafer_mask > 0] < 0).any():
        errors.append("chip_index inside wafer_mask must be non-negative")
    chip_width = int(sample.metadata["chip_blocks"]["width"])
    chip_height = int(sample.metadata["chip_blocks"]["height"])
    rows = int(sample.metadata["grid"]["rows"])
    cols = int(sample.metadata["grid"]["cols"])
    expected_shape = (rows * chip_height, cols * chip_width)
    if expected_shape != shape:
        warnings.append(
            f"grid/chip_blocks imply shape {expected_shape}, but severity shape is {shape}; verify product geometry"
        )
    chip_ids = np.unique(sample.chip_index[sample.chip_index >= 0])
    actual_net_die = int(sample.metadata.get("actual_net_die", 0))
    if actual_net_die > 0 and len(chip_ids) != actual_net_die:
        errors.append(
            f"actual_net_die metadata ({actual_net_die}) does not match chip_index die count ({len(chip_ids)})"
        )
    chip_area = max(chip_width * chip_height, 1)
    stby_area = int((sample.stby_mask > 0).sum())
    if stby_area > 0 and stby_area % chip_area != 0:
        warnings.append("stby area is not an integer multiple of chip area; verify chip-level missing-test parsing")
    if not (sample.valid_test_mask > 0).any():
        errors.append("sample has no valid tested pixels")
    if not (sample.wafer_mask > 0).any():
        errors.append("sample has no wafer pixels")
    if not (sample.stby_mask > 0).any():
        warnings.append("sample has no stby pixels")
    if ((sample.severity == 0) & (sample.valid_test_mask > 0)).sum() == 0:
        warnings.append("sample has no measured Grade 0 pixels")
    return errors, warnings


def feature_row(sample: SyntheticSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "actual_net_die": sample.metadata.get("actual_net_die", 0),
        **extract_feature_vector(sample),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_feature_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def observable_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return compact_observable_feature_names(rows)


def shared_observable_feature_names(query_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> list[str]:
    return shared_compact_feature_names(query_rows, reference_rows)


def feature_drift_summary(
    query_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    top_n: int = 12,
) -> dict[str, Any]:
    feature_names = shared_observable_feature_names(query_rows, reference_rows)
    if not feature_names:
        return {"compared_feature_count": 0, "top_shifted_features": []}
    query_x = feature_matrix(query_rows, feature_names)
    ref_x = feature_matrix(reference_rows, feature_names)
    query_mean = query_x.mean(axis=0)
    ref_mean = ref_x.mean(axis=0)
    ref_std = np.maximum(ref_x.std(axis=0), 1e-6)
    z_delta = (query_mean - ref_mean) / ref_std
    order = np.argsort(np.abs(z_delta))[::-1][:top_n]
    return {
        "query_sample_count": len(query_rows),
        "reference_sample_count": len(reference_rows),
        "compared_feature_count": len(feature_names),
        "top_shifted_features": [
            {
                "feature": feature_names[int(idx)],
                "query_mean": float(query_mean[int(idx)]),
                "reference_mean": float(ref_mean[int(idx)]),
                "reference_std": float(ref_std[int(idx)]),
                "z_delta": float(z_delta[int(idx)]),
            }
            for idx in order
        ],
    }


def nearest_neighbor_rows(
    query_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, str]],
    top_k: int,
    include_reference_labels: bool,
) -> list[dict[str, Any]]:
    feature_names = shared_observable_feature_names(query_rows, reference_rows)
    if not feature_names:
        raise ValueError("No shared observable feature columns between query and reference")
    ref_x = feature_matrix(reference_rows, feature_names)
    query_x = feature_matrix(query_rows, feature_names)
    neighbors, distances = cross_nearest_neighbor_indices(query_x, ref_x, top_k)
    label_cols = [name for name in reference_rows[0] if name.startswith(LABEL_PREFIX)] if include_reference_labels else []
    rows: list[dict[str, Any]] = []
    for query_idx, query in enumerate(query_rows):
        for rank, ref_idx in enumerate(neighbors[query_idx], start=1):
            ref = reference_rows[int(ref_idx)]
            row: dict[str, Any] = {
                "query_sample_id": query["sample_id"],
                "rank": rank,
                "neighbor_sample_id": ref.get("sample_id", ""),
                "distance": float(distances[query_idx, ref_idx]),
            }
            for label in label_cols:
                row[label] = ref.get(label, "")
            rows.append(row)
    return rows


def write_sanity_json(path: Path, records: list[dict[str, Any]], feature_drift: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"samples": records}
    if feature_drift is not None:
        payload["feature_drift"] = feature_drift
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(Path(target).resolve(), base_file.resolve().parent)).as_posix()


def html_report(
    sanity_records: list[dict[str, Any]],
    features_out: Path,
    sanity_out: Path,
    neighbors_out: Path | None,
    report_out: Path,
    review_template_out: Path | None = None,
    feature_drift: dict[str, Any] | None = None,
) -> str:
    passed = sum(1 for item in sanity_records if not item["errors"])
    failed = len(sanity_records) - passed
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['sample_id'])}</td>"
        f"<td>{'PASS' if not item['errors'] else 'FAIL'}</td>"
        f"<td>{html.escape('; '.join(item['errors']) or '-')}</td>"
        f"<td>{html.escape('; '.join(item['warnings']) or '-')}</td>"
        f"<td>{item['valid_pixel_count']}</td>"
        f"<td>{item['stby_pixel_count']}</td>"
        "</tr>"
        for item in sanity_records
    )
    neighbor_link = (
        f'<li><a href="{html.escape(relpath(neighbors_out, report_out))}">Nearest-neighbor CSV</a></li>'
        if neighbors_out is not None
        else ""
    )
    review_template_link = (
        f'<li><a href="{html.escape(relpath(review_template_out, report_out))}">Expert review template CSV</a></li>'
        if review_template_out is not None
        else ""
    )
    drift_section = ""
    if feature_drift is not None:
        drift_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(item['feature'])}</td>"
            f"<td>{item['query_mean']:.4f}</td>"
            f"<td>{item['reference_mean']:.4f}</td>"
            f"<td>{item['reference_std']:.4f}</td>"
            f"<td>{item['z_delta']:.2f}</td>"
            "</tr>"
            for item in feature_drift.get("top_shifted_features", [])
        )
        drift_section = f"""
  <h2>Reference 대비 Feature Drift</h2>
  <p>Reference synthetic feature store와 query feature 평균의 차이를 z-score로 요약한다. 이 값은 real wafer가 synthetic reference 분포와 얼마나 다른지 보는 sanity check이며, 성능 metric은 아니다.</p>
  <table>
    <tr><th>Feature</th><th>Query Mean</th><th>Reference Mean</th><th>Reference Std</th><th>Z Delta</th></tr>
    {drift_rows}
  </table>
"""
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Real-Unlabeled FBM Feature Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Real-Unlabeled FBM Feature Report</h1>
  <div class="note">이 리포트는 보안 raw wafer를 저장하지 않고 observable feature와 sanity status만 남기는 workflow용이다. Stby는 Grade 7이 아니라 unobserved missing-test mask로 분리되어야 한다.</div>
  <div class="summary">
    <div class="card"><div>Samples</div><div class="metric">{len(sanity_records)}</div></div>
    <div class="card"><div>PASS</div><div class="metric">{passed}</div></div>
    <div class="card"><div>FAIL</div><div class="metric">{failed}</div></div>
  </div>
  <h2>Sanity Checks</h2>
  <table>
    <tr><th>Sample</th><th>Status</th><th>Errors</th><th>Warnings</th><th>Valid Pixels</th><th>Stby Pixels</th></tr>
    {rows}
  </table>
  {drift_section}
  <h2>Outputs</h2>
  <ul>
    <li><a href="{html.escape(relpath(features_out, report_out))}">Observable feature CSV</a></li>
    <li><a href="{html.escape(relpath(sanity_out, report_out))}">Sanity JSON</a></li>
    {neighbor_link}
    {review_template_link}
  </ul>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    entries = manifest.get("samples", [])
    features_out = ensure_output_path(Path(args.features_out))
    sanity_out = ensure_output_path(Path(args.sanity_out))
    report_out = ensure_output_path(Path(args.report_out))
    neighbors_out = ensure_output_path(Path(args.neighbors_out))
    review_template_out = ensure_output_path(Path(args.review_template_out))

    samples = [load_real_like_sample(entry, manifest_path) for entry in entries]
    sanity_records = []
    rows = []
    for sample in samples:
        errors, warnings = validate_real_like_sample(sample)
        sanity_records.append(
            {
                "sample_id": sample.sample_id,
                "errors": errors,
                "warnings": warnings,
                "shape": list(sample.shape),
                "valid_pixel_count": int((sample.valid_test_mask > 0).sum()),
                "stby_pixel_count": int((sample.stby_mask > 0).sum()),
                "chip_index_die_count": int(len(np.unique(sample.chip_index[sample.chip_index >= 0]))),
                "actual_net_die": int(sample.metadata.get("actual_net_die", 0)),
                "stby_chip_count_est": float(
                    (sample.stby_mask > 0).sum()
                    / max(
                        int(sample.metadata["chip_blocks"]["width"]) * int(sample.metadata["chip_blocks"]["height"]),
                        1,
                    )
                ),
                "grade_min": int(sample.severity.min()),
                "grade_max": int(sample.severity.max()),
            }
        )
        if not errors:
            rows.append(feature_row(sample))

    if not rows:
        write_sanity_json(sanity_out, sanity_records)
        raise SystemExit("No valid samples available for feature extraction")

    write_csv(features_out, rows)

    neighbors_path: Path | None = None
    review_template_path: Path | None = None
    drift_summary: dict[str, Any] | None = None
    if args.reference_features:
        reference_rows = read_feature_csv(Path(args.reference_features))
        if not reference_rows:
            raise SystemExit(f"No reference rows found in {args.reference_features}")
        drift_summary = feature_drift_summary(rows, reference_rows)
        neighbor_rows = nearest_neighbor_rows(rows, reference_rows, args.top_k, args.include_reference_labels)
        neighbors_path = neighbors_out
        write_csv(neighbors_path, neighbor_rows)
        review_template_rows, _ = build_template_rows(neighbor_rows)
        review_template_path = review_template_out
        write_template_csv(review_template_path, review_template_rows)

    write_sanity_json(sanity_out, sanity_records, drift_summary)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        html_report(sanity_records, features_out, sanity_out, neighbors_path, report_out, review_template_path, drift_summary),
        encoding="utf-8",
    )
    print(f"Wrote real-unlabeled features: {features_out}")
    print(f"Wrote sanity report: {sanity_out}")
    print(f"Wrote HTML report: {report_out}")
    if neighbors_path:
        print(f"Wrote nearest-neighbor CSV: {neighbors_path}")
    if review_template_path:
        print(f"Wrote expert review template CSV: {review_template_path}")


if __name__ == "__main__":
    main()
