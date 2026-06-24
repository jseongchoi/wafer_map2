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
import csv
import html
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

import numpy as np

from wafermap.reporting import score_feature_rows, top_defects

os.environ.setdefault("MPLBACKEND", "Agg")
from wafermap.viz import save_preview

OUTPUT_ROOT = ROOT / "outputs"
SIMILAR_COLUMNS = ("query_sample_id", "rank", "neighbor_sample_id", "distance")
OVERLAY_COLORS = {
    "edge": "#ff3b30",
    "ring": "#34c759",
    "scratch": "#af52de",
    "local": "#ffcc00",
    "shot_grid": "#00c7be",
    "stby_pattern": "#5ac8fa",
}
OVERLAY_ALPHAS = {
    "edge": 0.28,
    "ring": 0.42,
    "scratch": 0.42,
    "local": 0.58,
    "shot_grid": 0.42,
    "stby_pattern": 0.58,
}
OVERLAY_LABELS = {
    "local": "Local hotspot",
    "ring": "Ring / radial band",
    "edge": "Edge concentration",
    "scratch": "Scratch / line candidate",
    "shot_grid": "Shot-repeat candidate",
    "stby_pattern": "STBY / missing-test area",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Product-folder raw PNG root, e.g. D:/secure_fbm/raw_png.")
    source.add_argument("--manifest", help="real_unlabeled_manifest/v1 JSON path.")
    source.add_argument("--features-csv", help="Existing feature CSV to interpret without re-running extraction.")
    parser.add_argument("--sanity-json", help="Existing sanity JSON, required only with --features-csv when available.")
    parser.add_argument("--neighbors-csv", help="Existing nearest-neighbor CSV, optional with --features-csv.")
    parser.add_argument("--out", default="outputs/interpretation", help="Output directory for the compact package.")
    parser.add_argument("--geometry-json", help="Product geometry JSON for --input.")
    parser.add_argument("--reference-features", help="Optional reference feature CSV for similar wafer search.")
    parser.add_argument("--focus-sample", help="Sample ID to show as the one-wafer detailed review target.")
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--production-run", action="store_true", help="Pass production guardrails to raw PNG batch input.")
    parser.add_argument("--allow-workspace-input", action="store_true", help="Allow local workspace input for smoke tests.")
    parser.add_argument(
        "--allow-output-outside-root",
        action="store_true",
        help="Allow output outside outputs/. Use only for isolated smoke tests.",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def output_dir(path: str | Path, *, allow_outside_root: bool) -> Path:
    candidate = Path(path)
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not allow_outside_root and not _is_inside(resolved, OUTPUT_ROOT):
        raise ValueError(f"--out must be under {OUTPUT_ROOT} unless --allow-output-outside-root is set: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_raw_png_pipeline(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    if args.production_run:
        internal = ROOT / "outputs" / "reports" / f"{out_dir.name}_internal"
    else:
        internal = out_dir / "_internal"
    manifest_path = (ROOT / "outputs" / "private" / f"{out_dir.name}_interpret_manifest.json").resolve()
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
    if args.production_run:
        command.append("--production-run")
    if args.allow_workspace_input:
        command.append("--allow-workspace-input")
    if args.allow_output_outside_root and not args.production_run:
        command.append("--allow-output-outside-root")
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
    if args.allow_output_outside_root:
        command.append("--allow-output-outside-root")
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
        write_csv(out_path, [], SIMILAR_COLUMNS)
        return
    rows = read_csv(source)
    if rows:
        write_csv(out_path, rows)
    else:
        write_csv(out_path, [], SIMILAR_COLUMNS)


def render_wafer_previews(manifest_path: Path | None, out_dir: Path) -> dict[str, Path]:
    if manifest_path is None:
        return {}
    samples = load_manifest_samples(manifest_path)
    return render_wafer_previews_from_samples(samples, out_dir)


def load_manifest_samples(manifest_path: Path | None) -> list[Any]:
    if manifest_path is None:
        return []
    module = _load_real_feature_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    module.validate_manifest(manifest)
    return [module.load_real_like_sample(entry, manifest_path) for entry in manifest.get("samples", [])]


def render_wafer_previews_from_samples(samples: list[Any], out_dir: Path) -> dict[str, Path]:
    image_dir = out_dir / "wafer_images"
    image_map: dict[str, Path] = {}
    for sample in samples:
        out_path = image_dir / f"{_safe_filename(sample.sample_id)}.png"
        save_preview(out_path, sample.severity, sample.wafer_mask, sample.stby_mask, sample.sample_id)
        image_map[sample.sample_id] = out_path
    return image_map


def render_annotated_previews_from_samples(
    samples: list[Any],
    defect_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    focus_sample_id: str | None,
) -> dict[str, Path]:
    image_dir = out_dir / "annotated_images"
    rows_by_sample = _defect_rows_by_sample(defect_rows)
    image_map: dict[str, Path] = {}
    for sample in samples:
        if focus_sample_id and sample.sample_id != focus_sample_id:
            continue
        masks = defect_overlay_masks(sample, rows_by_sample.get(sample.sample_id, []))
        if not masks:
            continue
        out_path = image_dir / f"{_safe_filename(sample.sample_id)}.png"
        save_annotated_preview(out_path, sample, masks)
        image_map[sample.sample_id] = out_path
    return image_map


def defect_overlay_masks(sample: Any, defect_rows: list[dict[str, Any]], min_score: float = 15.0) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for row in defect_rows:
        family = str(row.get("defect_family", ""))
        if family not in OVERLAY_COLORS or float(row.get("score", 0.0)) < min_score:
            continue
        mask = _family_overlay_mask(sample, family)
        if mask.any():
            masks[family] = mask
    return masks


def save_annotated_preview(path: Path, sample: Any, masks: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    image = sample.severity.astype(np.float32).copy()
    image[sample.wafer_mask == 0] = np.nan
    cmap = plt.get_cmap("gray_r").copy()
    cmap.set_bad("#202624")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(image, cmap=cmap, vmin=0, vmax=7, interpolation="nearest")
    handles: list[Patch] = []
    for family in OVERLAY_COLORS:
        mask = masks.get(family)
        if mask is None or not mask.any():
            continue
        color = OVERLAY_COLORS[family]
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask] = to_rgba(color, OVERLAY_ALPHAS.get(family, 0.46))
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.2)
        handles.append(Patch(facecolor=color, edgecolor=color, label=OVERLAY_LABELS[family], alpha=0.72))

    ax.set_title(f"{sample.sample_id} - colored defect candidate regions")
    ax.axis("off")
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def apply_overlay_location_labels(
    defect_rows: list[dict[str, Any]],
    annotated_image_map: dict[str, Path],
    samples: list[Any],
) -> None:
    samples_by_id = {sample.sample_id: sample for sample in samples}
    rows_by_sample = _defect_rows_by_sample(defect_rows)
    overlay_families: dict[str, set[str]] = {}
    for sample_id in annotated_image_map:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        overlay_families[sample_id] = set(defect_overlay_masks(sample, rows_by_sample.get(sample_id, [])).keys())

    for row in defect_rows:
        sample_id = str(row.get("sample_id", ""))
        family = str(row.get("defect_family", ""))
        if family in overlay_families.get(sample_id, set()):
            row["location"] = "marked on annotated wafer image"


def _family_overlay_mask(sample: Any, family: str) -> np.ndarray:
    valid = sample.valid_test_mask > 0
    fail = valid & (sample.severity > 0)
    if family == "stby_pattern":
        return sample.stby_mask > 0
    if not fail.any():
        return np.zeros(sample.shape, dtype=bool)
    radius, theta = _pixel_radius_theta(sample.shape, valid)
    if family == "edge":
        return fail & (radius >= 0.76)
    if family == "ring":
        return _ring_mask(sample, valid, fail, radius)
    if family == "local":
        return _local_mask(sample)
    if family == "scratch":
        return _angular_peak_mask(sample, valid, fail, theta)
    if family == "shot_grid":
        return _hot_chip_mask(sample)
    return np.zeros(sample.shape, dtype=bool)


def _ring_mask(sample: Any, valid: np.ndarray, fail: np.ndarray, radius: np.ndarray, bins: int = 24) -> np.ndarray:
    severity = sample.severity.astype(np.float32) / 7.0
    means = np.zeros(bins, dtype=np.float32)
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        band = valid & (radius >= low) & (radius < high)
        means[idx] = float(severity[band].mean()) if band.any() else 0.0
    best = int(means.argmax())
    low = max(0.0, (best - 0.5) / bins)
    high = min(1.0, (best + 1.5) / bins)
    return fail & (radius >= low) & (radius < high)


def _local_mask(sample: Any) -> np.ndarray:
    chip_values, chip_valid = _chip_mean_grid(sample)
    if chip_values is None or chip_valid is None or not chip_valid.any():
        return _largest_component(_high_severity_mask(sample))
    values = chip_values[chip_valid]
    threshold = max(float(np.quantile(values, 0.98)), float(np.median(values) + 2.0 * values.std()))
    hot_chips = chip_valid & (chip_values >= threshold) & (chip_values > 0)
    hot_chips = _largest_component(hot_chips)
    if not hot_chips.any():
        return _largest_component(_high_severity_mask(sample))
    return _expand_chip_mask(sample, hot_chips)


def _angular_peak_mask(sample: Any, valid: np.ndarray, fail: np.ndarray, theta: np.ndarray, bins: int = 24) -> np.ndarray:
    severity = sample.severity.astype(np.float32) / 7.0
    scaled = theta / (2.0 * np.pi)
    means = np.zeros(bins, dtype=np.float32)
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        sector = valid & (scaled >= low) & (scaled < high)
        means[idx] = float(severity[sector].mean()) if sector.any() else 0.0
    best = int(means.argmax())
    low = max(0.0, (best - 0.7) / bins)
    high = min(1.0, (best + 1.7) / bins)
    sector = fail & (scaled >= low) & (scaled < high)
    focused = sector & _high_severity_mask(sample)
    return focused if focused.any() else sector


def _hot_chip_mask(sample: Any) -> np.ndarray:
    chip_values, chip_valid = _chip_mean_grid(sample)
    if chip_values is None or chip_valid is None or not chip_valid.any():
        return _high_severity_mask(sample)
    values = chip_values[chip_valid]
    threshold = max(float(np.quantile(values, 0.88)), float(values.mean() + values.std()))
    return _expand_chip_mask(sample, chip_valid & (chip_values >= threshold) & (chip_values > 0))


def _high_severity_mask(sample: Any) -> np.ndarray:
    valid = sample.valid_test_mask > 0
    severity = sample.severity.astype(np.float32) / 7.0
    values = severity[valid & (sample.severity > 0)]
    if len(values) == 0:
        return np.zeros(sample.shape, dtype=bool)
    threshold = max(float(np.quantile(values, 0.90)), float(values.mean() + values.std()))
    return valid & (severity >= threshold) & (sample.severity > 0)


def _pixel_radius_theta(shape: tuple[int, int], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(shape, dtype=np.float32)
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_distance = float(distance[valid].max()) if valid.any() else 1.0
    radius = distance / max(max_distance, 1.0)
    theta = (np.arctan2(x - cx, -(y - cy)) + 2.0 * np.pi) % (2.0 * np.pi)
    return radius.astype(np.float32), theta.astype(np.float32)


def _chip_mean_grid(sample: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        rows = int(sample.metadata["grid"]["rows"])
        cols = int(sample.metadata["grid"]["cols"])
        chip_width = int(sample.metadata["chip_blocks"]["width"])
        chip_height = int(sample.metadata["chip_blocks"]["height"])
    except (KeyError, TypeError, ValueError):
        return None, None

    severity = sample.severity.astype(np.float32) / 7.0
    valid = sample.valid_test_mask > 0
    values = np.zeros((rows, cols), dtype=np.float32)
    chip_valid = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        y0 = row * chip_height
        y1 = min(y0 + chip_height, sample.shape[0])
        for col in range(cols):
            x0 = col * chip_width
            x1 = min(x0 + chip_width, sample.shape[1])
            block_valid = valid[y0:y1, x0:x1]
            if block_valid.any():
                values[row, col] = float(severity[y0:y1, x0:x1][block_valid].mean())
                chip_valid[row, col] = True
    return values, chip_valid


def _expand_chip_mask(sample: Any, chip_mask: np.ndarray) -> np.ndarray:
    try:
        chip_width = int(sample.metadata["chip_blocks"]["width"])
        chip_height = int(sample.metadata["chip_blocks"]["height"])
    except (KeyError, TypeError, ValueError):
        return np.zeros(sample.shape, dtype=bool)
    mask = np.zeros(sample.shape, dtype=bool)
    rows, cols = chip_mask.shape
    for row in range(rows):
        y0 = row * chip_height
        y1 = min(y0 + chip_height, sample.shape[0])
        for col in range(cols):
            if not chip_mask[row, col]:
                continue
            x0 = col * chip_width
            x1 = min(x0 + chip_width, sample.shape[1])
            mask[y0:y1, x0:x1] = sample.valid_test_mask[y0:y1, x0:x1] > 0
    return mask & (sample.severity > 0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(best):
            best = component
    out = np.zeros(mask.shape, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def _load_real_feature_module() -> Any:
    path = ROOT / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features_for_interpret", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load real feature extraction module: {path}")
    spec.loader.exec_module(module)
    return module


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe or "wafer"


def _legacy_multi_wafer_html_report(
    *,
    feature_rows: list[dict[str, str]],
    defect_rows: list[dict[str, Any]],
    sanity_summary: dict[str, Any],
    image_map: dict[str, Path],
    defect_scores_path: Path,
    sanity_path: Path,
    similar_path: Path,
    report_path: Path,
) -> str:
    top = top_defects(defect_rows)
    sample_blocks = "\n".join(
        _sample_block(row, top.get(str(row.get("sample_id", "")), []), image_map, report_path)
        for row in feature_rows
    )
    score_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['sample_id']))}</td>"
        f"<td>{html.escape(str(row['defect_family']))}</td>"
        f"<td><strong>{float(row['score']):.1f}</strong></td>"
        f"<td>{html.escape(str(row['confidence']))}</td>"
        f"<td>{html.escape(str(row['location']))}</td>"
        f"<td>{html.escape(str(row['evidence']))}</td>"
        "</tr>"
        for row in defect_rows
    )
    status = str(sanity_summary["overall_status"])
    status_class = {"PASS": "pass", "CHECK": "check", "FAIL": "fail"}.get(status, "check")
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FBM Wafer Interpretation Report</title>
  <style>
    body {{ margin: 0; background: #f5f7f8; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; line-height: 1.58; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }}
    a {{ color: #2567a8; text-underline-offset: 3px; }}
    .hero, .section, .sample {{ border: 1px solid #d8e0de; border-radius: 8px; background: #fff; box-shadow: 0 14px 34px rgba(23, 33, 31, 0.08); }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .section {{ padding: 20px; margin: 18px 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .card {{ border: 1px solid #d8e0de; border-radius: 8px; padding: 14px; background: #fbfcfc; }}
    .metric {{ display: block; font-size: 26px; font-weight: 800; }}
    .muted {{ color: #61716d; }}
    .badge {{ display: inline-flex; border-radius: 999px; padding: 5px 10px; font-weight: 800; font-size: 13px; }}
    .pass {{ background: #e3f2eb; color: #1f6b54; }}
    .check {{ background: #f4ead6; color: #805611; }}
    .fail {{ background: #f7dfdd; color: #9a3d38; }}
    .sample {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 16px; padding: 16px; margin: 12px 0; box-shadow: none; }}
    .wafer-img {{ width: 100%; border: 1px solid #d8e0de; border-radius: 8px; background: #17211f; }}
    .no-image {{ display: grid; min-height: 260px; place-items: center; border: 1px dashed #b8c5c1; border-radius: 8px; color: #61716d; background: #fbfcfc; text-align: center; padding: 12px; }}
    .defects {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .guide-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .bar {{ height: 9px; border-radius: 999px; background: #e8eeec; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: #25745d; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #d8e0de; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f1; }}
    code {{ background: #eef3f2; border-radius: 5px; padding: 2px 5px; }}
    @media (max-width: 900px) {{ .sample {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 760px) {{ .summary, .defects, .guide-grid {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>FBM Wafer Interpretation Report</h1>
    <p class="muted">Fail Bit Map에서 defect family별 점수와 근거를 먼저 보여주는 사용자용 리포트입니다. Similar wafer 검색은 보조 산출물입니다.</p>
    <span class="badge {status_class}">Sanity {html.escape(status)}</span>
    <p class="muted"><strong>Sanity PASS</strong>는 입력 PNG/manifest, mask, stby, geometry가 분석 가능한 형태라는 뜻입니다. defect score가 맞다는 성능 보증은 아니며, 실제 판단은 wafer 이미지와 score를 함께 보고 보정해야 합니다.</p>
    <div class="summary">
      <div class="card"><span class="metric">{len(feature_rows)}</span><span class="muted">interpreted wafers</span></div>
      <div class="card"><span class="metric">{sanity_summary['status_counts']['PASS']}</span><span class="muted">sanity PASS</span></div>
      <div class="card"><span class="metric">{sanity_summary['status_counts']['CHECK']}</span><span class="muted">sanity CHECK</span></div>
      <div class="card"><span class="metric">{sanity_summary['status_counts']['FAIL']}</span><span class="muted">sanity FAIL</span></div>
    </div>
  </section>

  <section class="section">
    <h2>이 리포트 읽는 순서</h2>
    <div class="guide-grid">
      <div class="card">
        <h3>1. Wafer 이미지를 먼저 봅니다</h3>
        <p>눈으로 보이는 edge, ring, local hotspot, scratch, shot 반복, stby 영역이 있는지 먼저 판단합니다. 점수보다 이미지 판단이 우선입니다.</p>
      </div>
      <div class="card">
        <h3>2. 상위 defect score 3개를 봅니다</h3>
        <p>각 wafer 카드에는 15점 이상인 상위 후보만 보여줍니다. 높게 나온 family가 이미지와 맞는지 확인합니다.</p>
      </div>
      <div class="card">
        <h3>3. Location과 Evidence를 봅니다</h3>
        <p>Location은 대략적인 위치 설명이고, Evidence는 점수를 만든 feature 값입니다. 위치가 틀리면 반드시 피드백해야 합니다.</p>
      </div>
      <div class="card">
        <h3>4. Similar wafer는 보조로 봅니다</h3>
        <p>유사맵 검색은 최종 판정이 아니라 과거 사례 참고입니다. 본질은 wafer 이미지와 defect score가 맞는지입니다.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>점수 해석 기준</h2>
    <table>
      <tr><th>Score</th><th>의미</th><th>리뷰할 때 판단</th></tr>
      <tr><td><strong>70~100</strong></td><td>강한 후보</td><td>이미지에서도 이 defect가 주된 패턴으로 보여야 합니다. 아니면 false positive입니다.</td></tr>
      <tr><td><strong>40~69</strong></td><td>중간 후보</td><td>일부 근거는 있지만 다른 defect와 섞였거나 위치/강도 보정이 필요할 수 있습니다.</td></tr>
      <tr><td><strong>15~39</strong></td><td>약한 후보</td><td>참고용입니다. 사람이 보기에도 약하게 보이는지, 아니면 놓친 defect인지 확인합니다.</td></tr>
      <tr><td><strong>0~14</strong></td><td>거의 없음</td><td>이미지에서 분명히 보이는데 낮게 나오면 false negative입니다.</td></tr>
    </table>
  </section>

  <section class="section">
    <h2>당신이 줘야 하는 피드백</h2>
    <p class="muted">각 wafer마다 아래 5가지만 알려주면 다음 calibration을 바로 할 수 있습니다.</p>
    <table>
      <tr><th>볼 항목</th><th>피드백 예시</th><th>이 피드백으로 고치는 것</th></tr>
      <tr><td>주 defect family가 맞는가</td><td><code>real_like_synth_000000: local은 맞고, ring은 과대평가</code></td><td>family별 score weight와 threshold</td></tr>
      <tr><td>빠진 defect가 있는가</td><td><code>scratch가 눈에 보이는데 score가 너무 낮음</code></td><td>scratch/local/ring feature 추가 또는 segmentation 후보</td></tr>
      <tr><td>점수가 너무 높거나 낮은가</td><td><code>edge 89는 너무 높음, 실제로는 40 정도</code></td><td>score calibration</td></tr>
      <tr><td>위치 설명이 맞는가</td><td><code>local 위치가 12시라고 나오는데 실제는 6시</code></td><td>clock/position feature 해석</td></tr>
      <tr><td>입력 해석 문제가 있는가</td><td><code>stby를 defect처럼 보고 있음</code>, <code>wafer 밖 영역이 이상함</code></td><td>parser, mask, stby 분리</td></tr>
    </table>
    <p class="muted">가장 좋은 피드백 형식은 <code>sample_id / 실제 주 defect / 틀린 점수 / 빠진 defect / 위치 오류 / 코멘트</code>입니다.</p>
  </section>

  <section class="section">
    <h2>Wafer별 해석 요약</h2>
    {sample_blocks}
  </section>

  <section class="section">
    <h2>전체 Defect Score Table</h2>
    <table>
      <tr><th>Sample</th><th>Defect Family</th><th>Score</th><th>Confidence</th><th>Location</th><th>Evidence</th></tr>
      {score_rows}
    </table>
  </section>

  <section class="section">
    <h2>사용자가 볼 파일</h2>
    <ul>
      <li><a href="{html.escape(_relpath(defect_scores_path, report_path))}">defect_scores.csv</a></li>
      <li><a href="{html.escape(_relpath(sanity_path, report_path))}">sanity_summary.json</a></li>
      <li><a href="{html.escape(_relpath(similar_path, report_path))}">similar_wafers.csv</a></li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def _sample_block(
    feature_row: dict[str, str],
    defects: list[dict[str, Any]],
    image_map: dict[str, Path],
    report_path: Path,
) -> str:
    sample_id = str(feature_row.get("sample_id", "unknown"))
    image_path = image_map.get(sample_id)
    if image_path is not None:
        image_html = (
            f'<img class="wafer-img" src="{html.escape(_relpath(image_path, report_path))}" '
            f'alt="Wafer map preview for {html.escape(sample_id)}">'
        )
    else:
        image_html = '<div class="no-image">No wafer preview available<br><span class="muted">manifest 또는 raw PNG 입력으로 실행하면 이미지가 생성됩니다.</span></div>'
    if defects:
        cards = "\n".join(
            f"""<div class="card">
  <h3>{html.escape(str(item["defect_family"]))} · {float(item["score"]):.1f}</h3>
  <div class="bar"><span style="width: {max(0.0, min(100.0, float(item["score"]))):.1f}%"></span></div>
  <p class="muted">{html.escape(str(item["location"]))}</p>
  <p>{html.escape(str(item["evidence"]))}</p>
</div>"""
            for item in defects
        )
    else:
        cards = '<div class="card"><h3>No dominant defect</h3><p class="muted">15점 이상 defect family가 없습니다.</p></div>'
    return f"""<article class="sample">
  <div>
    {image_html}
  </div>
  <div>
  <h3>{html.escape(sample_id)}</h3>
  <div class="defects">{cards}</div>
  </div>
</article>"""


def html_report(
    *,
    feature_rows: list[dict[str, str]],
    defect_rows: list[dict[str, Any]],
    sanity_summary: dict[str, Any],
    image_map: dict[str, Path],
    annotated_image_map: dict[str, Path],
    focus_sample_id: str,
    defect_scores_path: Path,
    sanity_path: Path,
    similar_path: Path,
    report_path: Path,
) -> str:
    focus_feature = next(row for row in feature_rows if str(row.get("sample_id", "")) == focus_sample_id)
    focus_rows = _defect_rows_by_sample(defect_rows).get(focus_sample_id, [])
    top_rows = [row for row in focus_rows if float(row.get("score", 0.0)) >= 15.0][:5]
    if not top_rows:
        top_rows = focus_rows[:3]
    status = str(sanity_summary["overall_status"])
    status_class = {"PASS": "pass", "CHECK": "check", "FAIL": "fail"}.get(status, "check")
    score_rows = "\n".join(_score_table_row(row) for row in focus_rows)
    focus_block = _focus_sample_block(
        focus_feature,
        top_rows,
        image_map,
        annotated_image_map,
        report_path,
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FBM One-Wafer Defect Review</title>
  <style>
    body {{ margin: 0; background: #f5f7f8; color: #17211f; font-family: "Segoe UI", "Noto Sans KR", Arial, sans-serif; line-height: 1.58; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }}
    a {{ color: #2567a8; text-underline-offset: 3px; }}
    .hero, .section, .focus {{ border: 1px solid #d8e0de; border-radius: 8px; background: #fff; box-shadow: 0 14px 34px rgba(23, 33, 31, 0.08); }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .section {{ padding: 20px; margin: 18px 0; }}
    .focus {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr); gap: 18px; padding: 16px; box-shadow: none; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .guide-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-items: start; }}
    .card {{ border: 1px solid #d8e0de; border-radius: 8px; padding: 14px; background: #fbfcfc; }}
    .metric {{ display: block; font-size: 26px; font-weight: 800; }}
    .muted {{ color: #61716d; }}
    .badge {{ display: inline-flex; border-radius: 999px; padding: 5px 10px; font-weight: 800; font-size: 13px; }}
    .pass {{ background: #e3f2eb; color: #1f6b54; }}
    .check {{ background: #f4ead6; color: #805611; }}
    .fail {{ background: #f7dfdd; color: #9a3d38; }}
    .wafer-img {{ width: 100%; border: 1px solid #d8e0de; border-radius: 8px; background: #17211f; display: block; }}
    .no-image {{ display: grid; min-height: 260px; place-items: center; border: 1px dashed #b8c5c1; border-radius: 8px; color: #61716d; background: #fbfcfc; text-align: center; padding: 12px; }}
    .defects {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .bar {{ height: 9px; border-radius: 999px; background: #e8eeec; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: #25745d; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid #d8e0de; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f1; }}
    code {{ background: #eef3f2; border-radius: 5px; padding: 2px 5px; }}
    @media (max-width: 980px) {{ .focus, .image-grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 760px) {{ .summary, .guide-grid {{ grid-template-columns: 1fr; }} h1 {{ font-size: 28px; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>FBM One-Wafer Defect Review</h1>
    <p class="muted">이 리포트는 여러 웨이퍼를 한꺼번에 해석하려는 보고서가 아닙니다. 지금은 <strong>웨이퍼 1장</strong>에서 불량 후보 family를 색으로 표시하고, 그 표시가 맞는지 검증하는 용도입니다.</p>
    <span class="badge {status_class}">Sanity {html.escape(status)}</span>
    <div class="summary">
      <div class="card"><span class="metric">{html.escape(focus_sample_id)}</span><span class="muted">focus wafer</span></div>
      <div class="card"><span class="metric">{len(feature_rows)}</span><span class="muted">CSV에 포함된 전체 wafer</span></div>
      <div class="card"><span class="metric">{sanity_summary['status_counts']['PASS']}</span><span class="muted">sanity PASS</span></div>
      <div class="card"><span class="metric">{len(top_rows)}</span><span class="muted">표시한 후보 defect</span></div>
    </div>
  </section>

  <section class="section">
    <h2>먼저 볼 것</h2>
    <div class="guide-grid">
      <div class="card">
        <h3>1. 색칠된 이미지</h3>
        <p>시계 방향 설명보다 이 이미지가 우선입니다. 색이 칠해진 영역이 실제 불량 위치와 맞는지 판단하면 됩니다.</p>
      </div>
      <div class="card">
        <h3>2. Defect 후보 점수</h3>
        <p>점수는 “그 defect family처럼 보이는 정도”입니다. 70 이상은 강한 후보, 40~69는 중간 후보, 15~39는 약한 후보입니다.</p>
      </div>
      <div class="card">
        <h3>3. 색이 틀린 부분</h3>
        <p>가장 중요한 피드백은 “어떤 색이 잘못 칠해졌는지”와 “어떤 불량이 빠졌는지”입니다.</p>
      </div>
      <div class="card">
        <h3>4. 다음 단계</h3>
        <p>이 한 장에서 색칠이 납득되면 같은 방식으로 실제 wafer 묶음에 확장하고, 이후 embedding 기반 유사맵 검색으로 연결합니다.</p>
      </div>
    </div>
  </section>

  <section class="section">
    <h2>Focus Wafer</h2>
    {focus_block}
  </section>

  <section class="section">
    <h2>당신이 주면 되는 피드백</h2>
    <table>
      <tr><th>확인 항목</th><th>피드백 예시</th><th>이 피드백으로 고치는 것</th></tr>
      <tr><td>색칠 영역이 맞는가</td><td><code>노란색 local은 맞고, 초록 ring은 너무 넓다</code></td><td>family별 mask 후보 생성 방식</td></tr>
      <tr><td>빠진 불량이 있는가</td><td><code>보라색 scratch가 있어야 하는데 안 보인다</code></td><td>scratch/line feature와 threshold</td></tr>
      <tr><td>점수가 납득되는가</td><td><code>edge 89는 너무 높고 40 정도가 맞다</code></td><td>score calibration</td></tr>
      <tr><td>STBY를 defect처럼 보는가</td><td><code>파란색은 실제 불량이 아니라 미측정 영역이다</code></td><td>stby 분리와 defect score 제외 규칙</td></tr>
    </table>
  </section>

  <section class="section">
    <h2>Focus Wafer Score Table</h2>
    <table>
      <tr><th>Defect Family</th><th>Score</th><th>Confidence</th><th>Image Mark</th><th>Evidence</th></tr>
      {score_rows}
    </table>
  </section>

  <section class="section">
    <h2>생성 파일</h2>
    <ul>
      <li><a href="{html.escape(_relpath(defect_scores_path, report_path))}">defect_scores.csv</a>: 전체 wafer의 defect family 점수</li>
      <li><a href="{html.escape(_relpath(sanity_path, report_path))}">sanity_summary.json</a>: 입력 파싱과 mask 상태 점검</li>
      <li><a href="{html.escape(_relpath(similar_path, report_path))}">similar_wafers.csv</a>: reference feature가 있을 때 유사 wafer 후보</li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def _focus_sample_block(
    feature_row: dict[str, str],
    defects: list[dict[str, Any]],
    image_map: dict[str, Path],
    annotated_image_map: dict[str, Path],
    report_path: Path,
) -> str:
    sample_id = str(feature_row.get("sample_id", "unknown"))
    annotated_html = _image_or_placeholder(
        annotated_image_map.get(sample_id),
        report_path,
        f"Annotated defect map for {sample_id}",
        "No annotated defect map available",
    )
    original_html = _image_or_placeholder(
        image_map.get(sample_id),
        report_path,
        f"Wafer preview for {sample_id}",
        "No wafer preview available",
    )
    cards = "\n".join(_defect_card(item) for item in defects)
    if not cards:
        cards = '<div class="card"><h3>No dominant defect</h3><p class="muted">15점 이상 defect 후보가 없습니다.</p></div>'
    return f"""<article class="focus">
  <div>
    <h3>{html.escape(sample_id)}</h3>
    <div class="image-grid">
      <div>
        <p class="muted"><strong>색칠된 후보 영역</strong></p>
        {annotated_html}
      </div>
      <div>
        <p class="muted"><strong>원본 grade preview</strong></p>
        {original_html}
      </div>
    </div>
  </div>
  <div>
    <h3>Defect 후보</h3>
    <div class="defects">{cards}</div>
  </div>
</article>"""


def _image_or_placeholder(path: Path | None, report_path: Path, alt: str, message: str) -> str:
    if path is None:
        return f'<div class="no-image">{html.escape(message)}<br><span class="muted">manifest 또는 raw PNG 입력으로 실행해야 이미지가 생성됩니다.</span></div>'
    return f'<img class="wafer-img" src="{html.escape(_relpath(path, report_path))}" alt="{html.escape(alt)}">'


def _defect_card(item: dict[str, Any]) -> str:
    family = str(item["defect_family"])
    color = OVERLAY_COLORS.get(family, "#8e8e93")
    label = OVERLAY_LABELS.get(family, family)
    score = float(item["score"])
    return f"""<div class="card">
  <h3><span class="swatch" style="background: {html.escape(color)}"></span>{html.escape(label)} · {score:.1f}</h3>
  <div class="bar"><span style="width: {max(0.0, min(100.0, score)):.1f}%"></span></div>
  <p class="muted">{html.escape(str(item["confidence"]))} confidence · 이미지에서 같은 색 영역을 확인하세요.</p>
  <p>{html.escape(str(item["evidence"]))}</p>
</div>"""


def _score_table_row(row: dict[str, Any]) -> str:
    family = str(row["defect_family"])
    color = OVERLAY_COLORS.get(family, "#8e8e93")
    label = OVERLAY_LABELS.get(family, family)
    return (
        "<tr>"
        f"<td><span class=\"swatch\" style=\"background: {html.escape(color)}\"></span>{html.escape(label)}</td>"
        f"<td><strong>{float(row['score']):.1f}</strong></td>"
        f"<td>{html.escape(str(row['confidence']))}</td>"
        f"<td>{html.escape(str(row['location']))}</td>"
        f"<td>{html.escape(str(row['evidence']))}</td>"
        "</tr>"
    )


def _defect_rows_by_sample(defect_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in defect_rows:
        grouped.setdefault(str(row.get("sample_id", "")), []).append(row)
    return grouped


def choose_focus_sample_id(feature_rows: list[dict[str, str]], requested: str | None) -> str:
    sample_ids = [str(row.get("sample_id", "")) for row in feature_rows]
    if requested:
        if requested not in sample_ids:
            raise ValueError(f"--focus-sample not found in feature rows: {requested}")
        return requested
    return sample_ids[0]


def _relpath(target: Path, base_file: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base_file.resolve().parent)).as_posix()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = output_dir(args.out, allow_outside_root=args.allow_output_outside_root)

    if args.input:
        features_path, source_sanity_path, source_neighbors_path, batch_metadata_path, manifest_path = run_raw_png_pipeline(args, out_dir)
    elif args.manifest:
        features_path, source_sanity_path, source_neighbors_path, batch_metadata_path, manifest_path = run_manifest_pipeline(args, out_dir)
    else:
        features_path, source_sanity_path, source_neighbors_path, batch_metadata_path, manifest_path = existing_feature_inputs(args)

    feature_rows = read_csv(features_path)
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

    write_csv(defect_scores_path, defect_rows)
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
