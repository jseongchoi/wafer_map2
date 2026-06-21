"""Build a real-unlabeled manifest from product folders of raw grayscale PNGs.

Expected input layout:

raw_root/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_001.png

The script writes a manifest and, unless --manifest-only is set, runs the
existing real-unlabeled feature extraction workflow on that manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.real import SOURCE_TYPE_PNG_GRAYSCALE_RAW, load_png_gray_values, manifest_payload, resolve_png_geometry

PARSER_NAME = "png_raw_folder_batch"
PARSER_VERSION = "0.1.0"
SAMPLE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ProductGroup:
    folder_name: str
    alias: str
    path: Path
    png_paths: list[Path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, help="Root directory containing product subfolders of PNG files.")
    parser.add_argument("--out-dir", default="outputs/reports/png_raw_batch")
    parser.add_argument(
        "--manifest-out",
        help="Private manifest output path. Defaults to outputs/private/<out-dir-name>_manifest.json.",
    )
    parser.add_argument(
        "--allow-workspace-manifest-output",
        action="store_true",
        help="Allow --manifest-out under the workspace outside outputs/private. Use only for local smoke tests.",
    )
    parser.add_argument(
        "--allow-output-outside-root",
        action="store_true",
        help="Allow derived report outputs outside outputs/. Use only for isolated synthetic smoke tests.",
    )
    parser.add_argument("--glob", default="*.png", help="PNG filename pattern inside each product folder.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan PNGs directly under each product folder.")
    parser.add_argument("--limit-per-product", type=positive_int)
    parser.add_argument(
        "--use-folder-names",
        action="store_true",
        help="Disabled for real data security; sample_id always uses opaque generated aliases.",
    )
    parser.add_argument(
        "--geometry-json",
        help=(
            "Optional product geometry JSON. Keys are product folder names or aliases; values may include "
            "chip_blocks, grid, and actual_net_die."
        ),
    )
    parser.add_argument(
        "--wafer-mask-strategy",
        default="centered_ellipse_from_png",
        choices=("centered_ellipse_from_png", "full_grid_from_png"),
    )
    parser.add_argument("--orientation", default="not_rotated")
    parser.add_argument(
        "--allow-workspace-input",
        action="store_true",
        help="Allow PNG inputs under the current workspace. Useful for local smoke tests only.",
    )
    parser.add_argument(
        "--production-run",
        action="store_true",
        help=(
            "Enable real-data production guardrails: raw input outside workspace, explicit per-product "
            "geometry with actual_net_die, private manifest, reports output, and reference features."
        ),
    )
    parser.add_argument("--manifest-only", action="store_true", help="Only write the manifest; do not run extraction.")
    parser.add_argument("--reference-features", help="Optional reference feature CSV for nearest-neighbor review.")
    parser.add_argument("--cpu-model", help="Optional CPU encoder .npz model for unlabeled scoring after extraction.")
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--include-reference-labels", action="store_true")
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_geometry_json(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("geometry-json must contain an object keyed by product name or alias")
    return {str(key): dict(value) for key, value in payload.items()}


def discover_products(
    raw_root: Path,
    *,
    glob_pattern: str,
    recursive: bool,
    limit_per_product: int | None,
    use_folder_names: bool,
) -> list[ProductGroup]:
    if use_folder_names:
        raise ValueError("--use-folder-names is disabled; sample_id must stay opaque for shareable outputs")
    if not raw_root.exists() or not raw_root.is_dir():
        raise ValueError(f"raw-root must be an existing directory: {raw_root}")

    product_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not product_dirs:
        product_dirs = [raw_root]

    groups: list[ProductGroup] = []
    for idx, product_dir in enumerate(product_dirs):
        iterator = product_dir.rglob(glob_pattern) if recursive else product_dir.glob(glob_pattern)
        png_paths = sorted(path for path in iterator if path.is_file())
        if limit_per_product is not None:
            png_paths = png_paths[:limit_per_product]
        if not png_paths:
            continue
        alias = opaque_product_alias()
        groups.append(ProductGroup(product_dir.name, alias, product_dir.resolve(), [path.resolve() for path in png_paths]))

    if not groups:
        raise ValueError(f"No PNG files found under {raw_root} with pattern {glob_pattern!r}")
    return groups


def safe_token(value: str) -> str:
    token = SAMPLE_TOKEN_RE.sub("_", value).strip("._-")
    return token or "product"


def stable_hash(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def opaque_product_alias() -> str:
    return f"product_{secrets.token_hex(5)}"


def stable_wafer_alias(group: ProductGroup, png_path: Path) -> str:
    try:
        rel = png_path.relative_to(group.path).as_posix()
    except ValueError:
        rel = png_path.name
    return f"w{stable_hash(rel.lower())}"


def geometry_for_product(
    group: ProductGroup,
    geometry_by_product: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    explicit = geometry_by_product.get(group.folder_name) or geometry_by_product.get(group.alias)
    if explicit:
        geometry = normalize_geometry(explicit)
        validate_product_png_shapes(group, geometry)
        return geometry
    return infer_geometry_from_product_pngs(group)


def validate_production_run(
    args: argparse.Namespace,
    *,
    raw_root: Path,
    out_dir: Path,
    manifest_path: Path,
    groups: list[ProductGroup],
    geometry_by_product: dict[str, dict[str, Any]],
) -> None:
    if not args.production_run:
        return
    if args.allow_workspace_input:
        raise ValueError("--production-run forbids --allow-workspace-input")
    if args.allow_workspace_manifest_output:
        raise ValueError("--production-run forbids --allow-workspace-manifest-output")
    if args.allow_output_outside_root:
        raise ValueError("--production-run forbids --allow-output-outside-root")
    if args.manifest_only:
        raise ValueError("--production-run must generate reports, not manifest-only output")
    if _is_inside(raw_root, ROOT):
        raise ValueError("--production-run raw-root must be outside the workspace")
    reports_root = (ROOT / "outputs" / "reports").resolve()
    if not _is_inside(out_dir, reports_root):
        raise ValueError("--production-run out-dir must be under outputs/reports")
    private_root = (ROOT / "outputs" / "private").resolve()
    if not _is_inside(manifest_path, private_root):
        raise ValueError("--production-run manifest-out must be under outputs/private")
    if not args.geometry_json:
        raise ValueError("--production-run requires --geometry-json with explicit product geometry")
    if not args.reference_features:
        raise ValueError("--production-run requires --reference-features for reviewer nearest-neighbor output")

    missing = []
    invalid_net_die = []
    for group in groups:
        explicit = geometry_by_product.get(group.folder_name) or geometry_by_product.get(group.alias)
        if explicit is None:
            missing.append(group.folder_name)
        elif int(explicit.get("actual_net_die", 0)) <= 0:
            invalid_net_die.append(group.folder_name)
    if missing:
        raise ValueError(f"--production-run geometry-json missing products: {', '.join(missing[:10])}")
    if invalid_net_die:
        raise ValueError(f"--production-run geometry-json requires positive actual_net_die: {', '.join(invalid_net_die[:10])}")


def normalize_geometry(payload: dict[str, Any]) -> dict[str, Any]:
    if "chip_blocks" not in payload or "grid" not in payload:
        raise ValueError("Explicit product geometry requires chip_blocks and grid")
    chip_width = int(payload["chip_blocks"]["width"])
    chip_height = int(payload["chip_blocks"]["height"])
    rows = int(payload["grid"]["rows"])
    cols = int(payload["grid"]["cols"])
    if chip_width < 1 or chip_height < 1:
        raise ValueError("Explicit product geometry chip_blocks width/height must be positive")
    if rows < 1 or cols < 1:
        raise ValueError("Explicit product geometry grid rows/cols must be positive")
    geometry: dict[str, Any] = {
        "chip_blocks": {"width": chip_width, "height": chip_height},
        "grid": {"rows": rows, "cols": cols},
    }
    if "actual_net_die" in payload:
        actual_net_die = int(payload["actual_net_die"])
        if actual_net_die < 0:
            raise ValueError("Explicit product geometry actual_net_die must be non-negative")
        if actual_net_die > rows * cols:
            raise ValueError("Explicit product geometry actual_net_die cannot exceed grid rows*cols")
        geometry["actual_net_die"] = actual_net_die
    return geometry


def infer_geometry_from_product_pngs(group: ProductGroup) -> dict[str, Any]:
    failures: list[str] = []
    geometries: dict[tuple[int, int, int, int], list[str]] = {}
    raw_shapes: dict[str, tuple[int, int]] = {}
    for png_path in group.png_paths:
        try:
            raw = load_png_gray_values(png_path)
            raw_shapes[png_path.name] = raw.shape
            chip_width, chip_height, rows, cols = resolve_png_geometry({"allow_geometry_inference": True}, raw)
        except Exception as exc:  # noqa: BLE001 - surfaced with product/file context below.
            failures.append(f"{png_path.name}: {exc}")
            continue
        geometries.setdefault((chip_width, chip_height, rows, cols), []).append(png_path.name)
    if len(geometries) > 1:
        details = "; ".join(f"{geometry}: {names[:5]}" for geometry, names in geometries.items())
        raise ValueError(f"Inconsistent inferred chip geometry in {group.folder_name}: {details}")
    if geometries:
        chip_width, chip_height, rows, cols = next(iter(geometries))
        geometry = {
            "chip_blocks": {"width": chip_width, "height": chip_height},
            "grid": {"rows": rows, "cols": cols},
        }
        mismatches = [
            f"{name}: shape {shape}"
            for name, shape in raw_shapes.items()
            if shape != (rows * chip_height, cols * chip_width)
        ]
        if mismatches:
            raise ValueError(
                f"PNG shape mismatch in {group.folder_name}; expected {(rows * chip_height, cols * chip_width)}. "
                f"Files: {'; '.join(mismatches[:5])}"
            )
        return geometry
    raise ValueError(
        "Could not infer chip geometry for "
        f"{group.folder_name}. Add --geometry-json for this product. Tried: {'; '.join(failures[:5])}"
    )


def validate_product_png_shapes(group: ProductGroup, geometry: dict[str, Any]) -> None:
    chip_width = int(geometry["chip_blocks"]["width"])
    chip_height = int(geometry["chip_blocks"]["height"])
    rows = int(geometry["grid"]["rows"])
    cols = int(geometry["grid"]["cols"])
    expected = (rows * chip_height, cols * chip_width)
    mismatches = []
    for png_path in group.png_paths:
        raw = load_png_gray_values(png_path)
        if raw.shape != expected:
            mismatches.append(f"{png_path.name}: shape {raw.shape}")
    if mismatches:
        raise ValueError(
            f"Explicit geometry for {group.folder_name} implies shape {expected}, "
            f"but PNG shapes differ: {'; '.join(mismatches[:5])}"
        )


def build_manifest(
    groups: list[ProductGroup],
    *,
    geometry_by_product: dict[str, dict[str, Any]],
    wafer_mask_strategy: str,
    orientation: str,
    allow_workspace_input: bool,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for group in groups:
        geometry = geometry_for_product(group, geometry_by_product)
        for png_path in group.png_paths:
            entry: dict[str, Any] = {
                "sample_id": f"{group.alias}_{stable_wafer_alias(group, png_path)}",
                "source_type": SOURCE_TYPE_PNG_GRAYSCALE_RAW,
                "png_path": str(png_path),
                "pseudonymized": True,
                "parser_name": PARSER_NAME,
                "parser_version": PARSER_VERSION,
                "orientation": orientation,
                "wafer_mask_strategy": wafer_mask_strategy,
                "chip_blocks": geometry["chip_blocks"],
                "grid": geometry["grid"],
            }
            if "actual_net_die" in geometry:
                entry["actual_net_die"] = geometry["actual_net_die"]
            if allow_workspace_input:
                entry["allow_workspace_input"] = True
            samples.append(entry)
    return manifest_payload(samples)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_batch_metadata(
    args: argparse.Namespace,
    *,
    groups: list[ProductGroup],
    manifest_path: Path,
    geometry_by_product: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    explicit_count = 0
    actual_net_die_count = 0
    for group in groups:
        explicit = geometry_by_product.get(group.folder_name) or geometry_by_product.get(group.alias)
        if explicit is not None:
            explicit_count += 1
            actual_net_die_count += int(int(explicit.get("actual_net_die", 0)) > 0)
    return {
        "schema_version": "png_raw_batch_metadata/v1",
        "generated_at": utc_now_iso(),
        "production_run": bool(args.production_run),
        "product_count": len(groups),
        "png_sample_count": sum(len(group.png_paths) for group in groups),
        "geometry_contract": "explicit" if args.geometry_json else "inferred_from_stby_smoke",
        "explicit_geometry_product_count": explicit_count,
        "actual_net_die_product_count": actual_net_die_count,
        "wafer_mask_strategy": args.wafer_mask_strategy,
        "manifest_location": "outputs/private" if _is_inside(manifest_path, ROOT / "outputs" / "private") else "external_or_test",
        "reference_features": bool(args.reference_features),
        "cpu_model_scoring": bool(args.cpu_model),
        "top_k": int(args.top_k),
    }


def validate_manifest_output_path(path: Path, *, allow_workspace_manifest_output: bool) -> None:
    if not _is_inside(path, ROOT):
        return
    private_root = (ROOT / "outputs" / "private").resolve()
    if _is_inside(path, private_root):
        return
    if allow_workspace_manifest_output:
        return
    raise ValueError(
        "--manifest-out inside the workspace must be under outputs/private unless "
        "--allow-workspace-manifest-output is set"
    )


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run_extraction(args: argparse.Namespace, manifest_path: Path, out_dir: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "extract_real_unlabeled_features.py"),
        "--manifest",
        str(manifest_path),
        "--features-out",
        str(out_dir / "features.csv"),
        "--sanity-out",
        str(out_dir / "sanity.json"),
        "--report-out",
        str(out_dir / "report.html"),
        "--neighbors-out",
        str(out_dir / "neighbors.csv"),
        "--review-template-out",
        str(out_dir / "review_template.csv"),
        "--top-k",
        str(args.top_k),
    ]
    if args.reference_features:
        command.extend(["--reference-features", args.reference_features])
    if args.include_reference_labels:
        command.append("--include-reference-labels")
    if args.allow_output_outside_root:
        command.append("--allow-output-outside-root")
    subprocess.run(command, check=True)


def run_cpu_scoring(args: argparse.Namespace, manifest_path: Path, out_dir: Path) -> None:
    if not args.cpu_model:
        return
    command = [
        sys.executable,
        str(ROOT / "scripts" / "score_unlabeled_cpu_encoder.py"),
        "--model",
        str(args.cpu_model),
        "--manifest",
        str(manifest_path),
        "--predictions-out",
        str(out_dir / "cpu_encoder_predictions.csv"),
        "--neighbors-out",
        str(out_dir / "cpu_encoder_neighbors.csv"),
        "--sanity-out",
        str(out_dir / "cpu_encoder_sanity.json"),
        "--report-out",
        str(out_dir / "cpu_encoder_report.html"),
        "--top-k",
        str(args.top_k),
    ]
    if args.allow_output_outside_root:
        command.append("--allow-output-outside-root")
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    groups = discover_products(
        raw_root,
        glob_pattern=args.glob,
        recursive=not args.no_recursive,
        limit_per_product=args.limit_per_product,
        use_folder_names=args.use_folder_names,
    )
    geometry_by_product = load_geometry_json(args.geometry_json)
    manifest = build_manifest(
        groups,
        geometry_by_product=geometry_by_product,
        wafer_mask_strategy=args.wafer_mask_strategy,
        orientation=args.orientation,
        allow_workspace_input=args.allow_workspace_input,
    )
    if args.manifest_out:
        manifest_path = Path(args.manifest_out).resolve()
    else:
        manifest_path = (ROOT / "outputs" / "private" / f"{safe_token(out_dir.name)}_manifest.json").resolve()
    validate_production_run(
        args,
        raw_root=raw_root,
        out_dir=out_dir,
        manifest_path=manifest_path,
        groups=groups,
        geometry_by_product=geometry_by_product,
    )
    validate_manifest_output_path(
        manifest_path,
        allow_workspace_manifest_output=args.allow_workspace_manifest_output,
    )
    batch_metadata = build_batch_metadata(
        args,
        groups=groups,
        manifest_path=manifest_path,
        geometry_by_product=geometry_by_product,
    )
    write_json(manifest_path, manifest)
    write_json(out_dir / "batch_metadata.json", batch_metadata)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote batch metadata: {out_dir / 'batch_metadata.json'}")
    print(f"Products: {len(groups)}")
    print(f"PNG samples: {len(manifest['samples'])}")
    if not args.manifest_only:
        run_extraction(args, manifest_path, out_dir)
        run_cpu_scoring(args, manifest_path, out_dir)


if __name__ == "__main__":
    main()
