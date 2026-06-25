"""Export wafer map previews and metadata for CVAT annotation tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import (
    DEFAULT_CVAT_FORMAT,
    cvat_labels_for_export,
    load_cvat_label_schema,
    preview_rgb,
    safe_name,
)

DEFAULT_LABEL_SCHEMA = ROOT / "configs" / "cvat" / "wafer_defect_labels.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="real_unlabeled_manifest/v1 JSON path.")
    parser.add_argument("--out-dir", default="data/cvat_exports/wafer_defect_task")
    parser.add_argument("--label-schema", default=str(DEFAULT_LABEL_SCHEMA))
    parser.add_argument("--sample-id", action="append", help="Sample ID to export. Repeat to export several samples.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum sample count after sample-id filtering.")
    return parser.parse_args(argv)


def load_real_feature_module() -> Any:
    path = ROOT / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features_for_cvat_export", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def selected_entries(manifest: dict[str, Any], sample_ids: list[str] | None, limit: int) -> list[dict[str, Any]]:
    entries = list(manifest.get("samples", []))
    if sample_ids:
        wanted = set(sample_ids)
        entries = [entry for entry in entries if str(entry.get("sample_id", "")) in wanted]
        missing = sorted(wanted - {str(entry.get("sample_id", "")) for entry in entries})
        if missing:
            raise ValueError(f"sample-id not found in manifest: {', '.join(missing)}")
    if limit > 0:
        entries = entries[:limit]
    return entries


def export_cvat_images(
    *,
    manifest_path: Path,
    out_dir: Path,
    label_schema_path: Path,
    sample_ids: list[str] | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    module = load_real_feature_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    module.validate_manifest(manifest)
    label_schema = load_cvat_label_schema(label_schema_path)
    entries = selected_entries(manifest, sample_ids, limit)
    if not entries:
        raise ValueError("no samples selected for CVAT export")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(label_schema_path, out_dir / "labels.json")

    samples: list[dict[str, Any]] = []
    for entry in entries:
        sample = module.load_real_like_sample(entry, manifest_path)
        image_name = f"{safe_name(sample.sample_id)}.png"
        image_path = images_dir / image_name
        Image.fromarray(preview_rgb(sample), mode="RGB").save(image_path)
        samples.append(
            {
                "sample_id": sample.sample_id,
                "image_name": image_name,
                "image_path": str(Path("images") / image_name),
                "height": int(sample.shape[0]),
                "width": int(sample.shape[1]),
                "source_type": str(entry.get("source_type", "")),
            }
        )

    payload = {
        "schema_version": "wafer_cvat_export/v1",
        "source_manifest": str(manifest_path),
        "label_schema": str(out_dir / "labels.json"),
        "cvat_format": DEFAULT_CVAT_FORMAT,
        "samples": samples,
        "labels": cvat_labels_for_export(label_schema),
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = export_cvat_images(
        manifest_path=Path(args.manifest).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        label_schema_path=Path(args.label_schema).resolve(),
        sample_ids=args.sample_id,
        limit=int(args.limit),
    )
    print(f"Wrote CVAT image package: {Path(args.out_dir).resolve()}")
    print(f"Images: {len(payload['samples'])}")
    print(f"Manifest: {Path(args.out_dir).resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
