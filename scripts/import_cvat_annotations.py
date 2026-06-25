"""Import CVAT annotations as reusable FBM pattern assets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wafermap.assets import TARGET_FAMILIES, save_pattern_assets

DEFAULT_LABEL_SCHEMA = ROOT / "configs" / "cvat" / "wafer_defect_labels.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvat-xml", required=True, help="CVAT for images annotations.xml path.")
    parser.add_argument("--cvat-manifest", required=True, help="manifest.json written by export_cvat_wafer_images.py.")
    parser.add_argument("--assets-root", default="data/pattern_assets")
    parser.add_argument("--label-schema", default=str(DEFAULT_LABEL_SCHEMA))
    parser.add_argument("--margin-ratio", type=float, default=0.20)
    parser.add_argument("--split-components", action="store_true")
    parser.add_argument("--ignore-unknown-labels", action="store_true")
    return parser.parse_args(argv)


def load_real_feature_module() -> Any:
    path = ROOT / "scripts" / "extract_real_unlabeled_features.py"
    spec = importlib.util.spec_from_file_location("extract_real_unlabeled_features_for_cvat_import", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    spec.loader.exec_module(module)
    return module


def load_label_schema(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "wafer_cvat_label_schema/v1":
        raise ValueError(f"unsupported label schema: {payload.get('schema_version')}")
    labels: dict[str, dict[str, Any]] = {}
    for item in payload.get("labels", []):
        info = dict(item)
        name = str(info.get("name", ""))
        family = str(info.get("asset_family", ""))
        if not name or not family:
            raise ValueError(f"label must define name and asset_family: {item}")
        if family not in TARGET_FAMILIES:
            raise ValueError(f"label {name} maps to unsupported asset_family={family}")
        labels[name] = info
        for alias in info.get("aliases", []):
            labels[str(alias)] = info
    return labels


def load_export_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "wafer_cvat_export/v1":
        raise ValueError(f"unsupported CVAT export manifest: {payload.get('schema_version')}")
    return payload


def load_source_samples(cvat_manifest: dict[str, Any]) -> dict[str, Any]:
    module = load_real_feature_module()
    source_manifest = Path(str(cvat_manifest["source_manifest"])).resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    module.validate_manifest(manifest)
    wanted = {str(item["sample_id"]) for item in cvat_manifest.get("samples", [])}
    samples: dict[str, Any] = {}
    for entry in manifest.get("samples", []):
        sample_id = str(entry.get("sample_id", ""))
        if sample_id in wanted:
            samples[sample_id] = module.load_real_like_sample(entry, source_manifest)
    missing = sorted(wanted - set(samples))
    if missing:
        raise ValueError(f"source manifest no longer contains exported samples: {', '.join(missing)}")
    return samples


def import_cvat_annotations(
    *,
    cvat_xml: Path,
    cvat_manifest_path: Path,
    label_schema_path: Path,
    assets_root: Path,
    margin_ratio: float,
    split_components: bool = False,
    ignore_unknown_labels: bool = False,
) -> dict[str, Any]:
    cvat_manifest = load_export_manifest(cvat_manifest_path)
    labels = load_label_schema(label_schema_path)
    samples = load_source_samples(cvat_manifest)
    image_to_sample = {Path(str(item["image_name"])).name: str(item["sample_id"]) for item in cvat_manifest["samples"]}
    image_to_sample.update({Path(str(item.get("image_path", ""))).name: str(item["sample_id"]) for item in cvat_manifest["samples"]})

    root = ET.parse(cvat_xml).getroot()
    saved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for image in root.findall(".//image"):
        image_name = Path(str(image.get("name", ""))).name
        sample_id = image_to_sample.get(image_name)
        if sample_id is None:
            skipped.append({"image": image_name, "reason": "image not in cvat manifest"})
            continue
        sample = samples[sample_id]
        masks_by_family: dict[str, np.ndarray] = {
            family: np.zeros(sample.shape, dtype=bool) for family in TARGET_FAMILIES
        }
        labels_by_family: dict[str, set[str]] = defaultdict(set)
        severity = sample.severity.copy()
        for shape in list(image.findall("polygon")) + list(image.findall("box")):
            label_name = str(shape.get("label", ""))
            label_info = labels.get(label_name)
            if label_info is None:
                if ignore_unknown_labels:
                    skipped.append({"image": image_name, "label": label_name, "reason": "unknown label"})
                    continue
                raise ValueError(f"unknown CVAT label {label_name!r}; add it to {label_schema_path}")
            mask = shape_mask(shape, sample.shape)
            if not mask.any():
                continue
            family = str(label_info["asset_family"])
            masks_by_family[family] |= mask
            labels_by_family[family].add(str(label_info["name"]))
            if "grade_override" in label_info:
                grade = int(label_info["grade_override"])
                severity[mask] = np.maximum(severity[mask], grade).astype(np.uint8)

        nonempty = {family: mask for family, mask in masks_by_family.items() if mask.any()}
        if not nonempty:
            continue
        sample_for_assets = replace(sample, severity=severity, metadata=dict(sample.metadata))
        sample_saved = save_pattern_assets(
            sample=sample_for_assets,
            masks_by_family=nonempty,
            assets_root=assets_root,
            margin_ratio=margin_ratio,
            source_manifest=Path(str(cvat_manifest["source_manifest"])).resolve(),
            split_components=split_components,
        )
        for item in sample_saved:
            annotate_asset_metadata(
                Path(item["path"]),
                cvat_xml=cvat_xml,
                cvat_manifest=cvat_manifest_path,
                labels=sorted(labels_by_family.get(item["family"], [])),
            )
        saved.extend(sample_saved)

    payload = {
        "schema_version": "wafer_cvat_import_report/v1",
        "cvat_xml": str(cvat_xml),
        "cvat_manifest": str(cvat_manifest_path),
        "assets_root": str(assets_root),
        "saved_count": len(saved),
        "saved": saved,
        "skipped": skipped,
    }
    assets_root.mkdir(parents=True, exist_ok=True)
    (assets_root / "cvat_import_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def shape_mask(shape: ET.Element, image_shape: tuple[int, int]) -> np.ndarray:
    if shape.tag == "polygon":
        points = parse_points(str(shape.get("points", "")))
    elif shape.tag == "box":
        xtl = float(shape.get("xtl", "0"))
        ytl = float(shape.get("ytl", "0"))
        xbr = float(shape.get("xbr", "0"))
        ybr = float(shape.get("ybr", "0"))
        points = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
    else:
        return np.zeros(image_shape, dtype=bool)
    image = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def parse_points(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in value.split(";"):
        if not token.strip():
            continue
        x, y = token.split(",")
        points.append((float(x), float(y)))
    if len(points) < 3:
        raise ValueError(f"polygon requires at least three points: {value}")
    return points


def annotate_asset_metadata(asset_dir: Path, *, cvat_xml: Path, cvat_manifest: Path, labels: list[str]) -> None:
    metadata_path = asset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["annotation_source"] = {
        "tool": "CVAT",
        "format": "CVAT for images 1.1",
        "annotations_xml": str(cvat_xml),
        "cvat_manifest": str(cvat_manifest),
        "labels": labels,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = import_cvat_annotations(
        cvat_xml=Path(args.cvat_xml).resolve(),
        cvat_manifest_path=Path(args.cvat_manifest).resolve(),
        label_schema_path=Path(args.label_schema).resolve(),
        assets_root=Path(args.assets_root).resolve(),
        margin_ratio=float(args.margin_ratio),
        split_components=bool(args.split_components),
        ignore_unknown_labels=bool(args.ignore_unknown_labels),
    )
    print(f"Imported CVAT assets: {payload['saved_count']}")
    print(f"Assets root: {payload['assets_root']}")


if __name__ == "__main__":
    main()
