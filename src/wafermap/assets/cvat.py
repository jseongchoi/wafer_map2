"""CVAT label schema helpers for wafer pattern asset workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wafermap.assets.library import TARGET_FAMILIES

SCHEMA_VERSION = "wafer_cvat_label_schema/v1"
DEFAULT_CVAT_FORMAT = "CVAT for images 1.1"


def load_cvat_label_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported label schema: {payload.get('schema_version')}")
    labels = payload.get("labels", [])
    if not isinstance(labels, list) or not labels:
        raise ValueError("label schema must contain a non-empty labels list")

    names: set[str] = set()
    for label in labels:
        name = str(label.get("name", "")).strip()
        family = str(label.get("asset_family", "")).strip()
        if not name or not family:
            raise ValueError(f"label must define name and asset_family: {label}")
        if family not in TARGET_FAMILIES:
            raise ValueError(f"label {name} maps to unsupported asset_family={family}")
        if name in names:
            raise ValueError(f"duplicate CVAT label name: {name}")
        names.add(name)
    return payload


def cvat_label_lookup(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in schema["labels"]:
        info = dict(item)
        keys = [str(info["name"]), *(str(alias) for alias in info.get("aliases", []))]
        for key in keys:
            if key in lookup:
                raise ValueError(f"duplicate CVAT label or alias: {key}")
            lookup[key] = info
    return lookup


def cvat_labels_for_export(schema: dict[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for label in schema["labels"]:
        labels.append(
            {
                "name": label["name"],
                "display_name": label.get("display_name", label["name"]),
                "asset_family": label["asset_family"],
                "color": label.get("color", ""),
                "aliases": label.get("aliases", []),
            }
        )
    return labels
