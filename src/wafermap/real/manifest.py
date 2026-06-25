"""Manifest contract for real-unlabeled wafer inputs."""

from __future__ import annotations

from typing import Any

REAL_UNLABELED_SCHEMA_VERSION = "real_unlabeled_manifest/v1"
OBSERVABLE_FEATURE_SCHEMA_VERSION = "observable_fbm_features/v1"

SOURCE_TYPE_NPZ_SEMANTIC_ARRAYS = "npz_semantic_arrays"
SOURCE_TYPE_PNG_GRAYSCALE_RAW = "png_grayscale_raw"
SOURCE_TYPE_SYNTHETIC_SAMPLE_DIR = "synthetic_sample_dir"


def manifest_payload(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": REAL_UNLABELED_SCHEMA_VERSION,
        "feature_schema_version": OBSERVABLE_FEATURE_SCHEMA_VERSION,
        "samples": samples,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != REAL_UNLABELED_SCHEMA_VERSION:
        raise ValueError(f"Manifest requires schema_version={REAL_UNLABELED_SCHEMA_VERSION}")
    if manifest.get("feature_schema_version") != OBSERVABLE_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Manifest requires feature_schema_version={OBSERVABLE_FEATURE_SCHEMA_VERSION}")
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
        sample_id = str(entry.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"samples[{idx}] requires sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"samples[{idx}] duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if source_type == SOURCE_TYPE_NPZ_SEMANTIC_ARRAYS:
            required = ("arrays_npz", "chip_blocks", "grid", "parser_name", "parser_version", "orientation")
            missing = [name for name in required if name not in entry]
            if missing:
                raise ValueError(f"samples[{idx}] missing required npz_semantic_arrays fields: {missing}")
        elif source_type == SOURCE_TYPE_PNG_GRAYSCALE_RAW:
            required = ("png_path", "parser_name", "parser_version", "orientation")
            missing = [name for name in required if name not in entry]
            if missing:
                raise ValueError(f"samples[{idx}] missing required png_grayscale_raw fields: {missing}")
            has_geometry = "chip_blocks" in entry and "grid" in entry
            if not has_geometry and entry.get("allow_geometry_inference") is not True:
                raise ValueError(
                    f"samples[{idx}] png_grayscale_raw requires chip_blocks and grid "
                    "unless allow_geometry_inference=true"
                )
        elif source_type == SOURCE_TYPE_SYNTHETIC_SAMPLE_DIR:
            if "sample_dir" not in entry:
                raise ValueError(f"samples[{idx}] missing sample_dir")
        else:
            raise ValueError(f"samples[{idx}] unsupported source_type: {source_type}")
