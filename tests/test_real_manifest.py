import pytest

from wafermap.real import (
    OBSERVABLE_FEATURE_SCHEMA_VERSION,
    REAL_UNLABELED_SCHEMA_VERSION,
    SOURCE_TYPE_PNG_GRAYSCALE_RAW,
    SOURCE_TYPE_SYNTHETIC_SAMPLE_DIR,
    manifest_payload,
    validate_manifest,
)


def test_manifest_payload_uses_public_schema_versions():
    payload = manifest_payload(
        [
            {
                "sample_id": "sample_001",
                "source_type": SOURCE_TYPE_SYNTHETIC_SAMPLE_DIR,
                "sample_dir": "synthetic/sample_001",
            }
        ]
    )

    assert payload["schema_version"] == REAL_UNLABELED_SCHEMA_VERSION
    assert payload["feature_schema_version"] == OBSERVABLE_FEATURE_SCHEMA_VERSION
    validate_manifest(payload)


def test_validate_manifest_accepts_raw_png_contract():
    validate_manifest(
        manifest_payload(
            [
                {
                    "sample_id": "product_a_wafer_001",
                    "source_type": SOURCE_TYPE_PNG_GRAYSCALE_RAW,
                    "png_path": "product_a/wafer_001.png",
                    "parser_name": "png_raw_folder_batch",
                    "parser_version": "0.1.0",
                    "orientation": "not_rotated",
                    "chip_blocks": {"width": 2, "height": 2},
                    "grid": {"rows": 10, "cols": 10},
                }
            ]
        )
    )


def test_validate_manifest_requires_raw_png_geometry_unless_inference_is_explicit():
    payload = manifest_payload(
        [
                {
                    "sample_id": "product_a_wafer_001",
                    "source_type": SOURCE_TYPE_PNG_GRAYSCALE_RAW,
                    "png_path": "product_a/wafer_001.png",
                    "parser_name": "png_raw_folder_batch",
                "parser_version": "0.1.0",
                "orientation": "not_rotated",
            }
        ]
    )

    with pytest.raises(ValueError, match="allow_geometry_inference=true"):
        validate_manifest(payload)

    payload["samples"][0]["allow_geometry_inference"] = True
    validate_manifest(payload)


def test_validate_manifest_rejects_duplicate_sample_ids():
    payload = manifest_payload(
        [
            {
                "sample_id": "dup",
                "source_type": SOURCE_TYPE_SYNTHETIC_SAMPLE_DIR,
                "sample_dir": "a",
            },
            {
                "sample_id": "dup",
                "source_type": SOURCE_TYPE_SYNTHETIC_SAMPLE_DIR,
                "sample_dir": "b",
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_manifest(payload)


def test_validate_manifest_accepts_readable_real_sample_id_without_extra_flag():
    payload = manifest_payload(
        [
            {
                "sample_id": "lot123_wafer07",
                "source_type": SOURCE_TYPE_PNG_GRAYSCALE_RAW,
                "png_path": "product_a/wafer_001.png",
                "parser_name": "png_raw_folder_batch",
                "parser_version": "0.1.0",
                "orientation": "not_rotated",
                "chip_blocks": {"width": 2, "height": 2},
                "grid": {"rows": 10, "cols": 10},
            }
        ]
    )

    validate_manifest(payload)
