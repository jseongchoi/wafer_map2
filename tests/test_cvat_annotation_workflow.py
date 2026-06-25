import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wafermap.data import PATTERN_CLASSES, save_npz, write_json
from wafermap.real import manifest_payload
from wafermap.synth import SyntheticConfig, generate_sample


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_sample_dir(sample, sample_dir: Path) -> None:
    save_npz(sample_dir / "arrays.npz", sample)
    write_json(sample_dir / "metadata.json", {"sample_id": sample.sample_id, **sample.metadata})


def test_cvat_export_writes_images_manifest_and_label_schema(tmp_path):
    exporter = _load_script("export_cvat_wafer_images.py")
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=61), 0)
    sample_dir = tmp_path / "source" / "sample"
    _write_sample_dir(sample, sample_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest_payload(
                [
                    {
                        "sample_id": sample.sample_id,
                        "source_type": "synthetic_sample_dir",
                        "sample_dir": str(sample_dir.relative_to(tmp_path)),
                    }
                ]
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = exporter.export_cvat_images(
        manifest_path=manifest_path,
        out_dir=tmp_path / "cvat",
        label_schema_path=Path("configs/cvat/wafer_defect_labels.json").resolve(),
        sample_ids=None,
        limit=0,
    )

    image_path = tmp_path / "cvat" / payload["samples"][0]["image_path"]
    assert image_path.exists()
    assert (tmp_path / "cvat" / "labels.json").exists()
    assert (tmp_path / "cvat" / "manifest.json").exists()
    with Image.open(image_path) as image:
        assert image.size == (sample.shape[1], sample.shape[0])
    assert any(label["name"] == "stby_blob" for label in payload["labels"])


def test_cvat_polygon_import_creates_pattern_assets_and_preserves_stby_blob_grade(tmp_path):
    exporter = _load_script("export_cvat_wafer_images.py")
    importer = _load_script("import_cvat_annotations.py")
    composer = _load_script("compose_synthetic_from_assets.py")
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=62), 0)
    sample.severity[:] = 0
    sample.stby_mask[:] = 0
    sample.valid_test_mask[:] = sample.wafer_mask
    sample.stby_mask[2:6, 2:6] = 1
    sample.valid_test_mask[2:6, 2:6] = 0
    sample.pattern_masks[:] = 0
    sample_dir = tmp_path / "source" / "sample"
    _write_sample_dir(sample, sample_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest_payload(
                [
                    {
                        "sample_id": sample.sample_id,
                        "source_type": "synthetic_sample_dir",
                        "sample_dir": str(sample_dir.relative_to(tmp_path)),
                    }
                ]
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cvat_dir = tmp_path / "cvat"
    cvat_manifest = exporter.export_cvat_images(
        manifest_path=manifest_path,
        out_dir=cvat_dir,
        label_schema_path=Path("configs/cvat/wafer_defect_labels.json").resolve(),
        sample_ids=None,
        limit=0,
    )
    image_name = cvat_manifest["samples"][0]["image_name"]
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="{image_name}" width="{sample.shape[1]}" height="{sample.shape[0]}">
    <polygon label="stby_blob" points="1,1;7,1;7,7;1,7"/>
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    result = importer.import_cvat_annotations(
        cvat_xml=xml_path,
        cvat_manifest_path=cvat_dir / "manifest.json",
        label_schema_path=Path("configs/cvat/wafer_defect_labels.json").resolve(),
        assets_root=tmp_path / "assets",
        margin_ratio=0.0,
    )

    assert result["saved_count"] == 1
    asset_dir = Path(result["saved"][0]["path"])
    assert asset_dir.parent.name == "local"
    metadata = json.loads((asset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["annotation_source"]["tool"] == "CVAT"
    assert metadata["annotation_source"]["labels"] == ["stby_blob"]
    assert metadata["grade_max"] == 7
    assets = composer.load_assets(tmp_path / "assets")
    assert assets[0]["family"] == "local"
    assert assets[0]["grade"][assets[0]["mask"]].max() == 7

    base = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=63), 0)
    base.severity[:] = 0
    composed = composer.compose_sample(
        base,
        assets,
        composer.random.Random(1),
        1,
        "cvat_composed",
        procedural_families=(),
    )
    assert composed.pattern_masks[PATTERN_CLASSES.index("local")].sum() > 0
    assert composed.severity.max() == 7
