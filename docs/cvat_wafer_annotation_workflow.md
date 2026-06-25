# CVAT Wafer Defect Annotation Workflow

This workflow makes CVAT the primary annotation surface and keeps the in-repo Pattern Asset Editor as a legacy fallback/reference tool.

The near-term pipeline is:

```text
real/unlabeled wafer manifest
-> CVAT image package
-> CVAT polygon/mask annotation
-> pattern asset library
-> synthetic segmentation dataset composition
```

Model training and evaluation are intentionally out of scope for this phase.

## Label Schema

CVAT labels are managed in:

```text
configs/cvat/wafer_defect_labels.json
```

Each label has:

- `name`: CVAT label name.
- `asset_family`: current pattern asset family used by the composer.
- `color`: recommended CVAT label color.
- `aliases`: optional names accepted during import.
- `grade_override`: optional grade to write into the saved asset mask.

Current important mapping:

```text
stby_blob, stby_fail, missing_test_blob -> asset_family local, grade_override 7
```

This lets CVAT expose STBY/missing-test mosaic chips as their own label while the current synthetic composer can still consume them as `local` pattern assets. When the segmentation model later needs a dedicated output channel, add a new model class in the data schema and update this mapping.

## 1. Export Wafer Images For CVAT

Use a `real_unlabeled_manifest/v1` input. Synthetic smoke manifests also work because they use the same real-like loader.

```powershell
python scripts/export_cvat_wafer_images.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --out-dir data/cvat_exports/smoke_task `
  --limit 10
```

Output:

```text
data/cvat_exports/smoke_task/
  images/
    real_like_synth_000000.png
    ...
  labels.json
  manifest.json
```

Upload the `images/` PNG files to CVAT. Create labels from `labels.json`.

Recommended CVAT export format after annotation:

```text
CVAT for images 1.1
```

The importer currently supports `polygon` and `box` shapes from `annotations.xml`. CVAT brush masks can be exported by CVAT native format too, but mask RLE import should be added as a follow-up if brush masks become the main workflow.

## 2. Import CVAT Annotations As Pattern Assets

After annotating and exporting CVAT annotations:

```powershell
python scripts/import_cvat_annotations.py `
  --cvat-xml data/cvat_exports/smoke_task/annotations.xml `
  --cvat-manifest data/cvat_exports/smoke_task/manifest.json `
  --assets-root data/pattern_assets
```

The importer:

- maps CVAT labels through `configs/cvat/wafer_defect_labels.json`,
- rasterizes polygons/boxes into binary masks,
- combines masks with the original wafer arrays,
- saves assets using the existing asset format:

```text
data/pattern_assets/<asset_family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

The metadata includes:

```json
"annotation_source": {
  "tool": "CVAT",
  "format": "CVAT for images 1.1",
  "labels": ["stby_blob"]
}
```

## 3. Compose Synthetic Dataset From Imported Assets

Use the existing composer:

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/cvat_asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

For the integrated report/manifest pipeline:

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/cvat_asset_composed `
  --work-dir outputs/cvat_pattern_asset_pipeline `
  --report-out outputs/reports/cvat_pattern_asset_project_report.html
```

## Legacy Fallback Editor

The browser editor remains available:

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

Use it for quick local experiments, fallback corrections, or validating wafer-specific UX ideas. Do not extend it as the primary annotation product unless CVAT cannot cover the workflow.

## Validation

Run:

```powershell
python -m pytest tests/test_cvat_annotation_workflow.py -q
python -m pytest tests/test_pattern_asset_pipeline.py -q
```

The CVAT workflow tests cover:

- CVAT image package export,
- label schema copy,
- CVAT polygon import,
- `stby_blob` grade override,
- imported asset compatibility with the synthetic composer.
