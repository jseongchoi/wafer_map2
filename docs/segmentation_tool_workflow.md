# Segmentation Tool Workflow

The in-repo segmentation tool is the local mask editing surface for wafer defect assets. It does not train the model; it turns wafer samples and optional model predictions into reusable `mask.png`, `grade.png`, `preview.png`, and `metadata.json` assets.

Use [End-To-End Workflow](end_to_end_workflow.md) for the full pipeline.

## Open A Wafer

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

Smoke sample:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

Model-assisted correction:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5

python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --prediction-json outputs/predictions/fbm_prediction_masks.json `
  --assets-root data/pattern_assets
```

`--prediction-json` must use `fbm_prediction_masks/v1` and matching `sample_id`.

## Families

The tool edits multi-label masks for:

```text
local, scratch, ring, edge, shot_grid, random
```

A pixel can belong to more than one family when the real defect is ambiguous or overlapping.

## Controls

| Control | Use |
|---|---|
| `Family` | Select target defect family. |
| `Paint` / `Erase` | Add or remove mask pixels. |
| `Pan`, `Zoom -`, `Zoom +`, `Fit View` | Inspect large wafers. |
| `Map Colors` | Change grade display colors without changing masks. |
| `Grow Seed` | Expand seed pixels through connected valid pixels above `Min Grade`. |
| `Grade Area` | Add all valid pixels above `Min Grade`. |
| `Lasso Fit` | Fill matching high-grade pixels inside a rough loop. |
| `Trace Line` | Extend scratch-like masks from painted seed pixels. |
| `Analyze` / `Load Model` | Preview geometry or model proposals. |
| `Geometry Fit` | Build edge/ring proposals from radius, width, and angle controls. |

## Annotation Rules

| Family | Rule |
|---|---|
| `local` | Compact blob or local cluster only. |
| `scratch` | Visible line, arc, or scratch-like path. |
| `ring` | Ring, annulus, or arc band; do not fill the whole disk. |
| `edge` | Abnormal edge band or sector; do not label the normal wafer boundary. |
| `shot_grid` | Repeated shot-relative pattern. |
| `random` | Sparse fail pattern without stable shape or process geometry. |

Large wafer-wide patterns should be segmented by geometry, not ignored. Use `edge`, `ring`, `shot_grid`, or `random` depending on the pattern.

## Save Pattern Assets

Saved assets use:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

`metadata.json` records the source wafer, family, bbox, pixel count, grade range, composition rule, and location summary. The location summary stores radial range, angular span, edge distance, and wafer-area ratio.

Save modes:

| Mode | Use |
|---|---|
| `One Family Asset` | Save all pixels in the active family as one asset. Best for rings or continuous patterns. |
| `Split Components` | Save disconnected components separately. Best for separate local blobs. |

## Review

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

Check mask tightness, family correctness, ring continuity, edge quality, and whether STBY/missing-test regions were incorrectly treated as defects.

## App Quality Checks

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py -q --basetemp .pytest_tmp_tool
```

This covers asset saving, sample payloads, prediction mask prefill, model proposal loading, canvas downsampling, geometry proposals, and the end-to-end pattern asset pipeline. Default pytest skips slow end-to-end tests; use `--run-slow` only for full pipeline validation.
