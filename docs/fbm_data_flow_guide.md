# FBM Data Flow Guide

This document is the artifact map. Use [End-To-End Workflow](end_to_end_workflow.md) for the full command sequence.

## Data Path

```text
raw wafer PNG or synthetic sample dir
-> real_unlabeled_manifest/v1
-> local segmentation tool
-> data/pattern_assets
-> data/synthetic/asset_composed
-> outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
-> coordinate-aware small U-Net
-> outputs/predictions/fbm_prediction_masks.json
```

## Folder Contract

| Stage | Path | Git policy |
|---|---|---|
| Raw wafer input | `data/raw/<product>/*.png` | ignored |
| Intermediate arrays | `data/interim/` | ignored |
| Processed local data | `data/processed/` | ignored |
| Pattern assets | `data/pattern_assets/<family>/<asset_id>/` | ignored |
| Synthetic samples | `data/synthetic/<run>/<sample_id>/` | ignored |
| Reports, manifests, predictions, checkpoints | `outputs/` | ignored |
| Source/config/docs/tests | `src/`, `scripts/`, `configs/`, `docs/`, `tests/` | tracked |

Git stores code, configs, schemas, docs, and tests. It should not store wafer source data, saved assets, synthetic samples, model checkpoints, reports, or prediction exports.

## Manifest Outputs

Real PNG analysis writes:

```text
outputs/manifests/real_png_batch_manifest.json
outputs/reports/real_png_batch/
```

The local segmentation tool consumes a manifest plus `sample_id`:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

## Pattern Asset Contract

Each saved defect asset has:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

Metadata includes:

- `schema_version = fbm_pattern_asset/v1`;
- `family`;
- `source_sample_id`;
- `bbox_xywh`;
- `mask_pixel_count`;
- grade range;
- composition rule;
- radial/angular/edge-distance location summary.

## Synthetic Sample Contract

Composed samples live under:

```text
data/synthetic/asset_composed/<sample_id>/
  arrays.npz
  metadata.json
```

Important arrays:

| Array | Meaning |
|---|---|
| `severity` | final grade 0-7 wafer map |
| `wafer_mask` | wafer area |
| `valid_test_mask` | testable area |
| `stby_mask` | missing-test/STBY area |
| `pattern_masks` | family-wise multi-label segmentation target |
| `pattern_intensity` | family-wise normalized intensity |
| `chip_index` | chip coordinate index |

## Training Manifest

Readiness generation writes:

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
outputs/pattern_asset_pipeline/asset_segmentation_readiness.html
outputs/pattern_asset_pipeline/asset_segmentation_readiness_metrics.json
outputs/pattern_asset_pipeline/asset_segmentation_gallery.png
```

`asset_segmentation_manifest.csv` is the direct input to `train_unet_segmentation.py`.

## Model Artifacts

Training writes:

```text
outputs/models/asset_unet_segmentation.pt
outputs/pattern_asset_pipeline/asset_unet_segmentation.html
outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json
```

The coordinate-aware small U-Net input contains severity, mask, and position channels:

```text
severity_mean, severity_max, fail_density,
wafer_mask, valid_test_mask, stby_mask,
x_norm, y_norm, radial_norm, angle_sin, angle_cos, edge_distance_norm
```

`export_unet_predictions.py` writes:

```text
outputs/predictions/fbm_prediction_masks.json
```

That file uses `fbm_prediction_masks/v1` and is loaded by `run_segmentation_tool.py --prediction-json`.
