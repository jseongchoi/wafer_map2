# Pattern Asset And Hybrid Synthetic Data Pipeline

This document explains how real-data masks become reusable pattern assets and then multi-label synthetic training data. Use [End-To-End Workflow](end_to_end_workflow.md) for the full command sequence.

## Core Contract

```text
local segmentation mask
-> data/pattern_assets
-> data/synthetic/asset_composed
-> asset_segmentation_manifest.csv
-> train_unet_segmentation.py
-> export_unet_predictions.py
-> segmentation tool correction loop
```

The goal is pixel-level multi-label segmentation, not wafer-level class labels. A single wafer pixel may belong to more than one defect family.

## Pattern Asset Format

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

| File | Meaning |
|---|---|
| `grade.png` | source wafer crop grade map |
| `mask.png` | binary segmentation mask for this asset |
| `preview.png` | RGB crop preview |
| `metadata.json` | source sample, bbox, family, grade range, location summary |

`metadata.json` uses `fbm_pattern_asset/v1`. The composer reads `mask.png` and pastes only valid masked pixels.

## Asset Families

| Family | Source policy |
|---|---|
| `local` | human asset primary |
| `scratch` | human asset primary plus procedural fallback |
| `ring` | human asset primary |
| `edge` | procedural primary plus optional human asset |
| `shot_grid` | procedural primary plus optional human asset |
| `random` | procedural only |

`edge`, `shot_grid`, and `random` can be generated quickly by rules, but real edge-sector or shot-relative signatures should still be saved as assets when they appear in real data.

## Composition Rule

```text
severity[pixel] = max(base severity, pasted asset grade)
pattern_masks[family][pixel] = 1 where that family is active
```

Rules:

- asset masks are clipped to `wafer_mask & valid_test_mask`;
- same-family overlap uses max intensity;
- multi-family overlap is allowed because the target is sigmoid multi-label;
- placement metadata records requested mode and actual fallback behavior.

## Placement Policy

| Mode | Use |
|---|---|
| `source_jitter` | Keep the asset near its original bbox with pixel jitter. |
| `polar_jitter` | Keep similar radial/angular wafer location using asset metadata. |
| `random_valid` | Place anywhere valid for stress-testing shape transfer. |

Prefer `polar_jitter` when process location matters. A local blob at the edge, a ring arc at a specific radius, or a shot-grid pattern is not necessarily location-free.

## Build Synthetic Dataset

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 200 `
  --assets-per-wafer 3 `
  --placement-mode polar_jitter `
  --procedural-families scratch,edge,shot_grid,random
```

Then run readiness:

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

Key outputs:

| Output | Meaning |
|---|---|
| `asset_segmentation_manifest.csv` | training/validation manifest |
| `asset_segmentation_readiness_metrics.json` | coverage, overlap, split metrics |
| `asset_segmentation_gallery.png` | visual sample check |
| `asset_segmentation_smoke.html` | input/target wiring check |
| `asset_unet_segmentation.html` | PyTorch dependency or training report |
| `pattern_asset_project_report.html` | project summary |

## Training Contract

Input channels:

```text
severity_mean
severity_max
fail_density
wafer_mask
valid_test_mask
stby_mask
x_norm
y_norm
radial_norm
angle_sin
angle_cos
edge_distance_norm
```

Target channels:

```text
local, scratch, ring, edge, shot_grid, random
```

Before training, `train_unet_segmentation.py` checks train-split positive coverage. Validation-only gaps do not block training, but those class metrics should not be trusted until validation positives exist.

## Model-Assisted Correction

```text
trained model
-> export_unet_predictions.py
-> fbm_prediction_masks/v1
-> run_segmentation_tool.py --prediction-json
-> human correction
-> updated pattern assets
-> retraining
```

Export command:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

## Validation

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py tests/test_segmentation_readiness.py tests/test_segmentation_training.py -q --basetemp .pytest_tmp_pattern_asset
```
