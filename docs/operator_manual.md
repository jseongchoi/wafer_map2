# Operator Manual

This is the short runbook for operating WaferMap. For the full command sequence and rationale, start with [End-To-End Workflow](end_to_end_workflow.md).

## Goal

Turn real or real-like wafer maps into reusable defect assets, compose those assets into multi-defect synthetic wafers, train a segmentation model, and feed model predictions back into the tool for correction.

```text
wafer manifest
-> run_segmentation_tool.py
-> data/pattern_assets
-> data/synthetic/asset_composed
-> asset_segmentation_manifest.csv
-> train_unet_segmentation.py
-> export_unet_predictions.py
-> correction loop
```

## Roles

| Role | Responsibility |
|---|---|
| Data operator | Prepare raw wafer input and manifest files. |
| Annotator | Draw family masks in `run_segmentation_tool.py`. |
| Dataset owner | Review pattern assets and synthetic dataset readiness. |
| ML owner | Train U-Net, export prediction masks, and track correction loop quality. |

## Preconditions

```powershell
pip install -e .[dev]
```

For real U-Net training:

```powershell
pip install -e .[dev,train]
```

Fast environment check:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py -q --basetemp .pytest_tmp_operator
```

## 1. Prepare Manifest

Real PNG batch:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch
```

Main output:

```text
outputs/manifests/real_png_batch_manifest.json
```

Quick smoke input:

```text
configs/eval/real_unlabeled_synthetic_smoke.json
```

## 2. Open Segmentation Tool

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

For model-assisted correction:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --prediction-json outputs/predictions/fbm_prediction_masks.json `
  --assets-root data/pattern_assets
```

## 3. Annotation Rules

| Family | Rule |
|---|---|
| `local` | Compact blob or local cluster only. Do not include broad normal background. |
| `scratch` | Visible line, arc, or scratch-like defect. Split unrelated blobs into `local`. |
| `ring` | Annulus, ring, or arc band. Do not fill the whole disk. |
| `edge` | Abnormal edge band or edge sector, not the normal wafer boundary. |
| `shot_grid` | Repeated shot-relative signature. Single blobs should usually be `local`. |
| `random` | Sparse unstructured fail pattern when no stable geometry fits. |

Large wafer-wide patterns are still segmentation targets. Use geometry to decide `edge`, `ring`, `shot_grid`, or `random`.

## 4. Save And Review Assets

The tool saves:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

Review:

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

Reject or correct assets with loose masks, wrong family assignment, accidental ring splitting, STBY-only pixels, or broad normal edge labeling.

## 5. Build Synthetic Dataset

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

Must inspect:

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
outputs/pattern_asset_pipeline/asset_segmentation_readiness.html
outputs/pattern_asset_pipeline/asset_segmentation_readiness_metrics.json
outputs/reports/pattern_asset_project_report.html
```

## 6. Train And Export

Train:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

Export predictions:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

Training blocks when the train split lacks positive examples for a target family unless `--allow-incomplete-target-coverage` is explicitly used for wiring/debug.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Saved asset count is 0 | No mask pixels were selected | Check active family and output pixel counts. |
| Synthetic sample has no assets | `data/pattern_assets` is empty or invalid | Run the tool and asset report first. |
| Edge/ring labels are too broad | Mask rule is too loose | Relabel only the visible abnormal band or arc. |
| Prediction masks do not load | Schema or sample id mismatch | Check `fbm_prediction_masks/v1` and `sample_id`. |
| U-Net training refuses to start | Missing positive train samples | Add assets or rebuild the synthetic dataset. |
| Tests fail after docs edit | Documentation quality test found drift | Check links and current project direction. |

## Release Checklist

Before committing:

```powershell
python -m compileall scripts src -q
python -m pytest -q --basetemp .pytest_tmp_release
git status --short
```

Also check:

- [End-To-End Workflow](end_to_end_workflow.md) still contains the main command sequence.
- [scripts command map](../scripts/README.md) lists any new operator-facing command.
- No generated wafer data, assets, checkpoints, or reports are staged.
