# Project Overview

WaferMap is a wafer-specific segmentation dataset factory. The current goal is to make reliable training data for defect segmentation, not to optimize model architecture first.

Canonical workflow: [End-To-End Workflow](end_to_end_workflow.md).

## Current Product Shape

```text
raw or real-like wafer samples
-> real_unlabeled_manifest/v1
-> in-repo segmentation tool
-> pattern asset library
-> hybrid synthetic data
-> segmentation readiness / smoke validation
-> train_unet_segmentation.py
-> export_unet_predictions.py
-> correction loop
```

## Main Components

| Component | Responsibility |
|---|---|
| `run_segmentation_tool.py` | Operator-facing local segmentation tool. |
| `run_pattern_asset_editor.py` | Compatibility engine filename retained for existing commands/tests. |
| `build_pattern_asset_report.py` | Review saved pattern assets. |
| `compose_synthetic_from_assets.py` | Paste pattern assets and procedural fallback onto base wafers. |
| `run_pattern_asset_pipeline.py` | Compose, validate readiness, run smoke checks, and generate reports. |
| `train_unet_segmentation.py` | Train or dependency-check the coordinate-aware small U-Net. |
| `export_unet_predictions.py` | Export trained masks to `fbm_prediction_masks/v1` for correction. |

## Defect Families

| Family | Current source | Notes |
|---|---|---|
| `local` | human asset primary | blob or compact cluster |
| `scratch` | human asset primary plus procedural fallback | line, arc, scratch-like path |
| `ring` | human asset primary | full ring, partial ring, annulus, arc |
| `edge` | procedural primary plus optional human asset | abnormal edge band or sector |
| `shot_grid` | procedural primary plus optional human asset | shot-relative repeated defect |
| `random` | procedural only | sparse unstructured fail baseline |

`stby_pattern` describes missing-test context. It is not a primary defect target.

## What Stays

- `src/wafermap/data`, `src/wafermap/real`, `src/wafermap/synth`, `src/wafermap/assets`, `src/wafermap/training`;
- local segmentation tool and pattern asset scripts;
- synthetic composition scripts;
- readiness, smoke, U-Net, and prediction export entrypoints;
- report generation and validation tests.

## Compatibility

`scripts/run_pattern_asset_editor.py` remains as a compatibility filename. New operator-facing documentation and commands should use `scripts/run_segmentation_tool.py`.

## Current Validation State

- Pattern asset pipeline tests pass.
- Segmentation readiness and smoke tests pass.
- PyTorch-free environments can run `train_unet_segmentation.py --check-deps`.
- Prediction export schema round-trips through the segmentation tool loader.

## Next Implementation Order

1. Build enough real `local`, `scratch`, `ring`, and `edge` assets from wafer manifests.
2. Review assets for tight masks, correct family assignment, split/merge issues, and location metadata.
3. Compose hybrid synthetic data and inspect readiness reports.
4. Train `train_unet_segmentation.py` in a PyTorch environment.
5. Export predictions with `export_unet_predictions.py`.
6. Reload predictions in the segmentation tool, correct them, and save improved assets.

## Practical Definition Of Done

- Annotators can choose families consistently.
- Edge, ring, and other global patterns are segmented by geometry, not ignored.
- Saved assets are inspectable through preview/mask/metadata.
- Synthetic `pattern_masks` can be used directly as U-Net targets.
- Readiness reports expose family coverage and overlap problems before training.
