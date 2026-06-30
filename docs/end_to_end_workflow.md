# End-To-End Workflow

This is the canonical operating path for the project.

```text
FBM maps
-> defect generation
-> multi-defect synthetic maps
-> multi-defect segmentation training and validation
-> local segmentation tool for real-data pattern asset extraction
-> corrected assets and retraining
```

The target is not wafer-level classification. The target is a pixel-level, multi-label defect segmentation dataset that can train a U-Net-style model.

## 1. Source Data

Use a real wafer PNG batch or a synthetic smoke manifest.

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch
```

Smoke manifest for quick local checks:

```text
configs/eval/real_unlabeled_synthetic_smoke.json
```

## 2. Extract Pattern Assets

Open one wafer sample in the in-repo segmentation tool and save family masks as reusable assets.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

Saved asset contract:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

Each asset stores its source wafer, family, crop box, mask pixel count, grade range, and location summary. The location summary matters because many wafer defects are not location-free; radial distance, angular sector, and edge distance can be part of the process signature.

## 3. Defect Families

| Family | Use | Main source |
|---|---|---|
| `local` | compact blob or cluster | human asset primary |
| `scratch` | line, arc, scratch-like defect | human asset primary plus procedural fallback |
| `ring` | full ring, partial ring, annulus, arc | human asset primary |
| `edge` | abnormal edge band or edge sector | procedural primary plus optional human asset |
| `shot_grid` | repeated shot-relative pattern | procedural primary plus optional human asset |
| `random` | sparse unstructured baseline fail pattern | procedural only |

Large wafer-wide patterns should still become segmentation masks, but the family depends on geometry:

- edge-wide abnormality: label abnormal edge band or sector as `edge`, not the whole wafer boundary;
- circular or arc-shaped process signature: label as `ring`;
- repeated die/shot-relative signature: label as `shot_grid`;
- broad mixed-area failure with no stable geometry: keep as `random` until a stronger taxonomy is justified.

## 4. Review Assets

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

Reject or correct assets when the mask includes normal background, the family is wrong, a ring is split unintentionally, or an edge mask describes the normal wafer boundary instead of the abnormal region.

## 5. Compose Synthetic Maps

Paste real-data pattern assets onto base wafers and fill missing families with procedural fallback.

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

Composition rule:

```text
severity[pixel] = max(base severity, pasted asset grade)
pattern_masks[family][pixel] = 1 where that family is active
```

Use `polar_jitter` when process location is meaningful. Use `random_valid` only when testing shape transfer independent of location.

## 6. Validate Training Readiness

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

Key outputs:

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
outputs/pattern_asset_pipeline/asset_segmentation_readiness.html
outputs/pattern_asset_pipeline/asset_segmentation_readiness_metrics.json
outputs/pattern_asset_pipeline/asset_segmentation_gallery.png
outputs/reports/pattern_asset_project_report.html
```

Readiness must show enough positive samples per family, valid-test target pixels, reasonable family overlap, and visually plausible synthetic wafers.

## 7. Train Segmentation Model

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

The model input uses:

```text
severity_mean, severity_max, fail_density,
wafer_mask, valid_test_mask, stby_mask,
x_norm, y_norm, radial_norm, angle_sin, angle_cos, edge_distance_norm
```

Resize policy:

- target masks: clip to `wafer_mask & valid_test_mask`, then max-pool so small local/scratch defects survive;
- severity input: keep `severity_mean`, `severity_max`, and `fail_density` together;
- support masks: max-pool;
- coordinate channels: mean-pool.

This is why "input mean only" is too weak, and "input max only" can exaggerate one high-grade die. The model gets both mean context and max/high-grade presence, plus fail density.

## 8. Export Predictions For Correction

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

Load the exported `fbm_prediction_masks/v1` file back into the tool:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --prediction-json outputs/predictions/fbm_prediction_masks.json `
  --assets-root data/pattern_assets
```

Human correction creates better assets, and better assets create better synthetic segmentation data.

## 9. Test Policy

Fast default tests skip slow end-to-end work:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

Run slow tests only when validating the full pipeline:

```powershell
python -m pytest -q --run-slow --basetemp .pytest_tmp_full
```

Targeted checks while editing this workflow:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py tests/test_segmentation_training.py tests/test_documentation_quality.py -q --basetemp .pytest_tmp_docs
```
