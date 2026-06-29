# Segmentation Tool Workflow

This workflow uses the in-repo browser tool as the primary annotation surface for wafer defect masks.

```text
real/unlabeled wafer manifest
-> local segmentation tool
-> reusable pattern assets
-> synthetic segmentation dataset composition
-> segmentation readiness / smoke validation
```

The core direction is:

1. Generate defect masks from FBM maps.
2. Compose multi-defect synthetic maps from reusable assets.
3. Train and validate multi-defect segmentation on those composed maps.
4. Use this tool to extract example masks from real or real-like wafers so the synthetic asset library reflects real data.

The browser app is only the local mask editing surface. It does not train the segmentation model; it turns a wafer sample plus optional model proposals into reusable `mask.png` / `grade.png` / `metadata.json` pattern assets.

## 1. Open A Wafer In The Tool

```powershell
python scripts/run_segmentation_tool.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

For a real PNG batch, use the manifest written by `scripts/analyze_png_raw_folders.py`:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

The tool edits multi-label masks for `local`, `scratch`, `ring`, `edge`, `shot_grid`, and `random`. It can load model prediction masks through `--prediction-json` and proposal masks through `--proposal-json`.

After U-Net training, export model masks before opening the correction tool:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

Then pass the exported JSON to `run_segmentation_tool.py --prediction-json` for human correction and asset saving.

## 2. Edit Masks

Main controls:

- `Family`: selects which defect family receives new mask pixels.
- `Paint` / `Erase`: manually add or remove pixels.
- `Pan`, `Zoom -`, `Zoom +`, `Fit View`: inspect large maps without changing masks.
- `Map Colors`: switch the base wafer color scheme while preserving the same mask data.
- `Grow Seed`: expands painted seed pixels through connected valid wafer pixels above `Min Grade`.
- `Grade Area`: adds every valid wafer pixel above `Min Grade` to the active family.
- `Lasso Fit`: draws a rough loop and fills matching valid high-grade pixels inside it.
- `Trace Line`: extends a scratch-like line from at least two painted seed pixels.
- `Analyze` / `Load Model`: loads geometry or model proposals that can be previewed and applied.
- `Geometry Fit`: creates edge or ring proposals from radius, width, and angle controls.

Before saving, check `Output` for per-family pixel counts. The app keeps masks multi-label, so a wafer pixel can belong to more than one family when the real defect is ambiguous.

## 3. Save Pattern Assets

Saved assets use the same reusable contract everywhere in the pipeline:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

`metadata.json` records the source wafer, family, crop box, mask pixel count, grade range, composition rule, and location summary. The location summary stores radial range, angular span, edge distance, and wafer-area ratio so later composition can place real-data assets near comparable wafer zones instead of treating every cutout as location-free.

Use `Save Mode` as follows:

- `One Family Asset`: saves all pixels in a family as one asset. Use this for rings or continuous patterns that may have small gaps.
- `Split Components`: saves disconnected blobs separately. Use this when separate local defects should become separate reusable assets.

## 4. Review Saved Assets

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

Review for tight masks, correct family assignment, ring continuity, and edge masks that describe only abnormal regions.

## 5. Compose Synthetic Data

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --placement-mode polar_jitter `
  --procedural-families scratch,edge,shot_grid,random
```

Placement modes:

- `source_jitter`: place near the original source bbox with pixel jitter;
- `polar_jitter`: place in a similar radial/angular wafer zone using the asset location summary;
- `random_valid`: place anywhere valid when stress-testing shape transfer rather than process location.

## 6. Validate Segmentation Readiness

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
outputs/reports/pattern_asset_project_report.html
```

## 7. Training Resize Policy

For U-Net training, use synthetic `pattern_masks` as the multi-label target. The resizing policy should preserve tiny defects without exaggerating every input pixel:

- target masks: clip to `wafer_mask & valid_test_mask`, then bin/max pool to keep thin scratches and small local defects visible;
- binary support masks (`wafer_mask`, `valid_test_mask`, `stby_mask`): bin/max pooling;
- severity input: `severity_mean`, `severity_max`, and `fail_density`;
- position input: bin/mean pooling for `x`, `y`, radial, angle, and edge-distance channels;
- production training candidate: `--output-size 256`;
- fast smoke tests: keep smaller sizes such as 96 or 128.

Avoid a blind resize-only path for labels. A 4x4 stride/max approach is fine when the source is exactly 1024x1024, but direct bin pooling to 256x256 is safer for 2000x1900 or other non-square wafer maps.

## 8. App Quality Checks

Run the fast app and asset tests after changing the editor:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py -q
```

These tests cover pattern asset saving, editor sample payloads, editor save payloads, prediction mask prefill, model proposal loading, canvas downsampling, automatic geometry proposals, and the end-to-end pattern asset pipeline. The default pytest command skips slow end-to-end tests; add `--run-slow` only when you need the full composition and training smoke path.

Manual browser smoke check:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root .pytest_tmp_tool_review_assets `
  --port 8777 `
  --no-browser
```

Then open `http://127.0.0.1:8777/` and verify:

- the canvas renders a wafer map;
- family buttons appear for all six target families;
- `Analyze` returns proposals on high-grade samples;
- `Save Assets` writes assets under the selected root;
- `Open Report` shows the saved asset cards.
