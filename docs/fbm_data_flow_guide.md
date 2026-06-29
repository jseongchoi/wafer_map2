# FBM Data Flow Guide

이 문서는 raw wafer 입력부터 직접 segmentation mask, pattern asset, hybrid synthetic data, segmentation manifest까지 어떤 파일이 어디에 생기는지 정리합니다.

## End-To-End Flow

```text
raw wafer PNG or synthetic sample dir
-> real_unlabeled_manifest/v1
-> local segmentation tool
-> data/pattern_assets
-> data/synthetic/asset_composed
-> outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
-> coordinate-aware small U-Net training input
```

## Folder Contract

| Stage | Path | Git policy |
|---|---|---|
| Raw wafer input | `data/raw/<product>/*.png` | ignored |
| Intermediate arrays | `data/interim/` | ignored |
| Processed local data | `data/processed/` | ignored |
| Pattern assets | `data/pattern_assets/<family>/<asset_id>/` | ignored |
| Synthetic samples | `data/synthetic/<run>/<sample_id>/` | ignored |
| Manifests/reports | `outputs/` | ignored |
| Source/config/docs/tests | `src/`, `scripts/`, `configs/`, `docs/`, `tests/` | tracked |

Git에는 code, schema, config, docs, tests만 넣습니다. wafer 원본, annotation 산출물, pattern asset, 합성 sample, model checkpoint는 로컬/운영 산출물입니다.

## 1. Real/Unlabeled Manifest

실제 PNG batch를 manifest로 변환합니다.

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

생성 위치:

```text
outputs/manifests/real_png_batch_manifest.json
outputs/reports/real_png_batch/
```

빠르게 manifest만 만들 때:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --manifest-only `
  --out-dir outputs/reports/local_raw_batch
```

## 2. Segmentation Tool

manifest에서 샘플 하나를 열고 family별 mask를 작성합니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

생성 구조:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

핵심 metadata:

```json
{
  "schema_version": "fbm_pattern_asset/v1",
  "family": "local",
  "source_sample_id": "wafer_001",
  "bbox_xywh": [120, 80, 48, 32],
  "mask_pixel_count": 220,
  "composition_rule": "max"
}
```

## 3. Pattern Asset QC

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

## 4. Hybrid Synthetic Data

pattern asset과 procedural fallback을 base wafer에 합성합니다.

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 200 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

각 sample:

```text
data/synthetic/asset_composed/<sample_id>/
  arrays.npz
  metadata.json
```

중요 배열:

| Array | Meaning |
|---|---|
| `severity` | final grade 0-7 wafer map |
| `wafer_mask` | wafer valid area |
| `valid_test_mask` | testable chip/pixel area |
| `stby_mask` | missing-test/STBY area |
| `pattern_masks` | family-wise multi-label target |
| `pattern_intensity` | family-wise normalized intensity |
| `chip_index` | chip coordinate index |

## 5. Segmentation Readiness

합성 sample을 학습 manifest와 validation report로 변환합니다.

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

주요 산출물:

```text
outputs/pattern_asset_pipeline/
  asset_segmentation_manifest.csv
  asset_segmentation_readiness.html
  asset_segmentation_readiness_metrics.json
  asset_segmentation_gallery.png
  asset_segmentation_smoke.html
  asset_embedding_smoke.html
  asset_unet_segmentation.html
  asset_unet_segmentation_metrics.json

outputs/reports/pattern_asset_project_report.html
```

`asset_segmentation_manifest.csv`가 `train_unet_segmentation.py`의 직접 입력입니다.

## 6. Model Training Entry Point

PyTorch 설치 환경:

```powershell
pip install -e .[train]
```

학습:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

현재 기본 목표는 coordinate-aware small U-Net입니다. 입력에는 grade map뿐 아니라 wafer mask, valid-test mask, stby mask, x/y/radial/angle/edge-distance channel이 포함됩니다.
