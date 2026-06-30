# FBM 데이터 흐름 가이드

이 문서는 raw wafer, label, asset, synthetic sample, model artifact가 어디에 저장되고
어떻게 이어지는지 설명합니다.

## 1. 전체 데이터 경로

```text
data/raw/
-> outputs/manifests/
-> data/pattern_assets/
-> data/synthetic/asset_composed/
-> outputs/pattern_asset_pipeline/
-> outputs/models/
-> outputs/predictions/
```

각 단계의 의미:

| 단계 | 입력 | 출력 |
|---|---|---|
| raw 분석 | PNG 폴더 | manifest JSON |
| 수동/규칙 라벨 | wafer image | pattern asset |
| 합성 | base wafer + asset | `arrays.npz`, `metadata.json` |
| readiness | synthetic samples | manifest CSV, report |
| 학습 | manifest CSV | model checkpoint |
| 예측 export | model + manifest | prediction JSON |

## 2. raw wafer 입력

예시 폴더:

```text
data/raw/product_A/
  lot_001/
    WAFER_0001.png
    WAFER_0002.png
```

manifest 생성:

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/product_A_manifest.json
```

manifest에는 아래 정보가 들어가야 합니다.

```json
{
  "sample_id": "WAFER_0001",
  "image_path": "data/raw/product_A/lot_001/WAFER_0001.png",
  "image_shape": [1024, 1024],
  "product_id": "product_A"
}
```

## 3. label package

사람이 wafer 하나를 보고 작업한 원천 label은 아래처럼 저장할 수 있습니다.

```text
data/labels/WAFER_0001/
  wafer.png
  labels.json
  masks/
    local.png
    scratch.png
    mixed_unknown.png
```

이 label package는 사람이 이해하기 위한 원천 기록입니다.
학습에 바로 들어가는 최종 형태는 `arrays.npz`의 `pattern_masks`입니다.

## 4. pattern asset

학습에 쓸 만한 clean mask는 pattern asset으로 승격합니다.

```text
data/pattern_assets/
  local/
    local_000001/
      image.png
      mask.png
      metadata.json
  scratch/
    scratch_000001/
      image.png
      mask.png
      metadata.json
```

metadata 예시:

```json
{
  "asset_id": "local_000001",
  "family": "local",
  "source_sample_id": "WAFER_0001",
  "label_type": "manual_mask",
  "bbox_xywh": [120, 720, 80, 70],
  "quality": "usable"
}
```

## 5. synthetic sample

합성 sample은 U-Net 학습이 읽는 최종 단위입니다.

```text
data/synthetic/asset_composed/
  asset_composed_000001/
    arrays.npz
    metadata.json
    preview.png
```

`arrays.npz`의 필수 key:

| Key | 의미 |
|---|---|
| `severity` | 최종 wafer fail/severity map |
| `wafer_mask` | wafer 내부 영역 |
| `valid_test_mask` | 실제 측정 가능한 영역 |
| `stby_mask` | STBY/missing-test 영역 |
| `chip_index` | die/chip id |
| `pattern_masks` | family별 binary target mask |
| `pattern_intensity` | family별 defect intensity |

## 6. training manifest

readiness 단계에서 학습용 CSV를 만듭니다.

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
```

대표 column:

```text
sample_id
split
arrays_path
metadata_path
input_channels
target_channels
has_local
local_mask_ratio
has_scratch
scratch_mask_ratio
...
```

이 manifest가 `train_unet_segmentation.py`의 직접 입력입니다.

## 7. model artifact

학습 결과는 `outputs/models/` 아래에 둡니다.

```text
outputs/models/
  asset_unet_segmentation.pt
```

모델 artifact는 git에 넣지 않습니다. 재현에 필요한 것은 코드, config, manifest,
문서화된 command입니다.

## 8. prediction output

예측 결과는 tool에서 다시 고칠 수 있게 JSON 또는 mask bundle로 저장합니다.

```text
outputs/predictions/
  fbm_prediction_masks.json
```

schema marker 예시:

```json
{
  "schema_version": "fbm_prediction_masks/v1",
  "model_path": "outputs/models/asset_unet_segmentation.pt",
  "samples": []
}
```

## 9. 좌표 기준

모든 mask와 array는 같은 좌표계를 사용합니다.

```text
y = row index
x = column index
shape = [H, W]
bbox_xywh = [x, y, width, height]
```

좌표가 어긋나면 합성 preview는 그럴듯해 보여도 U-Net target이 틀어집니다.
따라서 resize/crop을 할 때는 반드시 mask도 같은 변환을 적용해야 합니다.

## 10. 코드 위치

| 데이터 단계 | 주요 코드 |
|---|---|
| raw PNG manifest | `scripts/analyze_png_raw_folders.py`, `src/wafermap/real/` |
| pattern asset | `src/wafermap/assets/` |
| synthetic sample | `scripts/compose_synthetic_from_assets.py`, `src/wafermap/synth/` |
| readiness manifest | `scripts/build_segmentation_readiness.py` |
| coordinate-aware small U-Net | `src/wafermap/training/segmentation.py` |
| prediction export | `scripts/export_unet_predictions.py` |

자세한 학습 규격은 [학습 데이터 규격](training_data_contract.md)을 봅니다.
