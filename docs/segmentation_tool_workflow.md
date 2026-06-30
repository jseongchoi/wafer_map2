# 세그멘테이션 도구 흐름

이 문서는 `run_segmentation_tool.py`를 열었을 때 어떤 기준으로 mask를 만들고
asset을 저장해야 하는지 설명합니다.

## 1. 도구를 여는 명령

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

모델 예측을 먼저 깔아놓고 수정하려면:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets `
  --prediction-json outputs/predictions/fbm_prediction_masks.json
```

## 2. 도구에서 하는 일

작업자는 wafer image를 보고 family별 mask를 만듭니다.

```text
wafer image
-> family 선택
-> mask 또는 rule 생성
-> preview 확인
-> pattern asset 저장
```

저장된 asset은 나중에 합성 wafer를 만들 때 재사용됩니다.

## 3. Family 선택 기준

| Family | 도구에서 찾는 모습 | 좋은 저장 방식 |
|---|---|---|
| `local` | 작고 뚜렷한 blob | brush/lasso mask |
| `scratch` | 긴 선, 곡선, 대각선 | polyline + width 또는 mask |
| `ring` | 원형/호형 band | center/radius/width rule |
| `edge` | edge 근처 sector/band | angle range + radial width |
| `shot_grid` | shot마다 같은 상대 위치 반복 | shot layout + affected slot |
| `random` | 산발 fail | 학습 baseline용 sparse mask |

## 4. bbox와 mask의 차이

도구에서 bbox가 필요할 수 있지만, bbox는 학습 정답이 아닙니다.

```json
{
  "bbox_xywh": [120, 720, 80, 70],
  "mask_path": "masks/local.png"
}
```

- `bbox_xywh`: 사람이 review하거나 crop할 때 쓰는 위치 힌트
- `mask_path`: U-Net이 실제로 배우는 정답

불량이 사각형이 아니라면 bbox 안의 실제 불량 pixel만 mask로 칠하면 됩니다.

## 5. Parametric label이 필요한 경우

아래 유형은 손으로 전부 칠하지 않는 것이 좋습니다.

### shot_grid 예시

```json
{
  "family": "shot_grid",
  "label_type": "parametric_mask",
  "params": {
    "shot_rows": 3,
    "shot_cols": 3,
    "affected_slot": [2, 0],
    "region": "lower_left",
    "region_fraction": [0.0, 0.0, 0.35, 0.35]
  }
}
```

뜻:

```text
3x3 shot layout에서
각 shot의 왼쪽 아래 die 영역이 반복적으로 불량이다.
```

코드는 이 rule을 full-size binary mask로 바꿉니다.

### ring 예시

```json
{
  "family": "ring",
  "label_type": "parametric_mask",
  "params": {
    "center_xy": [512, 512],
    "radius": 310,
    "width": 24,
    "theta_range_deg": [0, 360]
  }
}
```

### scratch 예시

```json
{
  "family": "scratch",
  "label_type": "parametric_mask",
  "params": {
    "polyline_xy": [[180, 650], [340, 620], [620, 590], [810, 540]],
    "width": 18
  }
}
```

## 6. 저장하면 안 되는 경우

아래 상황에서는 학습 target으로 저장하지 않습니다.

- 경계가 너무 흐려서 family가 불명확함
- 여러 family가 섞였는데 분리하기 어려움
- sensor/전처리 artifact처럼 실제 defect인지 모름
- wafer 밖 영역이 mask에 많이 포함됨

이런 경우는 `mixed_unknown` 또는 review-only 기록으로 남깁니다.

## 7. 좋은 asset의 기준

좋은 asset:

```text
family가 명확하다.
mask가 원본 불량 위치와 맞다.
wafer 밖을 포함하지 않는다.
metadata에 source wafer와 label_type이 남아 있다.
합성에 재사용해도 자연스럽다.
```

나쁜 asset:

```text
왜 이 family인지 설명이 안 된다.
bbox만 있고 mask가 없다.
불량보다 정상 영역이 훨씬 많이 들어 있다.
애매한 diffuse를 억지로 scratch라고 저장했다.
```

## 8. Prediction prefill 사용법

모델이 만든 예측은 작업을 줄이기 위한 초안입니다.
그대로 정답으로 믿으면 안 됩니다.

예측 JSON은 보통 `export_unet_predictions.py`로 만듭니다.

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json
```

작업 순서:

1. prediction mask를 불러옵니다.
2. family가 틀린 영역을 지웁니다.
3. 빠진 명확한 영역을 추가합니다.
4. 수정된 mask를 새 asset 또는 correction data로 저장합니다.

이 loop가 쌓이면 synthetic-only 모델에서 실제 wafer에 더 맞는 모델로 이동할 수 있습니다.

## 9. 도구 품질 체크

도구를 수정한 뒤 확인할 것:

- wafer image가 정상적으로 표시되는가?
- brush/lasso가 mask를 실제로 바꾸는가?
- family별 mask가 저장되는가?
- `--prediction-json`을 넣으면 예측이 prefill되는가?
- 저장된 asset을 report에서 다시 볼 수 있는가?
