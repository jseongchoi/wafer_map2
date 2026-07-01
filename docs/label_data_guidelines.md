# 라벨 데이터 가이드

이 문서는 WaferMap에서 사람이 라벨을 어떻게 모으고 저장해야 하는지
설명합니다. 가장 중요한 원칙은 이것입니다.

```text
bbox_xywh = 어디를 볼지 알려주는 사각형 힌트
mask.png  = U-Net이 실제로 학습하는 정답
```

U-Net은 `bbox_xywh`를 보고 배우지 않습니다. U-Net은 최종적으로
family별 binary mask를 봅니다.

```text
local_mask[y, x] = 1 또는 0
scratch_mask[y, x] = 1 또는 0
ring_mask[y, x] = 1 또는 0
...
```

그래서 불량이 사각형이 아니어도 괜찮습니다. 오히려 그래서
segmentation을 쓰는 것입니다.

## 1. 라벨은 세 단계로 모읍니다

처음부터 모든 wafer의 모든 불량을 완벽하게 누끼 따려고 하면 오래
못 갑니다. 라벨은 세 단계로 나누어 저장합니다.

| 단계 | 사람이 하는 일 | 학습 사용 |
|---|---|---|
| Wafer-level tag | wafer 전체에 어떤 불량이 보이는지만 빠르게 체크 | 샘플링, triage |
| Rough region | bbox, lasso, sector 정도로 대략 위치 표시 | review, proposal |
| Clean mask | 학습에 쓸 명확한 불량만 pixel mask로 저장 | U-Net target, pattern asset |

애매한 불량은 억지로 `scratch`나 `ring`에 넣지 않습니다. `mixed_unknown`
또는 review-only label로 남겨두는 것이 더 낫습니다.

## 2. wafer 한 장 예시

원본 wafer image가 하나 있다고 가정합니다.

```text
data/raw/product_A/WAFER_0001.png
```

작업자가 이 wafer를 보고 이렇게 판단했다고 해봅니다.

- 왼쪽 아래에 명확한 `local` blob이 있음
- 중앙을 가로지르는 `scratch`가 있음
- 오른쪽 edge 근처에 애매한 diffuse 패턴이 있음

그러면 label package는 이렇게 저장합니다.

```text
data/labels/WAFER_0001/
  wafer.png
  labels.json
  masks/
    local.png
    scratch.png
    ring.png
    edge.png
    shot_grid.png
    random.png
    mixed_unknown.png
```

각 mask 이미지는 원본 wafer와 같은 크기입니다.

```text
검정 또는 0   = 해당 family 아님
흰색 또는 255 = 해당 family가 있는 pixel
```

현재 primary model target은 아래 여섯 개입니다.

```text
local, scratch, ring, edge, shot_grid, random
```

`mixed_unknown`은 학습 target이 아니라 review-only 보관함입니다.

## 3. labels.json 예시

```json
{
  "schema_version": "wafer_label/v1",
  "sample_id": "WAFER_0001",
  "image_path": "data/raw/product_A/WAFER_0001.png",
  "image_shape": {
    "height": 1024,
    "width": 1024
  },
  "wafer_level_tags": {
    "has_local": true,
    "has_scratch": true,
    "has_ring": false,
    "has_edge": false,
    "has_shot_grid": false,
    "has_random": false,
    "has_mixed_unknown": true
  },
  "masks": {
    "local": "masks/local.png",
    "scratch": "masks/scratch.png",
    "ring": "masks/ring.png",
    "edge": "masks/edge.png",
    "shot_grid": "masks/shot_grid.png",
    "random": "masks/random.png",
    "mixed_unknown": "masks/mixed_unknown.png"
  },
  "instances": [
    {
      "instance_id": "WAFER_0001_local_0001",
      "family": "local",
      "label_type": "manual_mask",
      "bbox_xywh": [120, 720, 80, 70],
      "mask_path": "masks/local.png",
      "label_confidence": "high",
      "mask_quality": "clean",
      "is_training_eligible": true
    },
    {
      "instance_id": "WAFER_0001_scratch_0001",
      "family": "scratch",
      "label_type": "manual_mask",
      "bbox_xywh": [300, 430, 420, 55],
      "mask_path": "masks/scratch.png",
      "label_confidence": "medium",
      "mask_quality": "usable",
      "is_training_eligible": true
    },
    {
      "instance_id": "WAFER_0001_unknown_0001",
      "family": "mixed_unknown",
      "label_type": "weak_region",
      "bbox_xywh": [760, 180, 120, 180],
      "mask_path": "masks/mixed_unknown.png",
      "label_confidence": "low",
      "mask_quality": "rough",
      "is_training_eligible": false,
      "ambiguous_reason": "edge인지 diffuse인지 명확하지 않음"
    }
  ]
}
```

## 4. bbox와 mask는 역할이 다릅니다

`bbox_xywh`는 외곽 사각형입니다.

```text
[x, y, width, height]
```

사용처는 이런 쪽입니다.

- defect 위치를 다시 찾기
- crop preview 만들기
- pattern asset 저장하기
- source wafer에서 어디서 나온 불량인지 추적하기
- synthetic wafer에 다시 붙일 때 위치와 크기 잡기

하지만 U-Net의 정답은 bbox가 아니라 mask입니다.

| 불량 형태 | bbox 역할 | mask 역할 |
|---|---|---|
| 작은 blob | blob 주변 crop | blob pixel만 1 |
| 긴 scratch | 선 전체를 감싸는 crop | 얇은 line/arc pixel만 1 |
| ring | ring 주변 crop | annulus 또는 arc band만 1 |
| edge sector | edge 근처 crop | 비정상 edge band/sector만 1 |
| shot repeat | 반복 패턴이 있는 영역 crop | 반복 die 위치만 1 |

## 5. local blob mask는 이렇게 만듭니다

작은 blob 불량은 보통 `local` family로 저장합니다. 이 유형은 처음부터 완전
자동으로 맡기기보다, 코드가 후보 mask를 만들고 사람이 마지막에 고치는 방식이
가장 현실적입니다.

```text
wafer image
-> blob 주변 bbox 선택
-> crop 안에서 자동 후보 mask 생성
-> 사람이 brush/eraser로 수정
-> 원본 wafer 크기의 full-size mask로 저장
-> 좋은 예시만 pattern asset으로 승격
```

### 5.1 bbox로 작업 영역을 자릅니다

원본 wafer가 `1024 x 1024`이고 왼쪽 아래에 blob이 있다고 가정합니다.

```json
{
  "family": "local",
  "bbox_xywh": [120, 720, 80, 70]
}
```

이 bbox는 학습 정답이 아닙니다. 작업자가 보기 편하게 blob 주변만 crop하는
힌트입니다.

```text
crop = wafer[y:y+h, x:x+w]
crop = wafer[720:790, 120:200]
```

### 5.2 crop 안에서 blob 후보를 자동으로 잡습니다

밝은 blob이면 주변 background보다 밝은 pixel을 후보로 잡습니다.

```python
crop = wafer[y:y+h, x:x+w]
threshold = crop.mean() + 2.0 * crop.std()
candidate = crop > threshold
```

어두운 blob이면 방향을 반대로 둡니다.

```python
threshold = crop.mean() - 2.0 * crop.std()
candidate = crop < threshold
```

실제 구현에서는 아래 후처리를 같이 씁니다.

```text
작은 점 제거
-> 구멍 메우기
-> morphology close/open
-> connected component 중 가장 그럴듯한 blob 선택
```

이 자동 후보는 정답이 아니라 초안입니다. wafer마다 밝기, noise, 주변 pattern이
다르기 때문에 사람이 preview를 보고 고쳐야 합니다.

### 5.3 사람이 마지막 mask를 고칩니다

작업자는 자동 후보를 보고 아래를 수정합니다.

| 상황 | 수정 |
|---|---|
| blob 일부가 빠짐 | brush로 추가 |
| 정상 background가 들어감 | eraser로 제거 |
| 작은 noise 점이 같이 잡힘 | 작은 component 삭제 |
| blob 안쪽 구멍이 생김 | fill holes |

최종 crop mask는 blob pixel만 1이어야 합니다.

```text
crop_mask[pixel] = 1   # local blob
crop_mask[pixel] = 0   # background
```

### 5.4 full-size mask로 되돌립니다

U-Net이 학습하는 mask는 crop 크기가 아니라 원본 wafer와 같은 크기입니다.

```python
full_mask = np.zeros_like(wafer, dtype=np.uint8)
full_mask[y:y+h, x:x+w] = crop_mask.astype(np.uint8) * 255
```

저장 예시:

```text
data/labels/WAFER_0001/masks/local.png
```

`local.png`는 원본 wafer와 같은 `1024 x 1024` 크기입니다.

```text
0 또는 검정      = local blob 아님
255 또는 흰색    = local blob pixel
```

### 5.5 metadata는 이렇게 남깁니다

```json
{
  "instance_id": "WAFER_0001_local_0001",
  "family": "local",
  "label_type": "manual_mask",
  "subtype": "small_isolated_blob",
  "subtype_status": "metadata_only",
  "mask_creation_method": "threshold_assisted_brush",
  "bbox_xywh": [120, 720, 80, 70],
  "mask_path": "masks/local.png",
  "label_confidence": "high",
  "mask_quality": "clean",
  "is_training_eligible": true,
  "notes": "왼쪽 아래의 명확한 local blob. 자동 후보를 brush/eraser로 수정함."
}
```

`label_type`은 `manual_mask`로 둡니다. 최종 책임은 사람이 확인한 mask에 있기
때문입니다. 자동 threshold를 썼다는 사실은 `mask_creation_method`에 남깁니다.
`subtype`은 처음에는 모델 target이 아니라 분석용 metadata입니다. subtype이
충분히 반복되고 리뷰어가 안정적으로 구분할 수 있을 때만 별도 target 승격을
검토합니다.

### 5.6 pattern asset으로 승격할지 판단합니다

승격해도 되는 blob:

```text
family가 local로 명확하다.
mask가 blob만 포함한다.
background가 과하게 들어가지 않았다.
source wafer와 bbox가 기록되어 있다.
합성 wafer에 다시 붙여도 자연스럽다.
```

승격하지 말아야 하는 blob:

```text
경계가 흐려서 diffuse defect에 가깝다.
scratch나 edge와 붙어 있어 local만 분리하기 어렵다.
threshold가 noise를 blob처럼 잡았다.
정상 background가 mask 대부분을 차지한다.
```

## 6. label_type을 꼭 나눕니다

모든 라벨이 사람이 직접 칠한 mask일 필요는 없습니다.

| `label_type` | 의미 | 학습 사용 |
|---|---|---|
| `manual_mask` | 사람이 brush/lasso로 직접 칠한 mask | clean/usable이면 사용 |
| `parametric_mask` | 사람이 규칙을 지정하고 코드가 만든 mask | preview 확인 후 사용 |
| `weak_region` | 대략적인 bbox/lasso/sector | review/proposal용 |
| `wafer_tag_only` | wafer 전체에 이런 불량이 있다는 tag | 샘플링/triage용 |

예를 들어 `shot_grid`는 사람이 모든 die를 칠하지 않습니다.

```json
{
  "family": "shot_grid",
  "label_type": "parametric_mask",
  "parameters": {
    "shot_rows": 3,
    "shot_cols": 3,
    "affected_slot": [2, 0],
    "anchor_region": "lower_left",
    "intra_die_region": {
      "x_min": 0.0,
      "x_max": 0.35,
      "y_min": 0.65,
      "y_max": 1.0
    }
  }
}
```

뜻은 이렇습니다.

```text
3x3 shot 반복 구조에서
왼쪽 아래 slot에 해당하는 die만 보고
그 die 내부의 lower-left 영역을 mask로 만든다
```

## 7. parametric_mask 예시

| Family | 사람이 지정하는 값 | 코드가 만드는 mask |
|---|---|---|
| `shot_grid` | shot rows/cols, affected slot, die 내부 영역 | 반복 위치 mask |
| `ring` | center, radius, width, angle range | 원형/부분원형 band |
| `edge` | radial range, angle sector | edge band/sector |
| `scratch` | polyline points, width | 선 주변 tube mask |
| `local` | center/radius 또는 lasso | blob mask |

`ring` 예시:

```json
{
  "family": "ring",
  "label_type": "parametric_mask",
  "parameters": {
    "center_xy": [512, 512],
    "radius": 310,
    "width": 18,
    "theta_start_deg": 20,
    "theta_end_deg": 210
  }
}
```

`scratch` 예시:

```json
{
  "family": "scratch",
  "label_type": "parametric_mask",
  "parameters": {
    "points_xy": [[180, 420], [350, 455], [620, 470]],
    "width": 8
  }
}
```

코드가 mask를 만들 때는 항상 아래 조건을 적용합니다.

```text
mask &= wafer_mask
mask &= valid_test_mask
```

필요하면 severity 조건도 씁니다.

```text
mask &= severity >= severity_threshold
```

## 8. 좋은 예시만 pattern asset으로 승격합니다

full-wafer label package는 라벨 관리용입니다. 학습과 합성에 재사용할
좋은 조각만 pattern asset으로 저장합니다.

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

예:

```text
data/pattern_assets/scratch/WAFER_0001_scratch_0001/
  grade.png
  mask.png
  preview.png
  metadata.json
```

`metadata.json` 예시:

```json
{
  "schema_version": "fbm_pattern_asset/v1",
  "family": "scratch",
  "source_sample_id": "WAFER_0001",
  "bbox_xywh": [300, 430, 420, 55],
  "mask_pixel_count": 1830,
  "mask_quality": "usable",
  "label_confidence": "medium",
  "is_training_eligible": true,
  "composition_rule": "max"
}
```

승격 기준은 단순합니다.

```text
애매함 -> 보류
대략 위치만 있음 -> review/proposal
명확하고 재사용 가능함 -> pattern asset
```

## 9. 처음 모을 목표량

처음부터 많이 모으는 것보다, 흔들리지 않는 예시를 먼저 모으는 게
좋습니다.

| Family | 1차 목표 |
|---|---:|
| `local` | 30-50 clean/usable assets |
| `scratch` | 30-50 clean/usable assets |
| `ring` | 30-50 clean/usable assets |
| `edge` | 30-50 clean/usable assets |
| `shot_grid` | 10-30 assets 또는 parametric rule 중심 |
| `random` | procedural 중심 |

추천 흐름:

```text
1. 많은 wafer에 wafer-level tag만 빠르게 저장
2. 명확한 불량만 rough region 표시
3. 학습에 쓸 것만 clean/usable mask 생성
4. 애매한 것은 mixed_unknown으로 남기고 학습 제외
5. 좋은 mask만 data/pattern_assets로 승격
6. 합성 데이터 생성 후 readiness report 확인
7. U-Net 예측을 실제 wafer에서 고쳐 다시 asset으로 저장
```

## 10. 최소 metadata 필드

각 instance에는 최소한 아래 필드를 남깁니다.

| Field | 의미 |
|---|---|
| `instance_id` | 이 결함 라벨의 고유 id |
| `family` | `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`, 또는 `mixed_unknown` |
| `label_type` | `manual_mask`, `parametric_mask`, `weak_region`, `wafer_tag_only` |
| `bbox_xywh` | source wafer 기준 외곽 사각형 |
| `mask_path` | full-size family mask 경로 |
| `label_confidence` | `high`, `medium`, `low` |
| `mask_quality` | `clean`, `usable`, `rough` |
| `is_training_eligible` | U-Net target으로 써도 되는지 |
| `ambiguous_reason` | 학습 제외 또는 low confidence일 때 이유 |

추가로 있으면 좋은 필드:

- `product_id`
- `lot_id`
- `process_step`
- `reviewer`
- `review_status`
- `source_manifest`
- `location_summary`

## 11. 판단 규칙

- 모양이 명확하면 공정 원인 추정보다 geometry 기준으로 라벨링합니다.
- family가 애매하면 `mixed_unknown`으로 보관하고 학습에서 제외합니다.
- normal background가 많이 들어간 mask는 training eligible이 아닙니다.
- wafer-wide pattern도 의미 있는 band, arc, 반복 위치만 mask로 잡습니다.
- 한 pixel이 여러 family에 해당하면 multi-label을 허용합니다.
