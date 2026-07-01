# 불량 심각도 수치화 가이드

이 문서는 defect mask가 생긴 뒤 “이 불량이 얼마나 심각한가”를 어떻게 수치화할지
설명합니다. 목표는 단순히 불량 유무를 표시하는 것이 아니라, 리뷰 우선순위와
유사맵 검색, 모델 correction loop에 쓸 수 있는 해석 가능한 severity score를
만드는 것입니다.

핵심 원칙:

```text
mask가 먼저다
-> mask feature를 뽑는다
-> component score를 계산한다
-> family별 weight로 final severity score를 만든다
-> 원본 feature와 score를 모두 저장한다
```

처음부터 ML로 severity를 예측하지 않습니다. 먼저 사람이 이해할 수 있는
rule-based score로 시작하고, 나중에 yield impact나 전문가 severity label이
쌓이면 supervised severity model을 검토합니다.

## 1. confidence와 severity는 다릅니다

가장 먼저 이 둘을 분리해야 합니다.

| 값 | 의미 | 예시 |
|---|---|---|
| `confidence_score` | 이 mask/label/model prediction을 얼마나 믿는가 | manual clean mask는 높음 |
| `severity_score` | 이 defect가 공정/수율 관점에서 얼마나 심각한가 | 큰 scratch는 높음 |

예시:

```text
confidence high, severity low:
  아주 선명하지만 작은 local blob

confidence low, severity high:
  모델은 애매해하지만 실제로는 큰 diffuse defect 가능성
```

둘을 섞으면 안 됩니다. confidence는 score의 보조 component로 쓸 수 있지만,
defect 자체의 심각도와 동일한 값이 아닙니다.

## 2. 공통 severity component

대부분의 family에 공통으로 쓸 수 있는 component는 아래입니다.

| Component | 의미 | 예시 |
|---|---|---|
| `area_score` | defect mask 면적이 얼마나 큰가 | blob area, ring coverage |
| `intensity_score` | mask 내부 fail grade가 얼마나 강한가 | 평균/최대 grade |
| `location_score` | 중요한 위치에 있는가 | center, edge, shot critical region |
| `spread_score` | 얼마나 퍼져 있는가 | component 수, bbox 크기 |
| `repeat_score` | 반복 구조가 있는가 | shot마다 반복 |
| `confidence_score` | 라벨/모델 신뢰도 | manual high, prediction low |
| `rarity_score` | reference 대비 드문 패턴인가 | rare subtype |
| `trend_score` | 최근 batch에서 증가 중인가 | lot별 증가 |

초기 MVP는 아래 다섯 개면 충분합니다.

```text
area_score
intensity_score
location_score
spread_score
confidence_score
```

## 3. raw feature와 score를 모두 저장합니다

score만 저장하면 나중에 왜 심각한지 알 수 없습니다.

나쁜 예:

```json
{
  "final_severity_score": 0.82
}
```

좋은 예:

```json
{
  "family": "local",
  "area_px": 820,
  "valid_area_px": 780000,
  "component_count": 4,
  "mean_grade": 5.8,
  "max_grade": 7,
  "radial_norm": 0.72,
  "edge_distance_norm": 0.28,
  "area_score": 0.36,
  "intensity_score": 0.83,
  "spread_score": 0.44,
  "location_score": 0.30,
  "confidence_score": 0.91,
  "final_severity_score": 0.52
}
```

원본 feature를 같이 저장하면 나중에 weight를 바꿔도 다시 계산할 수 있습니다.

## 4. family별 심각도 기준

같은 면적이라도 family마다 심각도 의미가 다릅니다.

### local/blob

blob은 크기, 진하기, 개수, 위치가 중요합니다.

```text
area_score        = blob 면적 / valid wafer area
intensity_score   = blob 내부 평균/최대 grade
spread_score      = component_count 또는 bbox size
location_score    = critical zone 여부
```

예시:

```json
{
  "family": "local",
  "subtype": "cluster_blob",
  "area_px": 820,
  "component_count": 4,
  "mean_grade": 5.8,
  "radial_norm": 0.72,
  "area_score": 0.36,
  "intensity_score": 0.83,
  "spread_score": 0.44,
  "location_score": 0.30,
  "final_severity_score": 0.52
}
```

### scratch

scratch는 면적보다 길이, 폭, 방향, die를 가로지르는 정도가 중요할 수 있습니다.

```text
length_score
width_score
cross_die_score
intensity_score
location_score
```

짧은 scratch 1개보다 wafer를 길게 가로지르는 scratch가 더 심각할 가능성이 큽니다.

### ring

ring은 coverage와 radius가 중요합니다.

```text
radius_score
coverage_score
width_score
intensity_score
center_offset_score
```

예시:

```text
partial ring 20도 < ring 280도
얇은 ring < 넓고 진한 ring
```

### edge

edge는 edge coverage와 안쪽 침투 정도가 중요합니다.

```text
edge_coverage_score
radial_depth_score
sector_width_score
intensity_score
```

### shot_grid

shot_grid는 반복성이 핵심입니다.

```text
repeat_score
affected_shot_ratio
slot_consistency_score
intensity_score
location_score
```

한두 pixel보다 “shot마다 같은 위치에서 반복적으로 발생”하는 것이 더 강한 공정 신호입니다.

## 5. final severity score 계산

처음에는 family별 weighted sum으로 시작합니다.

```text
final_severity =
  0.30 * area_score +
  0.30 * intensity_score +
  0.20 * location_score +
  0.10 * spread_score +
  0.10 * confidence_score
```

family별 weight는 다르게 둡니다.

local 예시:

```json
{
  "family": "local",
  "severity_weights": {
    "area_score": 0.30,
    "intensity_score": 0.35,
    "location_score": 0.15,
    "spread_score": 0.10,
    "confidence_score": 0.10
  }
}
```

shot_grid 예시:

```json
{
  "family": "shot_grid",
  "severity_weights": {
    "repeat_score": 0.40,
    "affected_shot_ratio": 0.25,
    "intensity_score": 0.20,
    "location_score": 0.10,
    "confidence_score": 0.05
  }
}
```

## 6. severity bucket

운영에서는 score 하나보다 bucket이 더 읽기 쉽습니다.

```text
0.0 <= score < 0.3  -> low
0.3 <= score < 0.7  -> medium
0.7 <= score <= 1.0 -> high
```

저장 예시:

```json
{
  "final_severity_score": 0.52,
  "severity_bucket": "medium"
}
```

## 7. local/blob MVP schema

blob부터 시작한다면 아래 정도면 충분합니다.

```json
{
  "schema_version": "defect_severity/v1",
  "sample_id": "WAFER_0001",
  "instance_id": "WAFER_0001_local_0001",
  "family": "local",
  "subtype": "small_isolated_blob",
  "source": "manual_mask",
  "raw_features": {
    "area_px": 820,
    "valid_area_px": 780000,
    "component_count": 1,
    "bbox_xywh": [120, 720, 80, 70],
    "mean_grade": 5.8,
    "max_grade": 7,
    "radial_norm": 0.72,
    "edge_distance_norm": 0.28
  },
  "component_scores": {
    "area_score": 0.36,
    "intensity_score": 0.83,
    "location_score": 0.30,
    "spread_score": 0.18,
    "confidence_score": 0.95
  },
  "final_severity_score": 0.50,
  "severity_bucket": "medium"
}
```

## 8. 파이프라인 위치

severity scoring은 segmentation 후처리 단계에 붙입니다.

```text
mask 생성
-> mask feature 추출
-> severity component 계산
-> final severity score 계산
-> review/report 표시
-> 유사맵 검색 feature로 사용
```

입력:

```text
pattern_masks
severity map
wafer_mask
valid_test_mask
geometry / coordinate channels
label confidence 또는 model probability
```

출력:

```text
defect_severity/v1 JSON
report table
유사맵 검색용 feature
review 우선순위
```

## 9. 유사맵 검색과의 연결

severity score는 유사맵 검색에서 filter나 rerank feature로 쓸 수 있습니다.

예시:

```text
query = high severity local cluster blob
candidate 검색:
  family = local
  subtype = cluster_blob
  severity_bucket = high 또는 medium/high
  blob area와 위치가 비슷함
```

단, severity만으로 유사도를 판단하지 않습니다.

```text
비슷한 severity라도 위치와 shape가 다르면 다른 defect일 수 있다.
비슷한 shape라도 severity bucket이 다르면 리뷰 우선순위가 다를 수 있다.
```

## 10. 판단 규칙

- severity scoring은 segmentation mask 이후에 계산합니다.
- final score만 저장하지 말고 raw feature와 component score를 같이 저장합니다.
- family별 weight를 다르게 둡니다.
- confidence와 severity를 분리합니다.
- 초기에는 rule-based score로 시작합니다.
- 전문가 severity label이나 yield impact가 쌓이면 supervised severity model로 확장합니다.
