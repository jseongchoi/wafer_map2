# 불량 Family 정의

이 문서는 WaferMap에서 사용하는 defect family를 정의합니다.
family 정의가 흔들리면 라벨 품질이 바로 무너지므로, 애매한 경우 이 문서를 기준으로
판단합니다.

## 1. 현재 모델 family

| Family | 정의 | 예시 | 라벨 방식 |
|---|---|---|---|
| `local` | 작고 국소적인 blob/cluster | 특정 die 근처만 fail | 수동 mask |
| `scratch` | 길게 이어지는 선형/곡선형 불량 | 중앙 대각선 scratch | polyline+width 또는 mask |
| `ring` | 원형/부분 원형 band | 반지름 300px 부근 ring | center/radius/width |
| `edge` | wafer edge 근처 band/sector | 오른쪽 edge fail | angle/radial range |
| `shot_grid` | shot 상대 위치 반복 불량 | 각 shot의 왼쪽 아래 die fail | shot layout + affected slot |
| `random` | 구조가 약한 산발 fail | 드문드문 fail pixel | procedural 또는 보조 |

## 2. local

`local`은 작은 영역에 국소적으로 몰린 불량입니다.

좋은 예:

```text
한 wafer의 왼쪽 아래에 20x30px 정도의 뚜렷한 blob
특정 die 주변만 fail density가 높음
```

나쁜 예:

```text
wafer 절반이 흐리게 나빠짐
edge 전체가 나빠짐
긴 선 형태
```

### local 안의 subtype은 처음에는 metadata로 둡니다

`local` blob 안에서도 세부 유형은 나뉠 수 있습니다. 예를 들면 아래와 같습니다.

| Subtype 후보 | 설명 | 처음 처리 |
|---|---|---|
| `tiny_isolated_blob` | 작은 점 하나 | `family=local`, metadata only |
| `small_isolated_blob` | 작은 blob 하나 | `family=local`, metadata only |
| `cluster_blob` | 여러 blob이 가까이 모임 | `family=local`, metadata only |
| `large_blob` | 큰 덩어리 | `family=local`, metadata only |
| `diffuse_blob` | 경계가 흐림 | `mixed_unknown` 또는 `local` low confidence |
| `comet_blob` | blob에 꼬리처럼 늘어짐 | `local` 또는 `scratch` review |
| `edge_near_blob` | edge 가까운 blob | `family=local`, 위치 metadata 기록 |

처음부터 이 subtype들을 target channel로 쪼개지 않습니다.

```text
초기 target:
local

metadata:
subtype = cluster_blob
subtype_status = metadata_only
```

이유는 subtype 기준이 초기에 흔들리기 쉽고, 샘플 수가 부족하면 모델이
subtype을 배우는 것이 아니라 라벨러의 흔들림을 배울 수 있기 때문입니다.

subtype은 아래 조건이 충분할 때만 별도 subfamily 또는 target channel로 승격합니다.

```text
공정/품질 관점에서 의미가 다르다
리뷰어가 일관되게 구분할 수 있다
샘플 수가 충분하다
mask 생성 방식이나 후처리가 다르다
모델 성능을 따로 봐야 할 필요가 있다
```

승격 전:

```json
{
  "family": "local",
  "subtype": "cluster_blob",
  "subtype_status": "metadata_only"
}
```

승격 후:

```json
{
  "family": "local_cluster",
  "parent_family": "local",
  "subtype_status": "target_channel"
}
```

## 3. scratch

`scratch`는 선처럼 이어지는 불량입니다. 직선일 수도 있고 곡선일 수도 있습니다.

좋은 예:

```text
wafer 중앙을 가로지르는 긴 선
여러 die를 관통하는 얇은 band
```

라벨 팁:

- 선 중심을 polyline으로 찍고 width를 지정할 수 있으면 parametric label이 좋습니다.
- 실제 경계가 복잡하면 수동 mask를 씁니다.

## 4. ring

`ring`은 wafer 중심 기준 특정 반지름 근처에 나타나는 원형 또는 호형 불량입니다.

좋은 예:

```text
반지름 300px 부근에 얇은 원형 band
270도 정도만 이어진 partial ring
```

라벨 팁:

```json
{
  "center_xy": [512, 512],
  "radius": 300,
  "width": 24,
  "theta_range_deg": [20, 310]
}
```

## 5. edge

`edge`는 wafer edge 근처에 붙어 있는 불량입니다.

좋은 예:

```text
오른쪽 2시~5시 방향 edge만 fail
wafer 외곽 10% ring 영역에만 이상
```

`ring`과 헷갈릴 때:

- wafer 중심에서 특정 반지름 전체가 문제면 `ring`
- wafer 가장자리 sector가 문제면 `edge`

## 6. shot_grid

`shot_grid`는 shot 구조 기준으로 같은 상대 위치에 반복되는 불량입니다.

예:

```text
모든 shot마다 왼쪽 아래 die가 fail
3x3 shot 안에서 [2, 0] slot만 반복적으로 낮음
```

이 경우 사람이 모든 반복 위치를 칠하지 않습니다.
shot layout과 affected slot을 저장하고 코드가 mask를 만듭니다.

## 7. random

`random`은 구조가 약한 sparse fail pattern입니다.
주요 목적은 모델이 모든 fail을 `local`이나 `scratch`로 오해하지 않게 하는
baseline context입니다.

주의:

- 뚜렷한 family가 있으면 `random`으로 숨기지 않습니다.
- 너무 많은 random을 넣으면 모델이 defect family를 덜 선명하게 배울 수 있습니다.

## 8. STBY / Missing Test

`stby_pattern`은 primary defect target이 아닙니다.
이는 측정/관측 가능 영역을 설명하는 context입니다.

학습 target:

```text
local, scratch, ring, edge, shot_grid, random
```

보조 context:

```text
stby_mask, valid_test_mask, wafer_mask
```

## 9. 애매한 경우

애매하면 억지로 family를 정하지 않습니다.

| 상황 | 처리 |
|---|---|
| scratch인지 edge인지 불명확 | review-only |
| diffuse라 경계가 없음 | `mixed_unknown` |
| 두 family가 겹쳐 분리 어려움 | instance를 나누거나 학습 제외 |
| 실제 defect인지 전처리 artifact인지 모름 | 학습 제외 |

## 10. Target 계약

U-Net target은 family별 binary mask입니다.

```text
target channel order:
local
scratch
ring
edge
shot_grid
random
```

각 pixel은 여러 family에 동시에 속할 수 있습니다.
따라서 multi-class softmax가 아니라 multi-label sigmoid 구조가 맞습니다.
