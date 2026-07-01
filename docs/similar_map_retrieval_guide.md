# 유사맵 검색 가이드

이 문서는 wafer map에서 “비슷한 wafer”를 어떻게 찾을지 설명합니다.
현재 주력은 segmentation dataset과 U-Net correction loop이지만, 유사맵 검색은
리뷰 후보를 찾고 신규 불량을 정리하는 데 중요한 보조 축입니다.

핵심 결론은 아래입니다.

```text
초기: 해석 가능한 feature 기반 nearest-neighbor
중기: segmentation mask feature로 rerank
후기: encoder embedding을 feature/mask 검색과 결합
```

처음부터 딥러닝 embedding만 믿지 않습니다. 먼저 사람이 이해할 수 있는 feature로
검색하고, mask와 correction data가 쌓이면 점점 강한 표현을 붙입니다.

## 1. 유사맵은 기준을 먼저 정해야 합니다

“비슷하다”는 말은 여러 의미가 있습니다.

| 검색 기준 | 질문 | 예시 |
|---|---|---|
| 전체 wafer 형태 | 전체 fail density와 radial/edge 분포가 비슷한가? | 전체적으로 edge fail이 강함 |
| family | 같은 defect family가 있는가? | 둘 다 `local` blob이 있음 |
| 위치 | defect가 비슷한 위치에 있는가? | 둘 다 왼쪽 아래 |
| subtype | 같은 세부 형태인가? | 둘 다 `cluster_blob` |
| mask shape | mask 크기/모양/방향이 비슷한가? | scratch 방향이 비슷함 |
| shot-relative | shot 기준 반복 위치가 비슷한가? | shot마다 왼쪽 아래 slot |

검색 목적이 다르면 feature weight도 달라져야 합니다.

```text
local blob 유사 검색:
  blob 위치, 면적, 개수, compactness가 중요

ring 유사 검색:
  radius, width, angle coverage가 중요

shot_grid 유사 검색:
  shot layout, affected slot, 반복 위치가 중요
```

## 2. 1단계: feature 기반 nearest-neighbor

처음에는 wafer 하나를 숫자 vector로 바꿉니다.

```json
{
  "sample_id": "WAFER_0001",
  "global_fail_density": 0.031,
  "severity_mean": 1.42,
  "edge_fail_ratio": 0.18,
  "ring_score": 0.64,
  "scratch_score": 0.12,
  "local_component_count": 4,
  "largest_blob_area": 820,
  "largest_blob_radial_norm": 0.72,
  "largest_blob_angle_deg": 238,
  "stby_ratio": 0.04
}
```

그 다음 query wafer와 candidate wafer 사이의 거리를 계산합니다.

```text
distance = weighted_distance(query_features, candidate_features)
distance가 작을수록 비슷한 wafer
```

이 방식의 장점은 검색 결과를 사람이 설명할 수 있다는 것입니다.

```text
이 wafer가 비슷한 이유:
local blob 개수가 비슷함
largest_blob_area가 비슷함
blob 위치가 왼쪽 아래로 비슷함
edge fail ratio도 비슷함
```

## 3. local/blob 유사맵 feature

blob 기준으로 유사맵을 찾을 때는 component feature가 중요합니다.

| Feature | 의미 |
|---|---|
| `local_component_count` | local blob component 개수 |
| `largest_blob_area` | 가장 큰 blob 면적 |
| `total_blob_area` | blob 전체 면적 |
| `blob_centroid_x`, `blob_centroid_y` | blob 중심 위치 |
| `blob_radial_norm` | wafer 중심에서 얼마나 먼지 |
| `blob_angle_deg` | clock/angle 위치 |
| `blob_compactness` | 얼마나 둥글고 조밀한지 |
| `blob_eccentricity` | 길쭉한 정도 |
| `blob_intensity_mean` | blob 내부 평균 강도 |
| `blob_near_edge_distance` | edge와의 거리 |

예시:

```text
query:
  local blob 1개
  area 800px
  왼쪽 아래
  compactness 높음

candidate A:
  local blob 1개
  area 760px
  왼쪽 아래
  compactness 높음
  -> 유사

candidate B:
  local blob 5개
  오른쪽 위
  cluster 형태
  -> 덜 유사
```

## 4. subtype은 검색 metadata로 먼저 씁니다

subtype은 처음에는 모델 target으로 쪼개지 않습니다. 하지만 유사맵 검색에는
적극적으로 사용할 수 있습니다.

```json
{
  "family": "local",
  "subtype": "cluster_blob",
  "subtype_status": "metadata_only"
}
```

검색 조건 예시:

```text
family = local
subtype = cluster_blob
radial_norm 차이 작음
largest_blob_area 차이 작음
```

즉 운영 원칙은 아래입니다.

```text
학습 target:
  local 하나로 유지

검색/리뷰:
  subtype metadata를 적극 사용
```

이렇게 하면 subtype이 실제로 의미 있는지 먼저 검증할 수 있습니다. 비슷한 wafer들이
같은 subtype으로 잘 묶이면, 나중에 target channel 승격 후보가 됩니다.

## 5. 2단계: mask feature로 rerank

수동 mask나 U-Net prediction mask가 생기면 검색이 더 정확해집니다.

사용할 수 있는 mask:

```text
local_mask
scratch_mask
ring_mask
edge_mask
shot_grid_mask
```

mask에서 뽑는 feature 예시:

| Family | Mask feature |
|---|---|
| `local` | component count, area, centroid, compactness |
| `scratch` | length, orientation, width, curvature |
| `ring` | radius, width, angle coverage, center offset |
| `edge` | sector angle, radial band width, edge coverage |
| `shot_grid` | affected slot, repetition ratio, intra-die region |

추천 구조:

```text
1차 검색:
  빠른 observable feature로 top 100 후보 찾기

2차 재정렬:
  family/mask/subtype feature로 top 10 rerank
```

## 6. 3단계: encoder embedding

데이터가 쌓이면 encoder embedding을 추가할 수 있습니다.

```text
wafer image 또는 multi-channel tensor
-> encoder
-> embedding vector
-> nearest-neighbor search
```

하지만 embedding은 후순위입니다.

초기에는 feature 기반 검색이 더 낫습니다.

```text
데이터가 적어도 동작함
왜 비슷한지 설명 가능
검색 실패 원인을 알 수 있음
family/subtype 정의 검증에 도움
```

embedding은 아래 조건이 생기면 붙입니다.

```text
수동/correction mask가 충분히 쌓임
feature 기반 검색의 한계가 반복적으로 확인됨
비슷함/다름에 대한 review label이 있음
검색 결과를 정량 평가할 수 있음
```

## 7. 실행 위치

현재 repo에는 feature 추출과 retrieval 관련 script가 보조 축으로 남아 있습니다.

| 목적 | Script |
|---|---|
| 실제 wafer feature 추출 | `scripts/extract_real_unlabeled_features.py` |
| synthetic sample feature 추출 | `scripts/extract_features.py` |
| feature retrieval 평가 | `scripts/evaluate_defect_feature_retrieval.py` |
| interest retrieval 평가 | `scripts/evaluate_interest_retrieval.py` |
| retrieval confidence 평가 | `scripts/evaluate_retrieval_confidence.py` |
| CPU encoder baseline | `scripts/train_cpu_encoder_model.py` |
| unlabeled wafer encoder scoring | `scripts/score_unlabeled_cpu_encoder.py` |

예시:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --out outputs/features/product_A_real_features.parquet
```

## 8. 리뷰 결과 형식

유사맵 검색 결과는 반드시 사람이 볼 수 있어야 합니다.

리뷰 column 예시:

| Column | 의미 |
|---|---|
| `query_sample_id` | 찾고 싶은 wafer |
| `neighbor_sample_id` | 검색된 유사 wafer |
| `rank` | 검색 순위 |
| `distance` | feature/embedding 거리 |
| `query_family` | query의 주요 family |
| `neighbor_family` | neighbor의 주요 family |
| `query_subtype` | query subtype metadata |
| `neighbor_subtype` | neighbor subtype metadata |
| `reviewer_decision` | `same`, `partial`, `different`, `not_sure` |
| `retrieval_failure_mode` | 틀렸다면 이유 |

실패 이유 예시:

```text
wrong_family
right_family_wrong_location
right_location_wrong_shape
subtype_mismatch
parser_or_mask_issue
```

## 9. local/blob부터 닫는 추천 순서

현재 진행 방향에서는 local/blob vertical slice와 유사맵 검색을 같이 닫는 것이 좋습니다.

```text
1. local blob mask를 만든다
2. subtype metadata를 저장한다
3. blob component feature를 뽑는다
4. feature 기반 top-k 유사 wafer를 찾는다
5. 사람이 top-k가 맞는지 리뷰한다
6. mask가 쌓이면 mask feature로 rerank한다
7. 충분히 쌓이면 embedding을 검토한다
```

이 방식은 subtype 승격 판단에도 도움이 됩니다.

```text
cluster_blob끼리 실제로 잘 묶이는가?
tiny_blob은 별도 검색 기준이 필요한가?
diffuse_blob은 local에서 빼야 하는가?
```

## 10. 판단 규칙

- 유사맵 검색은 segmentation 모델을 대체하지 않습니다. 리뷰 후보를 찾는 보조 축입니다.
- 처음에는 observable feature만 씁니다. 실제 wafer inference에서 없는 oracle mask 값은 쓰지 않습니다.
- 수동/correction mask가 생기면 mask feature를 추가합니다.
- subtype은 target channel로 승격하기 전에도 검색 metadata로 사용할 수 있습니다.
- embedding은 데이터와 review label이 쌓인 뒤 붙입니다.
