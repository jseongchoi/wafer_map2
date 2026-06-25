# 검증 프로토콜

이 문서는 FBM defect pattern pipeline을 어떤 기준으로 검증할지 정의한다. 검증의 핵심은 “모델이 무언가를 맞췄다”가 아니라, 실제 업무자가 wafer map을 보고 납득할 수 있는 위치, family, 수치가 나오는지 확인하는 것이다.

## 1. 입력 검증

실제 입력은 두 가지 형식을 우선 지원한다.

| 형식 | 용도 |
| --- | --- |
| `png_grayscale_raw` | 실제 Grade 0~7 이미지. 제품별 폴더 batch 처리에 적합 |
| `npz_semantic_arrays` | `severity`, `wafer_mask`, `valid_test_mask`, `stby_mask`를 직접 가진 semantic 배열 |

PNG 입력 sanity:

```text
gray value가 0, 31, 151, 175, 191, 207, 223, 255 중 하나인가?
255 전체 chip은 stby로 분리되는가?
일부 pixel만 255인 chip은 Grade 7 fail로 유지되는가?
제품별 chip block과 grid가 PNG shape와 맞는가?
wafer_mask가 실제 net die layout과 맞는가?
```

## 2. 누끼/마스크 검증

누끼 asset은 family별로 저장한다.

```text
data/masks/blob/
data/masks/ring/
data/masks/scratch/
data/masks/edge/
data/masks/random/
```

검증 기준:

- 한 defect가 여러 조각으로 저장되지 않았는지 확인한다.
- ring처럼 연결된 패턴은 하나의 component 또는 하나의 asset으로 보존한다.
- 사람이 따기 어려운 random/edge는 editor asset만 고집하지 않고 코드형 generator를 허용한다.
- mask preview에서 원본 wafer, 선택 영역, 저장된 binary mask를 함께 확인한다.

## 3. 합성 데이터 검증

합성 데이터는 반드시 image, label mask, metadata를 같이 만든다.

```text
outputs/synthetic_dataset/<run>/images/*.png
outputs/synthetic_dataset/<run>/masks/*.npz
outputs/synthetic_dataset/<run>/metadata.csv
outputs/synthetic_dataset/<run>/preview.html
```

필수 metadata:

```text
sample_id
base_wafer_id
family_list
family_count
mask_path
image_path
center_x
center_y
area_pixels
severity_before
severity_after
```

## 4. 모델 검증

추천 baseline은 U-Net 계열 multi-label segmentation이다. 한 wafer에 여러 defect family가 동시에 있을 수 있으므로 softmax 단일 클래스보다 sigmoid multi-channel output이 적합하다.

출력:

```text
family별 probability mask
family별 area / centroid / bbox / radial position / angular position
wafer-level family score
embedding vector
top-k similar wafer
```

평가 metric:

```text
per-family Dice/F1
per-family IoU
small defect recall
overlap defect recall
centroid distance
area error
top-k retrieval accept rate
missed major defect rate
```

## 5. 실제 리뷰 검증

실제 wafer에는 완전한 ground truth가 없을 수 있다. 그래서 전문가 리뷰를 모델 개선 루프의 중심으로 둔다.

리뷰 항목:

```text
query_sample_id
model_family
reviewer_family
position_correct
mask_quality
missed_major_defect
false_positive_family
comment
```

판정 기준:

- `accepted_match_rate`: top-k 또는 모델 제안이 같은 계열로 보이는 비율
- `query_topk_accept_rate`: query별 최소 1개 이상 납득 가능한 neighbor가 있는 비율
- `missed_major_defect_rate`: 사람이 볼 때 큰 defect를 모델이 놓친 비율
- `family_confusion`: 어떤 family끼리 자주 헷갈리는지

## 6. 개선 루프

1. 실제 FBM batch를 넣는다.
2. 리포트와 preview를 본다.
3. 틀린 family, 위치 오류, 누락 defect를 CSV로 기록한다.
4. 필요한 family의 mask asset 또는 generator를 보강한다.
5. synthetic dataset을 다시 만든다.
6. segmentation 모델과 embedding 검색을 다시 평가한다.

이 루프가 쌓이면 “무엇을 잘 못하는지”가 보이고, 그때부터는 딥러닝 모델 개선이 실제 업무 문제와 직접 연결된다.
