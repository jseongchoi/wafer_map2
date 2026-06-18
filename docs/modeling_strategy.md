# 모델링 전략

## 기본 입장

현재 성공 기준은 "딥러닝 모델을 학습했다"가 아니다.

먼저 확인해야 할 것은 다음이다.

```text
실제 wafer에서도 계산 가능한 feature가 있는가?
그 feature로 비슷한 wafer를 찾을 수 있는가?
관심 불량별 검색 신호가 있는가?
전문가가 top-k 결과를 업무적으로 받아들일 수 있는가?
```

따라서 현재 1차 방법론은 해석 가능한 feature와 nearest-neighbor 검색이다.

## 기준선 Feature

현재 기준선은 사람이 해석할 수 있는 계측 feature다.

주요 feature family:

- global fail density / grade-weighted severity
- stby ratio
- radial / angular profile
- edge density / localized edge sector
- ring / scratch morphology proxy
- local hotspot / connected component morphology
- shot-relative lower-left, bottom-edge, left-edge contrast

이 기준선의 역할:

- 합성 label로 검증 가능한 첫 비교 기준
- 라벨 없는 실제 wafer에 바로 적용 가능한 feature
- 전문가 리뷰에서 사람이 이해할 수 있는 feature 이름 제공
- segmentation/self-supervised model을 붙일 때 비교 기준 제공

## 현재 검증 결과

- Scale 155장 top-5 retrieval lift: 약 1.36x
- Holdout 120장 top-5 retrieval lift: 약 1.40x
- Structured defect target retrieval은 class, class_location, feature_key 기준에서 신호가 있다.
- 단순 resize representation은 compact feature보다 전체 유사 wafer 검색에 약하다.
- Scratch는 holdout에서 불안정하므로 wafer-level feature만으로 끝내지 않는다.

## Feature 사용 기준

전체 유사 wafer 검색:

```text
compact feature 50개만 사용
```

제외:

```text
label_*
*_mask_ratio
pattern_masks
pattern_intensity
polar_*
stby_polar_*
```

위치가 중요한 검색:

```text
class_location / feature_key 같은 target에서만 polar feature를 조건부로 사용
```

## Proposal의 역할

Patch proposal과 curve proposal은 보조 도구다.

- edge/local/stby: patch proposal로 리뷰 후보 영역 축소
- ring/center arc: curve proposal로 후보 곡선 영역 축소
- scratch: proposal 튜닝 중단, 별도 track으로 분리

Proposal recall을 최종 성능으로 해석하지 않는다.

## Segmentation / Self-Supervised Model의 역할

모델은 기준선을 무조건 대체하는 것이 아니라, 실제 리뷰에서 반복적으로 약한 defect family를 보강하는 층이다.

시작 조건:

- 라벨 없는 실제 wafer sanity check가 통과한다.
- 전문가 리뷰에서 특정 family 실패가 반복된다.
- feature 보강만으로 부족하다는 근거가 있다.

후보:

- lightweight U-Net 계열 multi-label segmentation
- scratch-specific line representation
- self-supervised encoder / metric learning
- domain adaptation

현재 구현 상태:

- 구현됨: feature extractor, nearest-neighbor 검색, 관심 불량 검색 평가, 라벨 없는 실제 wafer 리뷰 절차
- smoke 수준 구현됨: synthetic-label segmentation dataset helper와 NumPy-only 1x1 sigmoid segmentation smoke training
- 아직 아님: 실사용 U-Net/SegFormer/DINO model, 실제 wafer로 검증된 supervised/self-supervised model, calibrated defect probability model

## 왜 AutoEncoder부터 하지 않는가

현재 문제는 단순 anomaly detection이 아니다.

우리가 알고 싶은 것은 다음이다.

```text
어떤 defect pattern이 어디에, 얼마나 강하게 나타나는가?
stby chip이 어떤 defect origin을 가리고 있는가?
비슷한 defect 조합을 가진 wafer들이 가까이 묶이는가?
```

AutoEncoder는 stby 같은 큰 missing-test rectangle에 과민하게 반응할 수 있고, scratch/ring/edge/local/shot-relative 의미 분해를 직접 보장하지 않는다.

따라서 AutoEncoder는 후속 보조 도구로 둔다.

## 참고 연구

- Iterative Cluster Harvesting for Wafer Map Defect Patterns
  - feature extraction, dimensionality reduction, clustering을 반복해 manual labeling을 돕는 흐름.
  - https://arxiv.org/abs/2404.15436

- Graph-Theoretic Spatial Filtering for Mixed-Type Wafer Bin Maps
  - 인접 defect chip의 spatial dependence를 이용한 filtering/clustering.
  - https://arxiv.org/abs/2006.13824

- WaferSegClassNet
  - mixed-type wafer defect classification/segmentation을 함께 다루는 lightweight encoder-decoder.
  - https://arxiv.org/abs/2207.00960
