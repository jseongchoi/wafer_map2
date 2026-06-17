# Modeling Strategy

## 기본 입장

현재 성공 기준은 "딥러닝 모델을 학습했다"가 아니다.

먼저 확인해야 할 것은 다음이다.

```text
실제 wafer에도 계산 가능한 observable feature가 있는가?
그 feature로 비슷한 wafer를 찾을 수 있는가?
관심 defect별 retrieval 신호가 있는가?
전문가가 top-k 결과를 업무적으로 받아들일 수 있는가?
```

따라서 현재 1차 방법론은 interpretable observable feature와 nearest-neighbor retrieval이다.

## Baseline Layer

현재 baseline은 사람이 해석 가능한 계측층이다.

주요 feature family:

- global fail density / grade-weighted severity
- stby ratio
- radial / angular profile
- edge density / localized edge sector
- ring / scratch morphology proxy
- local hotspot / connected component morphology
- shot-relative lower-left, bottom-edge, left-edge contrast

이 baseline의 역할:

- synthetic label로 검증 가능한 첫 기준선
- real unlabeled wafer에 바로 적용 가능한 feature
- expert review에서 사람이 판단할 수 있는 feature 이름 제공
- segmentation/self-supervised model의 비교 기준

## 현재 검증 결과

- Scale 155장 top-5 retrieval lift: 약 1.36x
- Holdout 120장 top-5 retrieval lift: 약 1.40x
- Structured defect target retrieval은 class, class_location, feature_key 기준에서 신호가 있다.
- Resize-only representation은 compact feature보다 global retrieval에 약하다.
- Scratch는 holdout에서 불안정하므로 wafer-level feature만으로 끝내지 않는다.

## Feature Contract

Global retrieval:

```text
compact observable feature 50개만 사용
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

위치-aware retrieval:

```text
class_location / feature_key 같은 target에서만 polar feature를 조건부 사용
```

## Proposal의 위치

Patch proposal과 curve proposal은 보조 도구다.

- edge/local/stby: patch proposal로 review 후보 영역 축소
- ring/center arc: curve proposal로 후보 곡선 영역 축소
- scratch: proposal 튜닝 중단, 별도 track으로 분리

Proposal recall을 최종 성능으로 해석하지 않는다.

## Segmentation / Self-Supervised Model의 위치

모델은 baseline을 대체하기보다 약한 defect family를 보강하는 layer다.

시작 조건:

- real-unlabeled sanity가 통과한다.
- expert review에서 특정 family failure가 반복된다.
- observable feature 보강만으로 부족하다는 근거가 있다.

후보:

- lightweight U-Net 계열 multi-label segmentation
- scratch-specific line representation
- self-supervised encoder / metric learning
- domain adaptation

현재 구현 상태:

- 구현됨: observable feature extractor, nearest-neighbor retrieval, interest retrieval evaluation, real-unlabeled review workflow
- smoke 수준 구현됨: synthetic-label segmentation dataset helper와 NumPy-only 1x1 sigmoid segmentation smoke training
- 아직 아님: 실사용 U-Net/SegFormer/DINO model, real wafer로 검증된 supervised/self-supervised model, calibrated defect probability model

## 왜 AutoEncoder First가 아닌가

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
