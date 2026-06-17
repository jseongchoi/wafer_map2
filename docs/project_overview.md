# Project Overview

## 목적

이 프로젝트는 1채널 고해상도 Wafer Fail Bit Map에서 실제 wafer에도 적용 가능한 표현을 만드는 것이다.

목표 표현은 단일 classification label이 아니라 downstream task에 쓰이는 feature/retrieval 기반 표현이다.

```text
semantic FBM tensor
-> compact observable wafer feature
-> similar wafer retrieval
-> interest-conditioned defect retrieval
-> defect score / feature table
-> expert review feedback
-> real-unlabeled secure workflow
```

ANOVA와 공정 metadata 해석은 현재 목표가 아니다. 공정/설비/lot/recipe/chamber metadata가 붙은 뒤 feature table과 조인해 수행할 후속 분석이다.

## 현재 진행 상태

- Synthetic generator는 Grade 0~7, none-wafer, valid-test, stby, edge/local/shot/ring/scratch 계열을 생성한다.
- Global retrieval은 compact observable feature 50개 기준을 유지한다.
- Scale 155장 top-k retrieval lift는 약 1.36x, holdout 120장은 약 1.40x다.
- Interest-conditioned retrieval과 structured defect feature retrieval은 class, class_location, feature_key 기준에서 신호가 있다.
- Resize-only representation은 global retrieval 대체재로 부적합하다고 판단했다.
- Patch proposal은 edge/local/stby review 후보 생성 보조층으로 유지한다.
- Curve proposal은 ring/center arc review 후보 생성 보조층으로 유지한다.
- Scratch는 rule/proposal 과투자를 멈추고 segmentation 또는 scratch-specific line representation으로 분리한다.
- Real-unlabeled workflow MVP는 semantic `.npz` manifest, feature CSV, sanity JSON, nearest-neighbor CSV, expert review template까지 연결되어 있다.
- Global nearest-neighbor 경로에서는 `polar_*`, `stby_polar_*` 위치 feature를 제외한다.

## 본질 정렬 판정

판정: 본질대로 진행 중이다.

이유:

- 실제 inference feature는 observable-only 계약을 유지한다.
- Synthetic oracle인 `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 검증용으로 분리되어 있다.
- Global retrieval과 위치-aware retrieval의 feature 계약이 분리되어 있다.
- Proposal 계열을 주 경로가 아니라 review 후보 축소용 보조층으로 제한했다.
- 현재 다음 작업이 새 모델 도입이 아니라 real-unlabeled contract와 expert review loop에 맞춰져 있다.

가장 큰 리스크는 synthetic 성능을 real 성능으로 오해하는 것이다. 지금 수치는 방법론 가능성의 근거이지 실제 wafer 성능 인증이 아니다.

## 지금 우선해야 할 작업

1. Real-unlabeled manifest/schema를 운영 계약으로 고정한다.
2. 실제 보안 환경에서 semantic `.npz`를 export해 sanity와 feature drift를 확인한다.
3. Nearest-neighbor 결과를 expert review template으로 평가한다.
4. Reviewer의 `retrieval_failure_mode`, `next_action`을 feature/model backlog로 연결한다.
5. Scratch/local처럼 약한 계열은 morphology 또는 segmentation track으로 분리한다.

## 하지 말아야 할 일

- 실제 wafer raw image/array를 repo에 저장하지 않는다.
- Synthetic oracle label/mask를 real inference feature에 섞지 않는다.
- Global retrieval에 `polar_*`, `stby_polar_*`를 넣지 않는다.
- Proposal recall을 최종 성능처럼 해석하지 않는다.
- ANOVA를 현재 milestone 목표로 당기지 않는다.
- AutoEncoder나 대형 모델을 1차 해결책처럼 붙이지 않는다.

## 참고 연구의 위치

외부 연구는 현재 방향을 뒷받침하지만, 곧장 대형 모델로 넘어가라는 의미는 아니다.

- Iterative Cluster Harvesting for Wafer Map Defect Patterns
  - feature extraction, dimensionality reduction, clustering, manual labeling 지원 흐름이 현재 `feature -> retrieval/grouping -> expert review` 방향과 맞다.
  - https://arxiv.org/abs/2404.15436

- Graph-theoretic spatial filtering for mixed-type wafer bin maps
  - 인접 defect chip의 spatial dependence를 이용한 filtering/clustering 관점은 local/scratch morphology 보강의 참고선이다.
  - https://arxiv.org/abs/2006.13824

- WaferSegClassNet
  - mixed-type wafer defect classification/segmentation을 함께 다루며, scratch/local/overlap 보강을 segmentation track으로 분리한 현재 판단과 맞다.
  - https://arxiv.org/abs/2207.00960

