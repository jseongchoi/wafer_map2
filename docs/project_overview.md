# 프로젝트 개요

## 목표

이 프로젝트는 1채널 고해상도 Wafer Fail Bit Map에서 실제 wafer에도 쓸 수 있는 표현을 만드는 것이다.

여기서 말하는 표현은 단일 classification label이 아니다. 유사 wafer 검색, 관심 불량 검색, defect score table, 모델 입력에 쓸 feature 기반 표현이다.

```text
FBM 배열 정리
-> 실제 wafer에서도 계산 가능한 feature 추출
-> 비슷한 wafer 검색
-> 관심 불량별 검색
-> defect score / feature table
-> 전문가 리뷰 결과 반영
-> 라벨 없는 실제 wafer 검증
```

## 현재 상태

- 합성 FBM generator는 Grade 0~7, wafer 밖 영역, 실제 test 영역, stby, edge/local/shot/ring/scratch 계열을 생성한다.
- 전체 유사 wafer 검색은 실제 wafer에서도 계산 가능한 compact feature 50개를 기준으로 한다.
- Scale 155장 top-k retrieval lift는 약 1.36x, holdout 120장은 약 1.40x다.
- 관심 불량별 retrieval과 structured defect feature retrieval은 class, class_location, feature_key 기준에서 신호가 있다.
- 단순 resize representation은 전체 유사 wafer 검색의 대체재로 부적합하다고 판단했다.
- Patch proposal은 edge/local/stby 리뷰 후보를 줄이는 보조 도구로만 둔다.
- Curve proposal은 ring/center arc 리뷰 후보를 줄이는 보조 도구로만 둔다.
- Scratch는 rule/proposal에 더 깊게 투자하지 않고 segmentation 또는 scratch 전용 line feature 쪽으로 분리한다.
- 라벨 없는 실제 wafer 처리 절차는 `.npz` manifest, feature CSV, sanity JSON, nearest-neighbor CSV, 전문가 리뷰 template까지 연결되어 있다.
- 전체 유사 wafer 검색에서는 `polar_*`, `stby_polar_*` 위치 feature를 제외한다.

## 목표 대비 진행 상황

| 사용자가 원한 것 | 지금까지 한 일 | 확인한 근거 | 자세히 볼 곳 |
| --- | --- | --- | --- |
| 실제 wafer에도 적용 가능한 FBM feature | 합성 데이터의 oracle label 없이 계산 가능한 compact feature를 정리했다. 전체 검색은 50개 feature 기준으로 유지한다. | scale 155장 top-k lift 약 1.36x, holdout 120장 약 1.40x. | [모델링 전략](modeling_strategy.md), [로드맵](roadmap.md) |
| 유사 wafer 검색 | feature 표준화와 nearest-neighbor 계산을 공통 유틸로 분리했다. 전체 검색에서는 위치 feature를 제외했다. | random baseline 대비 retrieval lift가 유지된다. | `src/wafermap/evaluation/nearest.py`, `src/wafermap/features/selection.py` |
| 관심 불량별 검색 | class, class_location, feature_key 기준 retrieval을 분리해 검증했다. 위치 feature는 이 경로에서만 조건부로 쓴다. | feature_key/class_location에서 random 대비 lift가 관측됐다. | [모델링 전략](modeling_strategy.md) |
| defect score / feature table / downstream task용 표현 | edge, shot, stby, ring, local, scratch proxy feature와 structured defect feature row 방향을 정리했다. | feature table을 검색, 리뷰, 모델 입력으로 연결할 수 있는 형태가 생겼다. | [데이터 형식](data_schema.md), [불량 패턴 정리](pattern_taxonomy.md) |
| 실제 보안 환경 적용 | `real_unlabeled_manifest/v1`와 `.npz` 입력 형식을 만들고, raw data를 repo에 저장하지 않는 절차를 구현했다. | synthetic smoke로 feature CSV, sanity JSON, NN CSV, review template 생성이 통과했다. | [라벨 없는 실제 wafer 처리 절차](real_unlabeled_workflow.md), `scripts/extract_real_unlabeled_features.py` |
| 전문가 리뷰 연결 | nearest-neighbor 결과를 reviewer CSV로 바꾸고, 리뷰 결과에서 실패 유형과 다음 작업을 집계한다. | 사람의 판단을 feature 보강 또는 AI 모델 후보로 연결하는 절차가 생겼다. | [전문가 리뷰 절차](expert_review_protocol.md), `scripts/summarize_expert_review.py` |
| proposal 과투자 방지 | patch/curve proposal은 리뷰 후보 축소용으로 제한하고, scratch는 별도 track으로 분리했다. | resize-only와 proposal-only는 전체 검색을 대신할 수 없다고 정리했다. | [로드맵](roadmap.md), [모델링 전략](modeling_strategy.md) |
| 외부 연구 참고 | wafer map clustering/manual labeling, graph spatial filtering, segmentation 연구를 참고해 현재 전략의 위치를 정리했다. | 현재 방향은 `feature -> retrieval/grouping -> expert review`이고, scratch/local은 morphology/segmentation 보강이 필요하다는 판단과 맞다. | 아래 참고 연구, [모델링 전략](modeling_strategy.md) |

실험한 기법과 판단 과정은 [실험과 판단 기록](experiment_history.md)에 모아두었다.

## 처음 목표와 맞게 가고 있나

현재 방향은 처음 목표와 맞다.

이유:

- 실제 inference feature는 실제 데이터에서도 계산 가능한 값만 사용한다.
- 합성 데이터의 oracle인 `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 검증용으로 분리해 두었다.
- 전체 유사 wafer 검색과 위치-aware 검색의 feature 사용 기준을 분리했다.
- Proposal 계열은 주 경로가 아니라 리뷰 후보를 줄이는 보조 도구로 제한했다.
- 지금 우선순위는 새 모델 도입이 아니라 라벨 없는 실제 wafer 입력 형식과 전문가 리뷰 절차를 검증하는 것이다.

가장 큰 리스크는 합성 데이터 성능을 실제 wafer 성능으로 착각하는 것이다. 지금 수치는 방법론 가능성의 근거이지 실제 wafer 성능 인증이 아니다.

## 최종 점검 요약

| 영역 | 현재 상태 | 판단 |
| --- | --- | --- |
| 합성 FBM generator | 다양한 defect family와 stby/valid-test/wafer 밖 영역을 생성한다. | 기준선 검증용으로 충분 |
| 실제 데이터용 feature | 실제 wafer에도 계산 가능한 compact feature와 전체 검색 기준을 정리했다. | 주 경로 유지 |
| 전체 유사 wafer 검색 | scale/holdout에서 random baseline 대비 lift가 있다. | 1차 솔루션 후보로 유효 |
| 관심 불량별 검색 | class/class_location/feature_key 기준 신호가 있다. | defect별 search 방향 유효 |
| Resize/proposal | resize-only는 대체재가 아니고 proposal은 리뷰 후보 축소용이다. | 과투자 방지 완료 |
| 라벨 없는 실제 wafer 처리 | `.npz` manifest에서 feature/sanity/NN/review template까지 생성된다. | 실제 리뷰 직전 단계 |
| 전문가 리뷰 | reviewer decision, failure mode, next action을 summary로 연결한다. | 실제 검증 대기 |
| AI 모델 | segmentation smoke training 배관은 있으나 실사용 deep model은 아직 아니다. | 실제 리뷰 후 target 결정 |

현재 결론:

```text
1차 기준선 솔루션은 잡혔다.
아직 최종 솔루션이 확정된 것은 아니다.
다음 확인 단계는 실제 wafer 전문가 리뷰다.
```

## 지금 우선해야 할 작업

1. 실제 작업에서 쓸 `.npz` manifest 형식을 명확히 정한다.
2. 보안 환경에서 실제 wafer를 `.npz`로 export하고 sanity 결과와 feature drift를 확인한다.
3. Nearest-neighbor 결과를 전문가 리뷰 template으로 평가한다.
4. 리뷰어가 적은 `retrieval_failure_mode`, `next_action`을 feature 보강 또는 AI 모델 후보로 연결한다.
5. Scratch/local처럼 약한 계열은 morphology 또는 segmentation 쪽으로 분리한다.

## 하지 말아야 할 일

- 실제 wafer raw image/array를 repo에 저장하지 않는다.
- 합성 데이터의 oracle label/mask를 실제 inference feature에 섞지 않는다.
- 전체 유사 wafer 검색에 `polar_*`, `stby_polar_*`를 넣지 않는다.
- Proposal recall을 최종 성능처럼 해석하지 않는다.
- AutoEncoder나 대형 모델을 1차 해결책처럼 붙이지 않는다.

## 참고 연구의 위치

외부 연구는 현재 방향을 점검하는 참고 자료다. 곧장 대형 모델로 넘어가야 한다는 의미는 아니다.

- Iterative Cluster Harvesting for Wafer Map Defect Patterns
  - feature extraction, dimensionality reduction, clustering, manual labeling 지원 흐름이 현재 `feature -> retrieval/grouping -> expert review` 방향과 맞다.
  - https://arxiv.org/abs/2404.15436

- Graph-theoretic spatial filtering for mixed-type wafer bin maps
  - 인접 defect chip의 spatial dependence를 이용한 filtering/clustering 관점은 local/scratch morphology 보강의 참고선이다.
  - https://arxiv.org/abs/2006.13824

- WaferSegClassNet
  - mixed-type wafer defect classification/segmentation을 함께 다루며, scratch/local/overlap 보강을 segmentation 쪽으로 분리한 현재 판단과 맞다.
  - https://arxiv.org/abs/2207.00960
