# 실험과 판단 기록

이 문서는 지금까지 어떤 기법을 시도했고, 어떤 이유로 현재 방향을 선택했는지 설명한다.

빠르게 전체 맥락만 보고 싶으면 [프로젝트 개요](project_overview.md)와 [Real Wafer 리뷰 체크리스트](real_wafer_review_checklist.md)를 먼저 보면 된다. 이 문서는 "왜 그렇게 판단했는가"를 확인할 때 읽는다.

## 핵심 결론

현재까지의 결론은 다음이다.

```text
실제 wafer에서도 계산 가능한 feature + nearest-neighbor 검색은 1차 기준선으로 쓸 만하다.
단순 resize representation은 전체 유사 wafer 검색의 대체재로 부족하다.
patch/curve proposal은 최종 판정 도구가 아니라 리뷰 후보를 줄이는 보조 도구다.
scratch/local은 rule/proposal 튜닝보다 morphology 또는 segmentation 쪽으로 분리하는 편이 맞다.
최종 판단은 실제 wafer 전문가 리뷰를 통과해야 한다.
```

따라서 현재 주 경로는 다음이다.

```text
보안 환경의 실제 wafer .npz
-> sanity / feature drift 확인
-> compact feature 추출
-> nearest-neighbor 검색
-> 전문가 리뷰 template
-> 실패 유형 / 다음 작업 정리
```

## 1. 문제 정의와 데이터 형식

처음부터 목표는 불량 하나를 맞히는 분류 모델이 아니었다.

목표는 1채널 고해상도 Wafer Fail Bit Map에서 실제 wafer에도 계산 가능한 feature 표현을 만들고, 그 표현으로 유사 wafer 검색, 관심 불량 검색, defect score table, 후속 작업을 가능하게 하는 것이다.

그래서 먼저 분석에 필요한 입력 형식을 정했다.

- `severity`: Grade 0~7
- `wafer_mask`: wafer 내부와 wafer 밖 영역 구분
- `valid_test_mask`: 실제 test 여부
- `stby_mask`: stby fail chip 영역
- `chip_index`: 권장 die/chip id

중요한 결정:

- stby는 Grade 7로 섞지 않는다.
- wafer 밖 영역과 in-wafer Grade 0은 `wafer_mask`로 구분한다.
- 합성 데이터의 oracle인 `pattern_masks`, `pattern_intensity`, `label_*`, `*_mask_ratio`는 검증용이다.
- 실제 inference feature에는 합성 데이터 oracle을 넣지 않는다.

관련 문서:

- [데이터 형식](data_schema.md)
- [불량 패턴 정리](pattern_taxonomy.md)

## 2. 합성 FBM Generator

실제 wafer raw data를 repo에 둘 수 없기 때문에, 먼저 합성 데이터로 방법론을 검증했다.

생성한 계열:

- Grade 0~7 severity
- wafer 밖 영역 / 실제 test 영역 / stby 분리
- edge gradient와 localized edge sector
- shot-relative 반복 패턴
- local hotspot
- stby-origin 또는 stby가 origin을 가리는 형태
- ring / partial ring / center arc
- scratch-like arc 또는 radial line
- mixed overlap

이 generator의 역할은 "실제와 완전히 같은 wafer 생산기"가 아니다. 다음을 확인하기 위한 검증 장치다.

- feature extractor가 입력 형식을 제대로 지키는가
- oracle label 없이 실제 데이터용 feature만으로 유사 wafer 검색 신호가 생기는가
- defect family별로 어떤 feature가 강하고 약한가
- 실제 리뷰 전에 전체 절차가 끝까지 연결되는가

주의할 점:

- 합성 데이터 성능은 실제 wafer 성능 인증이 아니다.
- generator와 feature extractor가 같은 가정을 공유할 수 있으므로 실제 wafer 전문가 리뷰가 필요하다.

관련 문서:

- [검증 방법](validation_protocol.md)
- [로드맵](roadmap.md)

## 3. 실제 데이터용 Feature 기준선

첫 기준선은 사람이 해석할 수 있고 실제 wafer에서도 계산 가능한 feature다.

주요 feature family:

- fail density와 grade-weighted severity
- stby ratio
- radial profile
- angular profile
- edge density와 edge sector contrast
- ring radial peak
- scratch angular peak proxy
- local hotspot / morphology proxy
- shot-relative lower-left, bottom-edge, left-edge contrast

전체 유사 wafer 검색 기준:

```text
compact feature 50개
```

전체 검색에서 제외하는 것:

```text
label_*
*_mask_ratio
pattern_masks
pattern_intensity
polar_*
stby_polar_*
```

`polar_*`, `stby_polar_*`는 위치가 중요한 검색에서만 조건부로 쓴다. 예를 들어 `class_location`, `feature_key`처럼 위치 자체가 검색 조건인 경우다.

구현 위치:

- `src/wafermap/features/selection.py`
- `src/wafermap/features/wafer_vector.py`

## 4. 전체 유사 Wafer 검색

실험 방법:

1. 합성 데이터 feature CSV를 만든다.
2. compact feature만 고른다.
3. feature를 표준화한다.
4. Euclidean nearest-neighbor를 계산한다.
5. synthetic label Jaccard는 검증 metric으로만 사용한다.
6. random neighbor 대비 top-k lift를 본다.

확인된 결과:

- scale 155장 top-k retrieval lift 약 1.36x
- holdout 120장 top-k retrieval lift 약 1.40x

해석:

- 실제 데이터용 feature만으로도 비슷한 wafer를 더 가깝게 찾는 신호가 있다.
- 이것은 최종 실제 wafer 성능 인증이 아니라 1차 기준선 후보의 근거다.
- scratch/local처럼 작은 구조 결함은 전체 lift가 좋아도 별도로 봐야 한다.

공통화한 구현:

- `src/wafermap/evaluation/nearest.py`
- `src/wafermap/features/selection.py`

## 5. 관심 불량별 검색

전체 유사도 검색과 별도로 관심 불량별 검색을 실험했다.

기준:

- `class`
- `class_location`
- `feature_key`

판단:

- class/class_location/feature_key 기준에서 random 대비 신호가 있었다.
- 위치가 중요한 target에서는 polar feature를 조건부로 쓸 수 있다.
- 단, 전체 유사 wafer 검색에는 polar feature를 넣지 않는다.

이 결정이 중요한 이유:

- 전체 유사도와 위치-aware 검색은 목적이 다르다.
- 전체 feature에 위치 feature를 섞으면 실제 wafer 검색이 위치 편향에 과하게 끌릴 수 있다.
- 관심 불량 검색에서는 위치 자체가 검색 조건이므로 위치 feature가 의미를 가진다.

관련 문서:

- [모델링 전략](modeling_strategy.md)
- [전문가 리뷰 절차](expert_review_protocol.md)

## 6. 단순 Resize Representation 비교

단순 resize representation도 비교했다.

아이디어:

- wafer image를 작은 grid로 줄인다.
- 줄인 벡터로 nearest-neighbor 검색을 수행한다.

판단:

- 전체 유사 wafer 검색의 대체재로는 부적합하다.
- 고해상도 wafer의 stby, edge/local/ring/scratch 의미 분해를 보장하지 않는다.
- 현업 리뷰에서 "왜 가까운지" 설명하기 어렵다.

따라서 resize-only representation은 주 경로에서 제외했다.

## 7. Patch Proposal

patch proposal은 edge/local/stby 후보 영역을 줄이는 보조 실험이었다.

역할:

- 전체 wafer를 다 보지 않고 의심 영역 후보를 만든다.
- edge/local/stby 리뷰 후보를 좁힌다.

판단:

- proposal recall을 최종 성능처럼 해석하지 않는다.
- proposal 튜닝을 계속 깊게 파는 것은 현재 목표와 맞지 않다.
- 주 경로는 라벨 없는 실제 wafer 처리 절차와 전문가 리뷰다.

## 8. Curve Proposal

curve proposal은 ring/center arc 후보를 찾기 위한 보조 실험이었다.

역할:

- radial profile과 curve 후보로 ring/arc 리뷰 후보를 만든다.
- ring radius/width mismatch 같은 실패 유형을 리뷰에서 볼 수 있게 한다.

판단:

- curve proposal은 ring/center arc 후보 축소용이다.
- 전체 유사 wafer 검색의 대체재가 아니다.
- scratch까지 rule로 억지 확장하지 않는다.

## 9. Scratch / Local 판단

scratch와 local은 현재 기준선에서 가장 조심해야 하는 계열이다.

관찰:

- scratch는 holdout에서 불안정하다.
- scratch는 길이, 방향, 연속성, 곡률이 중요하다.
- local은 작은 connected-component topology가 중요하다.
- wafer-level scalar feature만으로는 놓칠 수 있다.

현재 판단:

- scratch rule/proposal에 더 과투자하지 않는다.
- scratch 전용 line feature 또는 segmentation 쪽으로 분리한다.
- local은 connected-component morphology를 먼저 보고, 부족하면 segmentation 후보로 올린다.

이 판단은 "약한 것만 AI로 한다"가 아니다. 실제 리뷰에서 어떤 family가 약한지 확인한 뒤, 그 문제에 맞는 모델을 붙이겠다는 뜻이다. 전체를 AI로 덮는 선택지는 남아 있지만, 비교 기준과 리뷰 label 없이 먼저 대형 모델로 가면 실패 원인을 해석하기 어렵다.

## 10. Segmentation Smoke Test

AI 모델 배관도 최소 수준으로 확인했다.

구현된 것:

- synthetic-label segmentation dataset helper
- NumPy-only 1x1 sigmoid segmentation smoke training
- weighted BCE loss 연결
- multi-label mask target 연결

의미:

- 이것은 실사용 segmentation model이 아니다.
- U-Net, SegFormer, DINO embedding 같은 production model도 아직 아니다.
- 향후 scratch/local/stby overlap이 실제 리뷰에서 반복적으로 실패할 때 넘어갈 수 있는 최소 배관 검증이다.

## 11. 라벨 없는 실제 Wafer 처리 절차

현재 가장 중요한 산출물이다.

구현된 흐름:

```text
manifest
-> .npz load
-> schema validation
-> feature extraction
-> sanity JSON
-> feature drift summary
-> nearest-neighbor CSV
-> expert review template CSV
-> HTML report
```

보안 원칙:

- 실제 wafer raw image/array는 repo에 저장하지 않는다.
- 실제 file path, lot id, wafer id, tool id, recipe, chamber는 공유 산출물에 남기지 않는다.
- `sample_id`는 익명 id만 사용한다.

현재 이 절차는 synthetic smoke test를 통과했고, 실제 wafer 리뷰 직전 단계다.

관련 문서:

- [라벨 없는 실제 wafer 처리 절차](real_unlabeled_workflow.md)
- [Real Wafer 리뷰 체크리스트](real_wafer_review_checklist.md)

## 12. 전문가 리뷰

nearest-neighbor 결과는 바로 정답이 아니다. 전문가가 봐야 한다.

리뷰어가 판단하는 것:

- 같은 defect family인가
- 위치/clock이 비슷한가
- query의 주요 defect를 neighbor가 놓쳤는가
- mismatch라면 실패 유형이 무엇인가
- 다음 작업은 feature 보강인가, parser validation인가, segmentation 후보인가

집계 지표:

- `same_family_rate`
- `partial_match_rate`
- `accepted_match_rate`
- `missed_major_defect_rate`
- `query_topk_accept_rate`
- `retrieval_failure_mode_counts`
- `next_action_queue`

이 리뷰의 목적은 모델을 바로 크게 만드는 것이 아니라, 실제로 어떤 family가 약한지 근거를 쌓는 것이다.

관련 문서:

- [전문가 리뷰 절차](expert_review_protocol.md)

## 13. 외부 연구 참고 위치

외부 논문은 현재 방향을 점검하는 기준으로 사용했다.

- Iterative Cluster Harvesting for Wafer Map Defect Patterns
  - feature extraction, dimensionality reduction, clustering, manual labeling을 반복하는 흐름이 현재 `feature -> retrieval/grouping -> expert review` 전략과 맞다.
  - https://arxiv.org/abs/2404.15436

- Graph-Theoretic Spatial Filtering for Mixed-Type Wafer Bin Maps
  - 인접 defect chip의 spatial dependence를 이용한다. local/scratch morphology 보강의 참고선이다.
  - https://arxiv.org/abs/2006.13824

- WaferSegClassNet
  - mixed-type wafer defect classification/segmentation을 함께 다룬다. scratch/local/overlap을 segmentation 쪽으로 분리하는 판단을 뒷받침한다.
  - https://arxiv.org/abs/2207.00960

## 현재 남은 검증

다음은 합성 데이터로 더 밀어붙이는 일이 아니라 실제 wafer 리뷰로 확인해야 한다.

1. real wafer 5~20장을 `.npz`로 export한다.
2. manifest 형식과 stby/valid-test/wafer 밖 영역 처리가 맞는지 sanity 결과로 확인한다.
3. reference 대비 feature drift가 parser 문제인지, 실제 분포 차이인지 본다.
4. nearest-neighbor top-k를 전문가 리뷰 template으로 평가한다.
5. 실패 유형과 `next_action_queue`를 보고 feature 보강 또는 AI model 쪽 작업을 결정한다.

이 과정을 통과하면 현재 feature 기반 검색을 실제 triage용 최소 버전으로 유지할지, 특정 defect family에 segmentation/self-supervised model을 붙일지 판단할 수 있다.
