# Experiment History

이 문서는 지금까지 어떤 기법을 실험했고, 어떤 판단으로 현재 방향을 선택했는지 설명한다.

짧게 읽고 싶으면 [Project Overview](project_overview.md)와 [Real Wafer Review Checklist](real_wafer_review_checklist.md)를 먼저 본다. 이 문서는 "왜 그렇게 결정했는가"를 확인할 때 읽는다.

## 핵심 결론

현재까지의 결론은 다음이다.

```text
observable feature + nearest-neighbor retrieval은 1차 baseline으로 유효하다.
resize-only representation은 global retrieval 대체재로 부족하다.
patch/curve proposal은 최종 솔루션이 아니라 review 후보 축소용 보조 도구다.
scratch/local은 rule/proposal 튜닝보다 morphology 또는 segmentation track으로 분리하는 편이 맞다.
real wafer 최종 판단은 expert review loop를 통과해야 한다.
```

따라서 현재 주 경로는 다음이다.

```text
secure semantic .npz
-> sanity / feature drift
-> compact observable feature
-> nearest-neighbor retrieval
-> expert review template
-> failure mode / next action backlog
```

## 1. 문제 정의와 데이터 계약

처음부터 목표는 단일 불량 분류 모델이 아니었다.

목표는 1채널 고해상도 Wafer Fail Bit Map에서 실제 wafer에도 계산 가능한 feature 표현을 만들고, 그 표현으로 유사 wafer 검색, 관심 defect retrieval, defect score table, downstream task를 가능하게 하는 것이다.

그래서 먼저 semantic tensor 계약을 잡았다.

- `severity`: Grade 0~7
- `wafer_mask`: wafer 내부와 none-wafer 구분
- `valid_test_mask`: 실제 test 여부
- `stby_mask`: stby fail chip 영역
- `chip_index`: 권장 die/chip id

중요한 결정:

- stby는 Grade 7로 섞지 않는다.
- none-wafer와 in-wafer Grade 0은 `wafer_mask`로 구분한다.
- synthetic oracle인 `pattern_masks`, `pattern_intensity`, `label_*`, `*_mask_ratio`는 검증용이다.
- real inference feature에는 synthetic oracle을 넣지 않는다.

관련 문서:

- [Data Schema](data_schema.md)
- [Pattern Taxonomy](pattern_taxonomy.md)

## 2. Synthetic FBM Generator

실제 wafer raw data를 repo에 둘 수 없기 때문에, 방법론 검증을 위해 synthetic generator를 만들었다.

생성한 계열:

- Grade 0~7 severity
- none-wafer / valid-test / stby semantic 분리
- edge gradient와 localized edge sector
- shot-relative 반복 패턴
- local hotspot
- stby-origin 또는 stby가 origin을 가리는 형태
- ring / partial ring / center arc
- scratch-like arc 또는 radial line
- mixed overlap

이 generator의 역할은 "실제와 똑같은 wafer 생산기"가 아니라 다음을 검증하는 것이다.

- feature extractor가 semantic 계약을 지키는가
- oracle label 없이 observable feature만으로 유사 wafer retrieval 신호가 생기는가
- defect family별로 어떤 feature가 강하고 약한가
- real review 전에 pipeline이 끝까지 연결되는가

주의할 점:

- synthetic 성능은 real 성능 인증이 아니다.
- generator와 feature extractor가 같은 가정을 공유할 수 있으므로 real expert review가 필요하다.

관련 문서:

- [Validation Protocol](validation_protocol.md)
- [Roadmap](roadmap.md)

## 3. Observable Feature Baseline

첫 baseline은 사람이 해석 가능한 observable feature다.

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

Global retrieval feature 계약:

```text
compact observable feature 50개
```

Global retrieval에서 제외하는 것:

```text
label_*
*_mask_ratio
pattern_masks
pattern_intensity
polar_*
stby_polar_*
```

`polar_*`, `stby_polar_*`는 위치-aware retrieval에서만 조건부로 쓴다. 예를 들어 `class_location`, `feature_key`처럼 위치 자체가 질의 조건인 경우다.

구현 위치:

- `src/wafermap/features/selection.py`
- `src/wafermap/features/wafer_vector.py`

## 4. Global Similar Wafer Retrieval

실험 방법:

1. synthetic feature CSV를 만든다.
2. compact observable feature만 고른다.
3. feature를 standardize한다.
4. Euclidean nearest-neighbor를 계산한다.
5. synthetic label Jaccard는 검증 metric으로만 사용한다.
6. random neighbor 대비 top-k lift를 본다.

확인된 결과:

- scale 155장 top-k retrieval lift 약 1.36x
- holdout 120장 top-k retrieval lift 약 1.40x

해석:

- observable feature만으로도 "비슷한 wafer를 더 가깝게 찾는 신호"가 있다.
- 이것은 최종 real wafer 성능 인증이 아니라 1차 baseline 후보의 근거다.
- scratch/local처럼 작은 구조 결함은 전체 lift가 좋아도 별도로 봐야 한다.

공통화한 구현:

- `src/wafermap/evaluation/nearest.py`
- `src/wafermap/features/selection.py`

## 5. Interest-Conditioned Retrieval

Global retrieval과 별도로 관심 defect별 retrieval을 실험했다.

기준:

- `class`
- `class_location`
- `feature_key`

판단:

- class/class_location/feature_key 기준에서 random 대비 신호가 있었다.
- 위치가 중요한 target에서는 polar feature를 조건부로 쓸 수 있다.
- 단, global retrieval에는 polar feature를 넣지 않는다.

이 결정이 중요한 이유:

- global 유사도와 위치-aware 검색은 목적이 다르다.
- global feature에 위치 feature를 섞으면 실제 wafer 검색에서 위치 편향이 과해질 수 있다.
- 관심 defect retrieval에서는 위치 자체가 검색 조건이므로 위치 feature가 의미를 가진다.

관련 문서:

- [Modeling Strategy](modeling_strategy.md)
- [Expert Review Protocol](expert_review_protocol.md)

## 6. Resize-Only Representation Benchmark

resize-only representation도 비교했다.

아이디어:

- wafer image를 작은 grid로 resize한다.
- resize된 벡터로 nearest-neighbor retrieval을 수행한다.

판단:

- global retrieval 대체재로는 부적합하다.
- 고해상도 wafer semantic, stby, edge/local/ring/scratch의 의미 분해를 직접 보장하지 않는다.
- 현업 review에서 "왜 가까운지" 설명하기 어렵다.

따라서 resize-only는 주 경로에서 제외했다.

## 7. Patch Proposal

patch proposal은 edge/local/stby 후보 영역을 줄이는 보조 실험이었다.

역할:

- 전체 wafer를 다 보지 않고 의심 영역 후보를 만든다.
- edge/local/stby review 후보를 좁힌다.

판단:

- proposal recall을 최종 성능처럼 해석하지 않는다.
- proposal 튜닝을 계속 깊게 파는 것은 현재 목표와 맞지 않다.
- 주 경로는 여전히 real-unlabeled workflow와 expert review loop다.

## 8. Curve Proposal

curve proposal은 ring/center arc 후보를 찾기 위한 보조 실험이었다.

역할:

- radial profile과 curve 후보로 ring/arc review 후보를 만든다.
- ring radius/width mismatch 같은 failure mode를 review에서 볼 수 있게 한다.

판단:

- curve proposal은 ring/center arc 후보 축소용이다.
- global retrieval 대체재가 아니다.
- scratch까지 rule로 억지 확장하지 않는다.

## 9. Scratch / Local 판단

scratch와 local은 현재 baseline에서 가장 조심해야 하는 계열이다.

관찰:

- scratch는 holdout에서 불안정하다.
- scratch는 길이, 방향, 연속성, 곡률이 중요하다.
- local은 작은 connected-component topology가 중요하다.
- wafer-level scalar feature만으로는 놓칠 수 있다.

현재 판단:

- scratch rule/proposal에 더 과투자하지 않는다.
- scratch-specific line representation 또는 segmentation track으로 분리한다.
- local은 connected-component morphology를 먼저 보고, 부족하면 segmentation 후보로 올린다.

이 판단은 "약한 것만 AI로 한다"가 아니라, real review에서 실제로 약한 family를 확인한 뒤 필요한 모델을 붙이겠다는 의미다. 전체를 AI로 덮는 선택지는 남아 있지만, 비교 기준과 review label 없이 먼저 대형 모델로 가면 실패 원인을 해석하기 어렵다.

## 10. Segmentation Smoke

AI 모델 배관도 최소 수준으로 확인했다.

구현된 것:

- synthetic-label segmentation dataset helper
- NumPy-only 1x1 sigmoid segmentation smoke training
- weighted BCE loss 연결
- multi-label mask target 연결

의미:

- 이것은 실사용 segmentation model이 아니다.
- U-Net, SegFormer, DINO embedding 같은 production model도 아직 아니다.
- 향후 scratch/local/stby overlap이 real review에서 반복적으로 실패할 때 넘어갈 수 있는 최소 배관 검증이다.

## 11. Real-Unlabeled Workflow

현재 가장 중요한 산출물이다.

구현된 흐름:

```text
manifest
-> semantic .npz load
-> schema validation
-> observable feature extraction
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

현재 이 workflow는 synthetic smoke test를 통과했고, real wafer review 직전 단계다.

관련 문서:

- [Real-Unlabeled Workflow](real_unlabeled_workflow.md)
- [Real Wafer Review Checklist](real_wafer_review_checklist.md)

## 12. Expert Review Loop

nearest-neighbor 결과는 바로 "정답"이 아니다. 전문가가 봐야 한다.

리뷰어가 판단하는 것:

- 같은 defect family인가
- 위치/clock이 비슷한가
- query의 주요 defect를 neighbor가 놓쳤는가
- mismatch라면 failure mode가 무엇인가
- 다음 action은 feature 보강인가, parser validation인가, segmentation 후보인가

집계 지표:

- `same_family_rate`
- `partial_match_rate`
- `accepted_match_rate`
- `missed_major_defect_rate`
- `query_topk_accept_rate`
- `retrieval_failure_mode_counts`
- `next_action_queue`

이 loop의 목적은 모델을 바로 크게 만드는 것이 아니라, 어떤 family가 실제로 약한지 근거를 쌓는 것이다.

관련 문서:

- [Expert Review Protocol](expert_review_protocol.md)

## 13. 외부 연구 참고 위치

외부 논문은 현재 방향을 검토하는 기준으로 사용했다.

- Iterative Cluster Harvesting for Wafer Map Defect Patterns
  - feature extraction, dimensionality reduction, clustering, manual labeling을 반복하는 흐름이 현재 `feature -> retrieval/grouping -> expert review` 전략과 맞다.
  - https://arxiv.org/abs/2404.15436

- Graph-Theoretic Spatial Filtering for Mixed-Type Wafer Bin Maps
  - 인접 defect chip의 spatial dependence를 이용한다. local/scratch morphology 보강의 참고선이다.
  - https://arxiv.org/abs/2006.13824

- WaferSegClassNet
  - mixed-type wafer defect classification/segmentation을 함께 다룬다. scratch/local/overlap을 segmentation track으로 분리하는 판단을 뒷받침한다.
  - https://arxiv.org/abs/2207.00960

## 현재 남은 검증

다음은 synthetic으로 더 밀어붙이는 일이 아니라 real wafer review로 확인해야 한다.

1. real wafer 5~20장을 semantic `.npz`로 export한다.
2. manifest schema와 stby/valid-test/none-wafer 계약을 sanity로 확인한다.
3. reference 대비 feature drift가 parser 문제인지, real distribution 차이인지 본다.
4. nearest-neighbor top-k를 expert review template으로 평가한다.
5. failure mode와 `next_action_queue`를 보고 feature 보강 또는 AI model track을 결정한다.

이 과정을 통과하면 current observable retrieval baseline을 real triage MVP로 유지할지, 특정 defect family에 segmentation/self-supervised model을 붙일지 판단할 수 있다.
