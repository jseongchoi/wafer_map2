# AI Data Science Validation Audit

## 목적

이 문서는 현재 WaferMap 프로젝트가 원래 목표대로 검증되고 있는지 AI 데이터사이언스 관점에서 점검한 결과다.

원래 목표는 다음이다.

```text
1채널 FBM에서 실제 wafer에도 계산 가능한 정보를 추출한다.
비슷한 불량 패턴의 wafer를 찾고 그룹핑한다.
wafer 내부의 여러 결함 관점에 대해 score를 만든다.
작은/중첩 결함은 이후 segmentation 또는 morphology 보강으로 확장한다.
ANOVA는 공정 metadata 조인 이후의 후속 단계로 둔다.
```

## 현재 판정

현재 진행 방향은 원래 목적과 맞다.

다만 성공 범위는 분리해서 해석해야 한다.

- 유사 wafer 검색: 현재 기준으로 유효하다.
- coarse grouping: 현재 기준으로 유효하다.
- shot/edge/stby 계열 score: triage 후보로 유효하다.
- ring 계열: 일부 유효하지만 real expert review가 필요하다.
- local 계열: morphology 보강 후 일부 유효하지만 expert review가 필요하다.
- scratch 계열: 아직 충분하지 않다.
- root cause 통계 분석: 아직 단계가 아니다.

## 현재 근거

155장 scale pilot 기준:

- observable feature 수: 50개
- top-5 유사검색 lift: 약 1.36x
- top-5 lift bootstrap 95% CI: 약 1.31x ~ 1.41x
- top-5 permutation p-value: 약 0.001
- 현재 scale 재계산 K: 4
- 관심 기준별 lift: edge 약 1.61x, shot 약 2.02x, ring 약 1.58x, stby 약 1.24x, local 약 1.11x, scratch 약 1.10x

120장 holdout stress 기준:

- observable feature 수: 50개
- top-5 유사검색 lift: 약 1.40x
- top-5 lift bootstrap 95% CI: 약 1.32x ~ 1.48x
- permutation p-value: 약 0.002
- 관심 기준별 lift: edge 약 2.64x, ring 약 1.52x, stby 약 1.48x, local 약 1.33x, shot 약 1.23x, scratch 약 0.92x

Feature family ablation 해석:

- shot 제거가 전체 lift를 가장 크게 낮춘다.
- stby와 edge 제거도 lift를 낮춘다.
- local morphology는 전체 retrieval보다 관심 기준 검색에서 의미가 더 잘 보인다.
- angular feature는 일부 조건에서 잡음일 수 있어 재검토 대상이다.
- scratch는 morphology feature를 추가해도 holdout에서 불안정하다.

## 잘 된 점

- 실제 wafer가 없는 보안 제약을 synthetic generator와 expert review loop로 우회했다.
- Stby Fail Chip을 Grade 7과 분리했다.
- `valid_test_mask`, `stby_mask`, `pattern_masks`를 별도로 유지한다.
- 실제 inference feature는 observable-only로 분리했다.
- `*_mask_ratio` synthetic oracle field는 검증 전용으로 격리했다.
- GPU 없이도 CPU 기반 similarity, stability, sweep, ablation 검증을 수행했다.

## 과장하면 안 되는 점

- 현재 결과는 synthetic scale pilot에서 유망한 baseline을 확인한 것이다.
- real wafer에서 같은 성능이 나온다는 검증은 아직 없다.
- shot feature는 generator와 extractor가 유사한 shot-relative 가정을 공유하므로 template 재발견 위험이 있다.
- local/stby label prevalence가 높아 전체 lift만으로 class별 성능을 해석하기 어렵다.
- 0.03 수준의 ablation drop은 bootstrap confidence interval 또는 permutation test 없이는 강하게 해석하면 안 된다.
- 현재 defect score는 calibrated probability가 아니라 feature proxy다.

## 수정한 점

- scale pilot 문서에 과장 방지 문구를 추가했다.
- 전체 lift와 class별 lift를 분리해서 해석하도록 validation protocol을 보강했다.
- scratch/local 보강이 필요한 이유를 modeling strategy에 명시했다.
- angular feature가 항상 이롭지 않을 수 있음을 명시했다.
- shot generator anchor를 lower-left/bottom-edge/left-edge 중심으로 제한해 feature extractor와 도메인 가정을 맞췄다.
- stby origin-hidden 보고에서 seeded origin chip과 latent-weighted random chip을 분리하도록 수정했다.
- retrieval confidence report를 추가해 bootstrap confidence interval과 permutation p-value를 산출했다.

## 남은 리스크

- synthetic label 기준 lift가 real wafer 성능을 보장하지 않는다.
- local defect는 connected-component morphology로 일부 잡히지만 실제 expert review가 필요하다.
- scratch와 ring은 실제 공정 패턴에서 서로 섞여 보일 수 있다.
- shot-relative feature가 synthetic generator parameter에 과적합됐는지 real sanity check가 필요하다.
- 현재 K는 hard label이 아니라 review용 coarse grouping으로 써야 한다.

## 다음 검증 순서

1. Real-unlabeled workflow 설계
   - 실제 wafer는 repo 밖에 둔다.
   - observable feature만 추출한다.
   - nearest-neighbor gallery와 defect score report만 생성한다.

2. Expert review protocol 적용
   - 유사맵 top-k가 실제로 같은 계열인지 사용자가 평가한다.
   - class별로 shot, edge, stby, scratch, ring, local을 따로 본다.

3. Scratch/local 보강
   - local은 현재 morphology feature를 expert review로 검증한다.
   - scratch는 synthetic-label segmentation baseline 또는 curve/arc-specific detector로 넘어간다.
   - 작은 droplet, double droplet, triple triangle, stby-hidden origin을 별도 recall 대상으로 둔다.

4. Robustness 보강
   - top-k lift의 bootstrap confidence interval과 permutation test는 scale pilot 기준으로 추가했다.
   - feature tuning batch와 holdout synthetic batch를 분리한다.
   - geometry, grade threshold, class prior, shot layout을 바꾼 stress test를 추가한다.

5. Process metadata statistics
   - 공정/설비/lot/recipe/chamber/test metadata가 조인된 뒤에 수행한다.
   - 이때 ANOVA 또는 더 적합한 통계 검정을 선택한다.

## 결론

현재 프로젝트는 원래 목적에서 벗어나지 않았다. 오히려 AutoEncoder나 무거운 모델로 바로 가지 않고, 해석 가능한 observable feature baseline을 먼저 세운 점이 맞는 방향이다.

다음 단계는 더 많은 모델을 붙이는 것이 아니라, real-unlabeled 실행 경로를 만들고 expert review로 synthetic-to-real gap을 확인하는 것이다.
