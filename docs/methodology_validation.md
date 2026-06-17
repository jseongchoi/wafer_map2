# 방법론 검증 정리

이 문서는 synthetic Fail Bit Map 데이터로 우리가 처음 원했던 방향이 유효한지 검증한 내용을 정리한다.

## 목표 정정

현재 1차 목표는 ANOVA가 아니다.

ANOVA는 나중에 공정 데이터, 설비 데이터, lot, recipe, chamber, test 조건 같은 메타데이터와 FBM feature를 조인했을 때 사용하는 후속 분석이다.

지금 가장 중요한 목표는 FBM 자체에서 정보를 잘 추출하는 것이다.

구체적으로는 다음을 먼저 해결해야 한다.

- wafer 안에서 어떤 불량 신호가 어디에 있는지 수치화한다.
- 비슷한 불량 패턴을 가진 wafer들을 그룹핑한다.
- 한 wafer 안에 여러 불량이 중첩되어도 주요 신호를 분리해서 본다.
- edge, ring, scratch, local, stby, shot-relative 같은 불량 관점을 다양하게 정의할 수 있어야 한다.
- 나중에 실제 unlabeled wafer가 들어왔을 때 feature 기반 유사맵 검색과 expert review가 가능해야 한다.

## 권장 방법론 순서

### 1. FBM Observable Feature Extraction

실제 wafer에서도 계산 가능한 feature만 사용한다.

- fail density
- grade-weighted severity
- radial profile
- angular profile
- edge minus center density
- edge chip outer-face minus inner-face density
- stby ratio
- shot-relative lower-left / edge contrast
- connected-component morphology

Synthetic ground-truth mask는 검증용 정답으로만 사용한다. 실제 wafer에는 mask가 없기 때문에 feature로 쓰면 안 된다.

### 2. 유사 wafer 검색과 그룹핑

Observable feature vector를 표준화한 뒤 wafer 간 거리를 계산한다.

이 단계의 목적은 다음과 같다.

- 비슷한 edge-heavy wafer끼리 모이는가?
- 비슷한 shot-relative defect wafer끼리 모이는가?
- stby-heavy wafer가 단순 Grade 7 high fail wafer와 분리되는가?
- ring, scratch, local처럼 다른 공간 구조가 feature 공간에서 어느 정도 구분되는가?

이것이 현재 가장 중요한 Phase 2 검증이다.

### 3. 다양한 불량 관점 정의

현업에서는 불량 정의가 고정되어 있지 않다.

따라서 하나의 모델 class만 믿기보다, 여러 관점의 score와 feature를 같이 가져가야 한다.

예:

- radial edge score
- localized edge score
- shot-relative lower-left score
- shot edge-band score
- stby origin-hidden score
- local blob score
- scratch/ring directional score

이 feature들이 쌓이면 wafer clustering, nearest-neighbor review, rule-based triage, downstream model 학습에 모두 사용할 수 있다.

### 4. 공간 분해 모델

Scratch나 local blob처럼 작은 위치 기반 defect는 wafer-level feature만으로는 한계가 있다.

이 경우 다음 단계에서 synthetic mask를 이용해 multi-label segmentation baseline을 만든다.

목표 출력은 다음 형태가 된다.

```text
12시 방향: ring/arc 계열 신호
3시 edge: edge-heavy defect
5시 방향: local blob + stby-hidden core
shot coordinate: lower-left 반복 defect 가능성
```

### 5. 공정데이터 조인 이후 통계 분석

ANOVA나 통계 검정은 이 단계에서 의미가 있다.

즉, FBM feature를 먼저 잘 뽑고 나서 다음 데이터와 붙인다.

- process step
- equipment/tool
- chamber
- recipe
- lot
- product
- test condition

그 다음에 예를 들어 "shot_relative_score가 특정 exposure tool에서 유의하게 높은가?" 같은 질문을 던진다.

## 첫 검증 결과

첫 methodology probe는 synthetic batch에서 label variation을 만들고, synthetic mask를 정답으로만 사용해 observable feature의 유효성을 확인했다.

산출물:

- `outputs/reports/methodology_validation_report.html`
- `outputs/reports/methodology_validation_metrics.json`
- `outputs/reports/methodology_validation_features.csv`

결과 요약:

- Observable feature만 사용: PASS
- 평가 가능 class: scratch, ring, edge, local, shot_grid, stby_pattern
- 유사 wafer 검색 label-Jaccard lift: random baseline보다 높음
- ring, edge, shot_grid, stby_pattern은 wafer-level feature만으로도 강한 신호가 있음
- scratch, local은 wafer-level feature만으로는 약해서 segmentation 또는 morphology feature 보강이 필요함

## 현재 해석

현재 방향은 유효하다.

- FBM observable feature extraction은 지금 바로 유효하다.
- 유사 wafer 검색과 그룹핑은 지금 바로 유효하다.
- 다양한 불량 관점의 score를 정의하는 방향이 중요하다.
- ANOVA는 지금의 핵심이 아니라, 공정데이터와 조인된 이후의 후속 분석이다.
- AutoEncoder 단독 접근은 1차 방법론으로 적합하지 않다. Stby missing chip이나 전체 reconstruction error에 과민하게 반응하고, 중첩 불량의 의미 분해에는 약할 수 있다.

## 초기 CPU-only 그룹핑 안정성 검증

사용자가 노트북 환경에서 우선 검증할 수 있도록 GPU가 필요 없는 안정성 검증을 추가했다.

산출물:

- `outputs/reports/fbm_grouping_report.html`
- `outputs/reports/fbm_grouping_stability_report.html`
- `outputs/reports/fbm_grouping_parameter_sweep_report.html`
- `outputs/reports/fbm_grouping_features.csv`

초기 36장 pilot 핵심 결과:

- 36장 pilot batch에서 observable feature 28개만 사용했다.
- 유사 wafer 검색은 top-3, top-5, top-10 모두 random baseline보다 높았다.
- top-3 label-Jaccard lift는 약 1.45x, top-5는 약 1.41x, top-10은 약 1.30x였다.
- K=4~6 cluster 설정은 반복 feature subset/noise 검증에서 안정성 기준을 통과했다.
- CPU pilot 기준 권장 cluster 수는 K=4이다.
- K를 너무 크게 잡으면 cluster 내부가 흔들릴 수 있으므로, 초기 실무 적용은 hard cluster label보다 coarse group, nearest-neighbor review, defect score ranking을 함께 쓰는 것이 안전하다.

초기 결론:

1. FBM feature는 유사 wafer 검색과 coarse grouping에 쓸 수 있는 신호를 담고 있다.
2. 지금 단계에서 가장 실용적인 output은 “이 wafer와 유사한 wafer 상위 N개”, “어떤 defect score가 높은가”, “coarse group은 어디인가”이다.
3. Scratch/local처럼 작은 공간 불량은 wafer-level feature만으로 약할 수 있으므로 morphology feature와 segmentation baseline으로 보강해야 한다.
4. 다음 검증으로 feature family ablation이 필요했다. radial, angular, edge, shot, stby feature를 하나씩 제거했을 때 관련 defect score와 retrieval 품질이 예상대로 떨어지는지 확인하는 방향이었다.

## 초기 Feature family ablation 결과

산출물:

- `outputs/reports/fbm_feature_ablation_report.html`
- `outputs/reports/fbm_feature_ablation_metrics.json`
- `outputs/figures/fbm_feature_ablation.png`
- `outputs/figures/fbm_neighbor_gallery.png`

첫 ablation에서는 `stby`와 `shot` feature가 실질적으로 기여했지만, `edge` feature는 중첩 defect 상황에서 전체 retrieval에 잡음처럼 섞이는 경향을 보였다.

이에 따라 feature extractor에 다음 보강을 추가했다.

- ring radial peak contrast / width
- scratch angular peak contrast / width
- local hotspot peak / top-3 / spread / count ratio
- local component compactness / triangle score
- scratch component elongation / span
- edge chip peak contrast
- edge sector peak contrast / concentration / localized sector ratio

보강 후 36장 pilot 핵심 결과:

- observable feature 수는 검증용 mask feature를 제외하고 40개가 되었다.
- 전체 observable feature baseline top-5 유사검색 lift는 약 1.46x로 상승했다.
- top-3 lift는 약 1.47x, top-5 lift는 약 1.46x, top-10 lift는 약 1.31x였다.
- CPU pilot 권장 cluster 수는 계속 K=4이다.
- `edge` family를 제거하면 전체 lift가 약 0.08 떨어져 가장 큰 하락을 보였다.
- 36장 pilot에서는 `shot`, `stby`, `angular`, `local_morphology`, `radial`, `ring_scratch_morphology`, `global` 제거도 대체로 baseline 대비 lift를 낮췄다.

해석:

- 새 feature 보강 이후 edge 정보축은 더 이상 단순 잡음이 아니라 retrieval에 기여하는 feature family가 되었다.
- shot-relative feature는 여전히 강한 정보축이다.
- stby feature는 missing-test chip 자체가 의미 있는 패턴으로 나타날 때 중요한 역할을 한다.
- 155장 scale pilot에서는 angular 제거 후 lift가 오히려 상승하고 local morphology 제거 영향도 작았다. 따라서 angular/local morphology는 최종 feature가 아니라 재검토 대상이다.
- scratch/ring/local은 개선됐지만 class별 lift 기준으로는 아직 shot/stby/edge보다 약하다. 다음 단계에서는 morphology feature를 더 보강하거나 synthetic mask 기반 segmentation baseline으로 보완해야 한다.

## Scale pilot 결과

36장 pilot에서 보인 신호가 작은 batch에만 맞는지 확인하기 위해 155장 scale pilot을 추가했다. 목표는 200장이었지만, 노트북 CPU runtime limit 때문에 완성된 155장을 scale 검증 기준으로 고정했다.

산출물:

- [Scale FBM 그룹핑 리포트](../outputs/reports/fbm_grouping_scale_report.html)
- [Scale FBM 안정성 리포트](../outputs/reports/fbm_grouping_scale_stability_report.html)
- [Scale FBM 파라미터 스윕 리포트](../outputs/reports/fbm_grouping_scale_parameter_sweep_report.html)
- [Scale FBM feature family ablation 리포트](../outputs/reports/fbm_feature_ablation_scale_report.html)

핵심 결과:

- sample 수는 155장이다.
- morphology 보강 후 observable feature 수는 50개다.
- top-5 유사검색 lift는 약 1.36x다.
- bootstrap 95% CI는 약 1.31x ~ 1.41x다.
- permutation p-value는 약 0.001이다.
- 현재 scale 재계산 cluster 수는 K=4다.
- feature family ablation에서 shot, stby, edge 제거가 전체 lift를 가장 크게 낮췄다.
- 관심 기준별 retrieval에서 edge 약 1.61x, shot 약 2.02x, ring 약 1.58x, stby 약 1.24x, local 약 1.11x, scratch 약 1.10x가 관측됐다.

## Holdout stress 결과

다른 seed/class prior/grade threshold 조건의 120장 holdout에서도 같은 observable feature contract를 적용했다.

핵심 결과:

- sample 수는 120장이다.
- observable feature 수는 50개다.
- top-5 유사검색 lift는 약 1.40x다.
- bootstrap 95% CI는 약 1.32x ~ 1.48x다.
- permutation p-value는 약 0.002다.
- 관심 기준별 retrieval에서 edge 약 2.64x, ring 약 1.52x, stby 약 1.48x, local 약 1.33x가 유지됐다.
- shot은 약 1.23x로 낮아져 조건 변화에 민감하다.
- scratch는 약 0.92x로 무너져 다음 단계 보강 대상이다.

현재 결론:

1. FBM feature 기반 유사 wafer 검색과 coarse grouping은 scale batch에서도 random baseline보다 유망하다.
2. 초기 36장보다 lift는 낮아졌지만, 더 현실적인 batch에서 유지된 수치이므로 현재 의사결정 기준은 155장 scale pilot이다.
3. shot/edge/ring/stby 계열 score는 실무 triage 후보로 유지한다.
4. local은 morphology feature를 expert review로 확인한다.
5. scratch 계열은 wafer-level/morphology feature만으로 약하므로 synthetic segmentation baseline 또는 scratch-specific representation으로 넘어가는 것이 맞다.
6. ANOVA는 여전히 현재 핵심이 아니라, 공정 metadata와 feature table이 조인된 뒤의 후속 분석이다.
