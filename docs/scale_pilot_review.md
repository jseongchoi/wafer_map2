# Scale Pilot Review

핵심 결론: 155장 scale pilot에서 morphology 보강 후 observable feature 50개 기준 top-5 유사검색 lift 약 1.36x를 유지했다. shot/edge/ring/stby feature 기여는 확인됐고, local은 관심 기준 검색에서 일부 개선됐다. scratch는 scale에서도 약하고 holdout에서 더 불안정하므로 다음 단계에서 segmentation 또는 scratch-specific representation으로 넘긴다.

## 목적

36장 pilot에서 확인한 FBM observable feature 기반 유사 wafer 검색/그룹핑 방법론이 더 큰 batch에서도 유지되는지 확인한다.

## Batch 상태

- 목표 count: 200장
- 완료 count: 155장
- 생성 위치: `data/synthetic/fbm_grouping_scale_pilot`
- config: `configs/synth/presets/fbm_grouping_scale_pilot.json`
- 상태: 노트북 CPU runtime limit 때문에 200장 완주는 중단하고, 완성된 155장을 scale pilot으로 고정했다.
- validation: 155장 모두 internal validation OK

운영 메모:

- full preview를 포함한 200장 생성은 노트북 CPU에서 너무 오래 걸렸다.
- `--no-preview`, `--resume` 옵션을 `scripts/generate_synthetic.py`에 추가했다.
- 나머지 45장을 이어서 만들 수는 있지만, 현재 방법론 검증에는 155장도 기존 36장보다 충분히 큰 scale check다.

## 산출물

리포트:

- [Scale FBM 그룹핑 리포트](../outputs/reports/fbm_grouping_scale_report.html)
- [Scale FBM 안정성 리포트](../outputs/reports/fbm_grouping_scale_stability_report.html)
- [Scale FBM 파라미터 스윕 리포트](../outputs/reports/fbm_grouping_scale_parameter_sweep_report.html)
- [Scale FBM feature family ablation 리포트](../outputs/reports/fbm_feature_ablation_scale_report.html)
- [Scale retrieval confidence 리포트](../outputs/reports/fbm_retrieval_confidence_scale_report.html)
- [Scale interest-based retrieval 리포트](../outputs/reports/fbm_interest_retrieval_scale_report.html)

그림:

- [Scale PCA](../outputs/figures/fbm_grouping_scale_pca.png)
- [Scale co-association](../outputs/figures/fbm_grouping_scale_coassociation.png)
- [Scale parameter sweep](../outputs/figures/fbm_grouping_scale_parameter_sweep.png)
- [Scale feature ablation](../outputs/figures/fbm_feature_ablation_scale.png)
- [Scale nearest-neighbor gallery](../outputs/figures/fbm_neighbor_gallery_scale.png)
- [Scale interest-based nearest-neighbor gallery](../outputs/figures/fbm_interest_neighbor_gallery_scale.png)

## 핵심 결과

Scale grouping 결과:

- sample count: 155
- observable feature count: 50
- cluster count: 4
- top-5 유사검색 label-Jaccard lift: 약 1.36x
- neighbor label-Jaccard: 약 0.60
- random pair label-Jaccard: 약 0.44

Grouping stability 결과:

- same-cluster co-association: 약 0.60
- different-cluster co-association: 약 0.21
- separation: 약 0.39
- nearest-neighbor overlap: 약 0.77
- acceptance: 모두 PASS

Parameter sweep 결과:

- top-3 lift: 약 1.39x
- top-5 lift: 약 1.36x
- top-10 lift: 약 1.33x
- 권장 coarse K: 3
- K=3, K=4는 안정성 통과
- K=5 이상은 현재 batch에서 상대적으로 과분할 경향

Feature ablation 결과:

- baseline lift: 약 1.36x
- class별 lift:
  - scratch: 약 1.07x
  - ring: 약 1.34x
  - edge: 약 1.30x
  - local: 약 1.00x
  - shot_grid: 약 1.90x
  - stby_pattern: 약 1.12x
- 전체 lift drop 상위:
  - shot 제거: 약 0.05 하락
  - edge 제거: 약 0.02 하락
  - global 제거: 약 0.01 하락
- 주의:
  - local morphology 제거는 전체 retrieval lift를 거의 낮추지 않았지만, local class lift는 낮춘다.
  - angular 제거는 전체 lift를 오히려 높여, 현재 angular feature가 scratch/ring 구분에 항상 이로운 것은 아니다.
  - 따라서 feature set은 “현재 baseline”이지 고정된 최종 feature schema가 아니다.

Retrieval confidence 결과:

- top-5 lift: 약 1.36x
- bootstrap 95% CI: 약 1.31x ~ 1.41x
- permutation p-value: 약 0.001
- class별 top-5 precision lift:
  - scratch: 약 1.07x
  - ring: 약 1.34x
  - edge: 약 1.30x
  - local: 약 1.00x
  - shot_grid: 약 1.90x
  - stby_pattern: 약 1.12x

Interest-based retrieval 결과:

- overall: 약 1.36x
- edge_focus: 약 1.61x
- shot_focus: 약 2.02x
- stby_focus: 약 1.24x
- ring_focus: 약 1.58x
- scratch_focus: 약 1.10x
- local_focus: 약 1.11x

## 해석

36장 pilot보다 lift는 낮아졌지만 더 현실적인 수치다. 중요한 점은 scale batch에서도 random baseline보다 일관되게 높고, stability와 top-k sweep이 기준을 통과했다는 것이다.

현재 방법론은 다음 용도에 유효하다.

- 유사 wafer nearest-neighbor 검색
- coarse grouping
- shot/stby/edge 계열 score 기반 triage
- expert review용 visual gallery 생성

아직 약한 부분:

- local defect는 morphology 보강으로 일부 좋아졌지만, scale에서는 아직 약하다.
- scratch는 morphology feature를 추가해도 충분하지 않아 segmentation 또는 curve/arc detector 보강이 필요하다.
- ring은 1.26x 수준으로 잡히지만, 실제 데이터에서 ring과 spin scratch가 섞이면 별도 검증이 필요하다.
- 현재 lift는 synthetic validation label 기준이므로 real wafer expert review 없이는 실무 성능으로 과장하면 안 된다.
- 200장 full generation은 generator 성능 또는 preview 전략 최적화가 필요하다.

## 다음 결정

다음 단계로 넘어갈 수 있다.

권장 순서:

1. Scale batch 155장을 기준으로 real-unlabeled inference workflow 설계
2. local morphology 결과를 expert review에 올리고, scratch는 synthetic segmentation baseline 또는 scratch-specific representation 준비
3. generator runtime 최적화 후 200장 이상 full batch 재시도
