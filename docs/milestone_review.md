# Milestone Review

## 검토 목적

이 문서는 현재 이정표에서 잠깐 멈춰, 지금까지의 작업이 원래 문제 정의와 맞게 진행됐는지, 코드와 산출물이 과도하게 지저분해졌는지, 다음 단계로 넘어가도 되는지 점검한 기록이다.

## 1. 원래 목적과의 정렬

원래 목적은 다음이었다.

```text
고해상도 1채널 Fail Bit Map에서
여러 불량 신호가 중첩된 상태를 분석하고,
유사한 wafer를 찾고,
wafer 내부의 주목 defect를 위치/종류/강도 관점으로 정량화한다.
```

현재 작업은 이 목적에 맞게 진행되고 있다.

- Synthetic FBM generator는 실제 데이터를 repo에 두지 못하는 제약을 우회하기 위한 수단이다.
- 현재 핵심은 ANOVA가 아니라 FBM observable feature extraction이다.
- 유사 wafer 검색과 coarse grouping은 이미 CPU-only pilot으로 검증했다.
- Synthetic mask는 학습/검증용 정답으로만 사용하고, real wafer inference path에는 넣지 않는 구조를 유지했다.
- Stby Fail Chip은 Grade 7과 분리해 `stby_mask`, `valid_test_mask`로 관리한다.

## 2. 현재 방법론 상태

현재 pipeline은 다음 순서다.

```text
synthetic FBM 생성
-> visual/expert realism check
-> observable feature extraction
-> nearest-neighbor similarity search
-> coarse grouping
-> grouping stability
-> parameter sweep
-> feature family ablation
-> visual neighbor gallery
```

아직 GPU 기반 segmentation이나 self-supervised encoder 단계로 가지 않았다. 이는 의도된 결정이다. 사용자의 노트북 환경에서는 CPU로 방법론의 유효성을 먼저 확인하는 것이 맞다.

## 3. 최신 핵심 수치

현재 의사결정 기준은 `fbm_grouping_scale_pilot` 155장 결과다.

Scale pilot 기준:

- sample 수: 155장
- observable feature 수: 50개
- top-5 유사 wafer 검색 lift: 약 1.36x
- bootstrap 95% CI: 약 1.31x ~ 1.41x
- 현재 scale 재계산 cluster 수: K=4
- feature family ablation 기준 전체 lift drop 상위:
  - shot 제거: 약 0.07 하락
  - stby 제거: 약 0.03 하락
  - edge 제거: 약 0.03 하락

Holdout stress 기준:

- sample 수: 120장
- observable feature 수: 50개
- top-5 유사 wafer 검색 lift: 약 1.40x
- bootstrap 95% CI: 약 1.32x ~ 1.48x
- 관심 검색에서 edge/ring/stby/local은 유지되지만 scratch는 약 0.92x로 불안정하다.

초기 `fbm_grouping_pilot` 36장 기준:

- observable feature 수: 40개
- top-5 유사 wafer 검색 lift: 약 1.46x
- top-3 lift: 약 1.47x
- top-10 lift: 약 1.31x
- CPU pilot 권장 cluster 수: K=4

해석:

- 36장 pilot보다 scale pilot의 lift는 낮아졌지만, 더 현실적인 batch에서 random baseline 대비 우위를 유지했다.
- 새 edge/local/ring/scratch morphology feature 보강 이후 edge family는 실제 retrieval에 기여하는 정보축이 되었다.
- shot-relative defect와 stby defect는 현재 feature set에서 강한 신호로 작동한다.
- local/ring은 개선됐지만 expert review가 필요하고, scratch는 segmentation baseline이나 scratch-specific representation 보강 대상이다.

## 4. 검증 완료 항목

이번 이정표에서 확인한 것:

- `python -m compileall -q scripts src tests` 성공
- `python -m pytest -q` 결과: 22 passed
- 주요 HTML report link check: missing 없음
- feature leakage 점검: `label_*`, `*_mask_ratio`는 observable feature에서 제외
- report 재생성: `project_progress_briefing.html` 최신화

주요 report:

- `outputs/reports/project_progress_briefing.html`
- `outputs/reports/fbm_grouping_report.html`
- `outputs/reports/fbm_grouping_stability_report.html`
- `outputs/reports/fbm_grouping_parameter_sweep_report.html`
- `outputs/reports/fbm_feature_ablation_report.html`
- `outputs/reports/fbm_grouping_scale_report.html`
- `outputs/reports/fbm_grouping_scale_stability_report.html`
- `outputs/reports/fbm_grouping_scale_parameter_sweep_report.html`
- `outputs/reports/fbm_feature_ablation_scale_report.html`

주요 image:

- `outputs/figures/fbm_grouping_pca.png`
- `outputs/figures/fbm_grouping_coassociation.png`
- `outputs/figures/fbm_grouping_parameter_sweep.png`
- `outputs/figures/fbm_feature_ablation.png`
- `outputs/figures/fbm_neighbor_gallery.png`
- `outputs/figures/fbm_grouping_scale_parameter_sweep.png`
- `outputs/figures/fbm_feature_ablation_scale.png`
- `outputs/figures/fbm_neighbor_gallery_scale.png`

## 5. 코드 정리 상태

리팩토링 원칙은 Karpathy guideline에 맞춰 보수적으로 적용했다.

- 큰 공통 abstraction은 아직 만들지 않았다. 현재 검증 script들이 빠르게 변하고 있어, premature abstraction 위험이 크다.
- 대신 문서와 report 템플릿의 목표 정렬을 우선 정리했다.
- `README.md`, `docs/problem_definition.md`, `docs/data_schema.md`, `docs/modeling_strategy.md`, `docs/roadmap.md`, `docs/validation_protocol.md`를 현재 목표 기준으로 갱신했다.
- `scripts/make_report.py`의 ANOVA 중심 표현을 후속 분석 표현으로 낮췄다.
- feature extractor에는 목적이 명확한 morphology/localized edge feature만 추가했다.

## 6. 산출물과 clutter 점검

현재 repo에는 재생성 가능한 synthetic/output 산출물이 많다.

보존할 산출물:

- `data/synthetic/fbm_grouping_pilot`
- `data/synthetic/fbm_grouping_scale_pilot`
- `data/synthetic/methodology_probe`
- `data/synthetic/final_review`
- `outputs/reports/fbm_*`
- `outputs/reports/methodology_*`
- `outputs/reports/project_progress_briefing.html`
- `outputs/figures/fbm_*`
- `outputs/figures/final_review_gallery.png`

정리 대상 또는 보존 가치가 낮은 산출물:

- `data/synthetic/debug*`: 제거 완료
- `outputs/reports/synthetic_features*.csv`: 제거 완료
- `outputs/figures/debug_*_gallery.png`: 제거 완료
- `.pytest_cache`: 제거 완료
- `__pycache__`: 제거 완료

주의:

- `data/synthetic/**`와 `outputs/**`는 `.gitignore` 대상이므로 git에 포함하지 않는 재생성 산출물이다.
- 오래된 flow 포함 debug 산출물은 현재 taxonomy와 맞지 않아 제거했다.
- Python cache와 pytest cache는 안전하게 삭제 가능하다.

## 7. 남은 리스크

- Synthetic realism은 사용자의 expert review가 계속 필요하다.
- 현재 retrieval lift는 synthetic label 기준이다. real wafer sanity check가 필요하다.
- scratch/ring/local은 wafer-level feature만으로 완전히 충분하지 않다.
- 현재 script 간 `standardize`, `kmeans`, `retrieval_metrics` 중복이 있다. 다만 아직 검증 script가 변동 중이라, 공통 module refactor는 다음 안정화 시점에 수행하는 것이 낫다.
- 실제 공정 root cause 분석은 process metadata 조인 이후에만 의미가 있다.

## 8. 다음 이정표

다음 단계는 두 갈래다.

1. CPU-only 확장 검증
   - 155장 scale pilot 결과를 현재 기준으로 고정
   - generator runtime 최적화 후 필요하면 200장 이상 full batch 재시도
   - local morphology score를 expert review로 확인
   - scratch는 segmentation 또는 curve/arc-specific detector 후보로 이동

2. Real-unlabeled 준비
   - 실제 wafer를 repo에 저장하지 않는 feature extraction workflow 설계
   - feature NaN/폭주 방지
   - nearest-neighbor gallery를 expert가 검토하는 절차 정의

이 상태라면 다음 단계로 넘어갈 수 있다. 단, 오래된 debug 산출물 삭제 여부는 disk 관리 정책에 따라 결정하면 된다.
