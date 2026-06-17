# Roadmap

## 현재 위치

현재 프로젝트는 synthetic validation을 지나 **Phase 4: Secure Real-Unlabeled Adaptation** 진입 단계다.

```text
완료/검증됨:
synthetic generator
-> observable feature extraction
-> global similar wafer retrieval
-> interest-conditioned retrieval
-> holdout stress validation
-> real-unlabeled workflow MVP
-> expert review template MVP

현재 주 경로:
secure real `.npz` input
-> sanity / drift report
-> nearest-neighbor retrieval
-> expert review
-> feature/model backlog
```

## Phase 0. Problem And Contract

상태: 완료

목표:

- 문제 정의
- semantic tensor schema
- synthetic generator 요구사항
- validation protocol

핵심 문서:

- [Project Overview](project_overview.md)
- [Data Schema](data_schema.md)
- [Validation Protocol](validation_protocol.md)
- [Pattern Taxonomy](pattern_taxonomy.md)

## Phase 1. Synthetic Generator

상태: 완료, 필요 시 보정

완료된 것:

- 약 600 net die급 wafer geometry
- Grade 0~7
- none-wafer, valid-test, stby 분리
- edge, local, shot-relative, stby-origin, ring, scratch 계열 생성
- synthetic oracle mask는 검증용으로만 사용

주의:

- Synthetic 성능은 real 성능 인증이 아니다.
- Generator와 feature extractor가 같은 가정을 공유하므로 real review gate가 필요하다.

## Phase 2. Observable Feature And Retrieval

상태: 완료, 운영 baseline 유지

완료된 것:

- compact observable feature 50개
- scale 155장 top-k retrieval lift 약 1.36x
- holdout 120장 top-k retrieval lift 약 1.40x
- class/class_location/feature_key 기준 interest retrieval 신호 확인
- resize-only representation은 global retrieval 대체재로 부적합하다고 판단

운영 원칙:

- Global retrieval은 compact observable feature 50개 기준을 유지한다.
- `polar_*`, `stby_polar_*`는 위치-aware retrieval에서만 조건부 사용한다.
- `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 inference feature가 아니다.

## Phase 3. Proposal And Segmentation Readiness

상태: 보조층으로 정리 완료

판정:

- Patch proposal: edge/local/stby review 후보 축소용
- Curve proposal: ring/center arc review 후보 축소용
- Scratch: rule/proposal 과투자를 멈추고 segmentation 또는 scratch-specific line representation으로 분리

주의:

- Proposal recall은 최종 성능 지표가 아니다.
- 지금은 proposal 튜닝보다 real-unlabeled workflow와 expert review loop가 우선이다.

## Phase 4. Secure Real-Unlabeled Adaptation

상태: 현재 주 작업

목표:

- 실제 wafer raw data를 repo에 저장하지 않고 feature를 추출한다.
- 보안 환경의 semantic `.npz` manifest를 입력으로 받는다.
- feature CSV, sanity JSON, drift report, nearest-neighbor CSV, expert review template을 생성한다.

완료된 것:

- `scripts/extract_real_unlabeled_features.py`
- `real_unlabeled_manifest/v1`
- `observable_fbm_features/v1`
- `.npz` semantic array validation
- reference 대비 feature drift summary
- nearest-neighbor CSV와 expert review template 연결

다음 gate:

1. 실제 보안 환경에서 standard-key `.npz` 1건 export
2. `array_keys` mapping이 필요한 `.npz` 1건 export
3. sanity JSON에서 stby/Grade7/none-wafer/valid-test 계약 확인
4. top-k nearest-neighbor 결과를 전문가가 최소 20~50 pair 평가
5. `next_action_queue`를 기준으로 feature/model backlog 결정

## Phase 5. Scratch/Local Model Backlog

상태: 대기

시작 조건:

- Expert review에서 scratch/local 계열 failure가 반복적으로 확인된다.
- Observable morphology 보강으로 충분하지 않다는 근거가 생긴다.

후보:

- connected-component morphology
- line enhancement / skeleton continuity
- lightweight multi-label segmentation
- self-supervised embedding 또는 metric learning

## Phase 6. Process Metadata Statistics

상태: 대기

시작 조건:

- FBM feature table이 안정화된다.
- 공정/설비/lot/recipe/chamber/test metadata와 조인 가능하다.

분석 예:

- 특정 tool/chamber에서 shot-relative score가 높은가?
- 특정 recipe 이후 edge-localized score가 상승하는가?
- 특정 lot에서 stby-origin-hidden score가 반복되는가?

ANOVA는 이 단계의 후속 분석이다.
