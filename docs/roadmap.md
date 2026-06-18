# 로드맵

## 현재 위치

현재 프로젝트는 합성 데이터 검증을 지나 **라벨 없는 실제 wafer 적용 준비 단계**에 있다.

```text
완료/검증됨:
합성 FBM generator
-> 실제 데이터용 feature 추출
-> 전체 유사 wafer 검색
-> 관심 불량별 검색
-> holdout 검증
-> 라벨 없는 실제 wafer 처리 절차
-> 전문가 리뷰 template

현재 주 경로:
보안 환경의 실제 wafer .npz
-> sanity / drift report
-> nearest-neighbor 검색
-> 전문가 리뷰
-> feature 보강 또는 AI 모델 후보 정리
```

## Phase 0. 문제 정의와 입력 기준

상태: 완료

목표:

- 문제 정의
- FBM 분석용 배열 형식 정의
- 합성 generator 요구사항 정리
- 검증 방법 정리

핵심 문서:

- [프로젝트 개요](project_overview.md)
- [데이터 형식](data_schema.md)
- [검증 방법](validation_protocol.md)
- [불량 패턴 정리](pattern_taxonomy.md)

## Phase 1. 합성 FBM Generator

상태: 완료, 필요 시 보정

완료된 것:

- 약 600 net die급 wafer geometry
- Grade 0~7
- wafer 밖 영역, 실제 test 영역, stby 분리
- edge, local, shot-relative, stby-origin, ring, scratch 계열 생성
- 합성 데이터 oracle mask는 검증용으로만 사용

주의:

- 합성 데이터 성능은 실제 wafer 성능 인증이 아니다.
- Generator와 feature extractor가 같은 가정을 공유할 수 있으므로 실제 wafer 리뷰가 필요하다.

## Phase 2. Feature와 유사 Wafer 검색

상태: 완료, 기준선으로 유지

완료된 것:

- compact feature 50개
- scale 155장 top-k retrieval lift 약 1.36x
- holdout 120장 top-k retrieval lift 약 1.40x
- class/class_location/feature_key 기준 관심 불량 검색 신호 확인
- 단순 resize representation은 전체 유사 wafer 검색 대체재로 부적합하다고 판단

사용 원칙:

- 전체 유사 wafer 검색은 compact feature 50개 기준을 유지한다.
- `polar_*`, `stby_polar_*`는 위치가 중요한 검색에서만 조건부로 쓴다.
- `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 실제 inference feature가 아니다.

## Phase 3. Proposal과 Segmentation 준비

상태: 보조 도구로 정리 완료

현재 판단:

- Patch proposal: edge/local/stby 리뷰 후보를 줄이는 용도
- Curve proposal: ring/center arc 리뷰 후보를 줄이는 용도
- Scratch: rule/proposal 과투자를 멈추고 segmentation 또는 scratch 전용 line feature로 분리

주의:

- Proposal recall은 최종 성능 지표가 아니다.
- 지금은 proposal 튜닝보다 라벨 없는 실제 wafer 처리 절차와 전문가 리뷰가 우선이다.

## Phase 4. 라벨 없는 실제 Wafer 적용

상태: 현재 주 작업

목표:

- 실제 wafer raw data를 repo에 저장하지 않고 feature를 추출한다.
- 보안 환경의 `.npz`와 manifest를 입력으로 받는다.
- feature CSV, sanity JSON, drift report, nearest-neighbor CSV, 전문가 리뷰 template을 생성한다.

완료된 것:

- `scripts/extract_real_unlabeled_features.py`
- `real_unlabeled_manifest/v1`
- `observable_fbm_features/v1`
- `.npz` array validation
- reference 대비 feature drift summary
- nearest-neighbor CSV와 전문가 리뷰 template 연결

다음 확인 단계:

1. 실제 보안 환경에서 표준 key를 쓰는 `.npz` 1건 export
2. `array_keys` mapping이 필요한 `.npz` 1건 export
3. sanity JSON에서 stby/Grade7/wafer 밖 영역/valid-test 처리가 맞는지 확인
4. top-k nearest-neighbor 결과를 전문가가 최소 20~50 pair 평가
5. `next_action_queue`를 보고 feature 보강 또는 AI 모델 후보 결정

## Phase 5. Scratch/Local 보강

상태: 대기

시작 조건:

- 전문가 리뷰에서 scratch/local 계열 실패가 반복적으로 확인된다.
- 현재 feature 보강만으로 부족하다는 근거가 생긴다.

후보:

- connected-component morphology
- line enhancement / skeleton continuity
- lightweight multi-label segmentation
- self-supervised embedding 또는 metric learning

## Phase 6. 공정 Metadata 분석

상태: 대기

시작 조건:

- FBM feature table이 안정화된다.
- 공정/설비/lot/recipe/chamber/test metadata와 조인 가능하다.

분석 예:

- 특정 tool/chamber에서 shot-relative score가 높은가?
- 특정 recipe 이후 edge-localized score가 상승하는가?
- 특정 lot에서 stby-origin-hidden score가 반복되는가?

ANOVA는 이 단계의 후속 분석이다.
