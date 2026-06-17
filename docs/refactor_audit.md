# Refactor Audit

기준일: 2026-06-17

## 결론

현재 코드는 원래 목표에서 벗어나지 않았다. 지금 필요한 것은 대규모 리팩토링이 아니라, real-unlabeled workflow와 expert review loop가 실제 데이터 흐름에서 한 번 검증될 때까지 구조를 안정적으로 유지하는 것이다.

이번 리팩토링도 같은 원칙으로 진행했다. 검증 script 전체를 framework로 갈아엎지 않고, 여러 script가 반복하던 안정 계산부만 `src/wafermap` 아래로 분리했다.

## 이번 업데이트

추가한 공통 유틸:

- `src/wafermap/features/selection.py`
  - compact observable feature 선택
  - `label_*`, `*_mask_ratio`, `polar_*`, `stby_polar_*` 제외 정책
  - 위치-aware retrieval에서만 polar feature를 켤 수 있는 옵션
- `src/wafermap/evaluation/nearest.py`
  - 표준화
  - matrix 기반 Euclidean distance
  - self nearest-neighbor와 query/reference cross nearest-neighbor
- `src/wafermap/reporting/expert_review.py`
  - nearest-neighbor CSV를 expert review template row로 변환
  - reference `label_*` 컬럼은 reviewer bias 방지를 위해 template에 복사하지 않음
  - query/neighbor defect family, retrieval failure mode, next action field를 공통 schema로 관리

수정한 script:

- `scripts/extract_real_unlabeled_features.py`
  - compact observable feature selector와 cross nearest-neighbor 유틸 사용
  - nearest-neighbor CSV 생성 시 expert review template CSV도 함께 생성
- `scripts/make_expert_review_template.py`
  - template row 생성 로직을 공통 reporting helper로 이동
- `scripts/evaluate_retrieval_confidence.py`
- `scripts/evaluate_resize_benchmark.py`
- `scripts/analyze_fbm_grouping.py`
- `scripts/evaluate_grouping_stability.py`
- `scripts/evaluate_interest_retrieval.py`
- `scripts/evaluate_defect_feature_retrieval.py`
- `scripts/evaluate_feature_ablation.py`
- `scripts/evaluate_methodology.py`
- `scripts/sweep_grouping_parameters.py`

추가 hardening:

- real `.npz` loader는 semantic cast 전에 severity/mask/chip_index dtype, shape, finite, integer-like/binary 조건을 검사한다.
- manifest schema version과 `source_type`을 명시적으로 검증한다.
- global methodology/ablation/sweep 경로는 compact observable feature만 쓰도록 polar spatial feature를 제외한다.

남겨둔 중복:

- HTML report helper는 각 report의 메시지와 layout이 계속 바뀌므로 아직 공통화하지 않았다.
- 위치-aware retrieval의 feature subset 정책은 실험 의미가 강하므로 selector만 무리하게 통합하지 않았다.
- real `.npz` loader와 validator는 실제 보안 데이터 1회 검증 후에만 `src/wafermap/data` 쪽으로 승격한다.

## 양호한 점

- `Grade 0`, `none-wafer`, `valid_test_mask`, `stby_mask`가 의미적으로 분리되어 있다.
- 실제 inference feature는 observable-only로 유지된다.
- `label_*`, `*_mask_ratio`, `pattern_masks`는 검증 전용으로 분리되어 있다.
- synthetic generator, feature extractor, real-unlabeled workflow, expert review protocol에 테스트가 있다.
- `outputs/**`, `data/synthetic/**`는 재생성 산출물로 `.gitignore`에 들어가 있다.
- morphology 보강 후에도 observable feature 계약은 유지된다. 현재 50개 feature는 synthetic mask를 inference input으로 쓰지 않는다.

## 주의할 점

- `src/wafermap/synth/generator.py`는 약 730라인으로 커졌다. 다만 defect realism이 계속 바뀌는 단계라 지금 분리하면 오히려 수정 비용이 커질 수 있다.
- `scripts/extract_real_unlabeled_features.py`는 loader, validator, nearest-neighbor, HTML report를 함께 가진 MVP 구조다. 실제 semantic `.npz` 검증 후 `src/wafermap` 내부 모듈로 승격하는 것이 좋다.
- report script 사이에 HTML/CSS helper 중복이 있다. 보고서 포맷이 안정되기 전까지는 허용한다.
- `other_sample/`은 reference 성격이므로 pipeline source of truth로 해석하지 않는다.
- scratch는 feature를 계속 덧대도 holdout에서 불안정하다. 단순 rule feature 추가보다 segmentation 또는 scratch-specific representation으로 넘기는 편이 낫다.

## 다음 리팩토링 순서

1. 실제 보안 semantic `.npz` 1회 검증
2. real validator를 `src/wafermap/evaluation` 또는 `src/wafermap/data`로 이동
3. generator를 geometry, pattern drawing, stby placement로 분리
4. report helper를 `src/wafermap/reporting`으로 공통화
5. expert review에 reviewer agreement가 필요해질 때만 schema 확장
