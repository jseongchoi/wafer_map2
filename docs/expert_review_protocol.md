# 전문가 리뷰 절차

## 목적

이 문서는 FBM feature 기반 유사 wafer 검색 결과가 실제 현업 눈에도 의미 있는지 확인하기 위한 최소 리뷰 절차다.

핵심은 모델을 더 복잡하게 만드는 것이 아니다. Query wafer와 top-k neighbor를 전문가가 보고, 검색 결과가 업무적으로 쓸 만한지 판단을 남기는 것이다.

```text
nearest-neighbor CSV
-> 전문가 리뷰 양식
-> reviewer decision
-> 요약 지표
-> feature 보강 / morphology / segmentation 우선순위 결정
```

## 현재 위치

이 절차는 라벨 없는 실제 wafer 처리 다음 단계다.

- `scripts/extract_real_unlabeled_features.py`가 feature와 nearest-neighbor CSV를 만든다.
- `scripts/extract_real_unlabeled_features.py`는 reference feature가 주어지면 reviewer 입력용 CSV도 함께 만든다.
- `scripts/make_expert_review_template.py`는 이미 만든 nearest-neighbor CSV를 리뷰 양식 CSV/HTML로 바꿀 때 사용한다.
- `scripts/summarize_expert_review.py`가 사람이 채운 리뷰 결과를 집계한다.

## 리뷰 단위

리뷰 단위는 query wafer 하나와 neighbor wafer 하나의 쌍이다.

기본 컬럼:

- `query_sample_id`: query wafer id
- `rank`: nearest-neighbor rank
- `neighbor_sample_id`: neighbor wafer id
- `distance`: feature space distance
- `reviewer_decision`: 같은 불량 계열인지 판단
- `query_defect_family`: query wafer에서 리뷰어가 본 주된 결함군
- `neighbor_defect_family`: neighbor wafer에서 리뷰어가 본 주된 결함군
- `dominant_defect`: pair 판단에서 가장 중요한 불량 축
- `clock_position_match`: 위치/방향성이 맞는지 판단
- `missed_major_defect`: query의 중요한 불량을 neighbor가 놓쳤는지 판단
- `retrieval_failure_mode`: mismatch/partial-match의 구조적 실패 유형
- `next_action`: 다음 feature/model 보강 후보
- `safe_comment`: 리뷰어 자유 메모

## 허용 라벨

`reviewer_decision`:

- `same_family`: 주된 불량 계열과 위치감이 충분히 비슷함
- `partial_match`: 일부 축은 맞지만 중요한 차이가 있음
- `mismatch`: 현업 관점에서 유사맵으로 보기 어려움
- `not_sure`: 판단 보류

`dominant_defect`:

- `edge`
- `shot_grid`
- `stby_pattern`
- `stby_hidden_origin`
- `ring`
- `scratch`
- `local`
- `random`
- `mixed`
- `none`
- `unknown`

`clock_position_match`:

- `yes`
- `partial`
- `no`
- `not_applicable`

`missed_major_defect`:

- `yes`
- `no`
- `not_sure`

`retrieval_failure_mode` 주요 값:

- `none`
- `wrong_family`
- `right_family_wrong_location`
- `missed_query_defect`
- `scratch_orientation_span_mismatch`
- `shot_phase_layout_mismatch`
- `ring_radius_width_mismatch`
- `local_blob_topology_mismatch`
- `stby_hidden_origin_mismatch`
- `severity_scale_mismatch`
- `parser_or_mask_issue`
- `insufficient_evidence`
- `not_sure`

`next_action` 주요 값:

- `keep_baseline`
- `feature_weight_tuning`
- `add_location_aware_feature`
- `add_scratch_component_features`
- `add_shot_phase_features`
- `add_ring_radius_width_features`
- `add_local_topology_features`
- `add_stby_origin_coupling_features`
- `segmentation_candidate`
- `scratch_specific_track`
- `parser_validation`
- `review_more_samples`

## 실행

라벨 없는 실제 wafer 처리에서 바로 리뷰 양식을 만들 수 있다.

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --neighbors-out outputs/reports/real_unlabeled_neighbors.csv `
  --review-template-out outputs/reports/real_unlabeled_expert_review_template.csv
```

이미 생성된 neighbor CSV를 별도로 리뷰 양식/리포트로 바꿀 수도 있다.

```powershell
python scripts/make_expert_review_template.py `
  --neighbors outputs/reports/real_unlabeled_neighbors.csv `
  --template-out outputs/reports/expert_review_template.csv `
  --report-out outputs/reports/expert_review_template.html
```

리뷰어가 `expert_review_template.csv`를 채운 뒤 요약한다.

```powershell
python scripts/summarize_expert_review.py `
  --review outputs/reports/expert_review_template.csv `
  --out outputs/reports/expert_review_summary.html `
  --metrics outputs/reports/expert_review_summary_metrics.json
```

## 집계 지표

주요 지표:

- `same_family_rate`: review row 기준 same-family 비율
- `partial_match_rate`: review row 기준 partial-match 비율
- `accepted_match_rate`: same-family 또는 partial-match 비율
- `missed_major_defect_rate`: 주요 불량을 놓쳤다고 표시된 비율
- `query_topk_same_family_rate`: query별 top-k 안에 same-family가 하나라도 있는 비율
- `query_topk_accept_rate`: query별 top-k 안에 same-family 또는 partial-match가 하나라도 있는 비율
- `dominant_defect_metrics`: defect family별 accepted/missed-major 비율
- `retrieval_failure_mode_counts`: 실패 유형별 건수
- `next_action_counts`: 후속 액션별 건수
- `next_action_queue`: defect family x failure mode x action 단위의 다음 작업 목록

해석 기준 초안:

- accepted rate가 높고 missed-major가 낮으면 현재 feature 검색을 triage에 쓸 수 있다.
- mismatch가 높으면 feature distance가 현업 유사도와 어긋난다.
- 특정 defect family에서 missed-major가 높으면 그 family는 feature 보강 또는 segmentation 후보로 올린다.
- scratch/local에서 계속 약하면 wafer-level feature보다 morphology 또는 segmentation으로 넘기는 것이 맞다.
- `next_action_queue`에서 `stby_hidden_origin_mismatch`, `shot_phase_layout_mismatch`, `ring_radius_width_mismatch` 같은 항목이 쌓이면 해당 feature family 보강 작업으로 바로 연결한다.

## 파일럿 성공 판정 기준

파일럿 성공 판정은 아래 조건을 모두 만족해야 한다. 이 기준을 만족하기 전에는 기업 판매용 성능 주장으로 쓰지 않는다.

- 리뷰한 query wafer: 최소 `20`, 권장 `50` 이상
- `accepted_match_rate`: `70%` 이상
- `query_topk_accept_rate`: `80%` 이상
- `missed_major_defect_rate`: `5%` 이하
- `parser_or_mask_issue`: `0`

## 다음 의사결정

전문가 리뷰 결과는 다음 순서로 사용한다.

1. 파일럿 성공 판정 기준을 만족하면 현재 feature 검색을 파일럿용 최소 분류 보조 도구로 유지한다.
2. 특정 family만 약하면 해당 family의 feature를 보강한다.
3. scratch/local처럼 위치와 형태가 핵심이면 connected-component morphology를 먼저 시도한다.
4. morphology로도 부족하면 synthetic mask 기반 segmentation baseline으로 넘어간다.
