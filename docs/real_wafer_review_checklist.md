# Real Wafer 리뷰 체크리스트

## 결론

내일 real wafer로 리뷰해볼 수 있다.

단, 실제 wafer raw image/array를 repo에 넣는 방식이 아니다. 보안 환경에서 `.npz`와 manifest를 만들고, 이 repo의 라벨 없는 실제 wafer 처리 절차를 실행한 뒤 생성된 리뷰 template을 채우는 방식이다.

```text
보안 환경의 실제 FBM
-> .npz export
-> manifest 작성
-> feature / sanity / nearest-neighbor 생성
-> 전문가 리뷰 CSV 작성
-> 요약 지표 확인
```

## 내가 해야 할 TODO

1. real wafer sample 5~20장을 고른다.
   - 가능하면 정상/edge/local/shot/ring/scratch/stby가 섞이게 고른다.
   - sample id는 익명 id로 바꾼다.

2. 각 wafer를 `.npz`로 export한다.
   - 필수: `severity`, `wafer_mask`, `valid_test_mask`, `stby_mask`
   - 권장: `chip_index`
   - stby는 Grade 7이 아니라 `stby_mask=1`, `valid_test_mask=0`, `severity=0`인지 확인한다.

3. manifest를 만든다.
   - 표준 key면 `configs/eval/real_unlabeled_manifest_template_standard.json`
   - key 이름이 다르면 `configs/eval/real_unlabeled_manifest_template_keymap.json`
   - lot/tool/recipe/chamber/wafer id는 넣지 않는다.

4. 라벨 없는 실제 wafer 처리 script를 실행한다.
   - `real_unlabeled_report.html`
   - `real_unlabeled_sanity.json`
   - `real_unlabeled_neighbors.csv`
   - `real_unlabeled_expert_review_template.csv`

5. sanity 결과를 먼저 본다.
   - FAIL이면 retrieval 리뷰보다 parser/export 문제를 먼저 고친다.
   - PASS이면 top-k review로 넘어간다.

6. review template을 채운다.
   - query당 top-5 neighbor를 본다.
   - 전체 20~50 pair 정도를 먼저 채운다.
   - `reviewer_decision`, `dominant_defect`, `retrieval_failure_mode`, `next_action`은 가능하면 꼭 채운다.

7. review summary를 생성한다.
   - `same_family_rate`
   - `accepted_match_rate`
   - `missed_major_defect_rate`
   - `next_action_queue`

8. 나에게 공유 가능한 산출물만 준다.
   - raw image/array, 실제 path, lot/tool/recipe/chamber/wafer id는 공유하지 않는다.

## 내일 준비할 입력

권장 수량:

- 첫 리뷰: real wafer 5~20장
- 가능하면 정상/edge/local/shot/ring/scratch/stby가 섞인 sample
- sample id는 익명 id만 사용

각 wafer를 `.npz`로 export한다.

필수 array:

- `severity`: Grade 0~7, `[H, W]`
- `wafer_mask`: wafer 내부 1, wafer 밖 0
- `valid_test_mask`: 실제 test된 pixel 1
- `stby_mask`: stby fail chip 영역 1

권장 array:

- `chip_index`: die/chip id. wafer 밖은 -1

중요:

- Stby는 Grade 7이 아니다.
- Stby는 `stby_mask=1`, `valid_test_mask=0`, `severity=0`이어야 한다.
- Wafer 밖 영역과 in-wafer Grade 0은 `wafer_mask`로 구분한다.

## Manifest 작성

템플릿:

- 표준 key: `configs/eval/real_unlabeled_manifest_template_standard.json`
- key mapping 필요: `configs/eval/real_unlabeled_manifest_template_keymap.json`

반드시 확인할 것:

- `schema_version`: `real_unlabeled_manifest/v1`
- `feature_schema_version`: `observable_fbm_features/v1`
- `pseudonymized`: `true`
- `parser_name`, `parser_version`, `orientation` 기록
- `chip_blocks.width`, `chip_blocks.height`
- `grid.rows`, `grid.cols`
- `arrays_npz`는 보안 환경의 local path

금지:

- lot id, wafer id, tool id, recipe, chamber 정보
- 실제 raw file path가 들어간 report 공유
- raw image/array repo 저장

## 실행 명령

현재 local reference feature가 있으면 다음처럼 실행한다.

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest D:/secure_fbm/real_manifest.json `
  --reference-features outputs/reports/fbm_grouping_scale_features.csv `
  --features-out outputs/reports/real_unlabeled_features.csv `
  --sanity-out outputs/reports/real_unlabeled_sanity.json `
  --report-out outputs/reports/real_unlabeled_report.html `
  --neighbors-out outputs/reports/real_unlabeled_neighbors.csv `
  --review-template-out outputs/reports/real_unlabeled_expert_review_template.csv
```

먼저 볼 것:

1. `real_unlabeled_report.html`
2. `real_unlabeled_sanity.json`
3. `real_unlabeled_expert_review_template.csv`

Sanity가 FAIL이면 retrieval 리뷰보다 parser/export 수정을 먼저 한다.

## 리뷰어가 채울 것

`outputs/reports/real_unlabeled_expert_review_template.csv`를 채운다.

최소 리뷰 권장:

- query당 top-5 neighbor
- 전체 20~50 pair

필수로 채우면 좋은 column:

- `reviewer_decision`
  - `same_family`
  - `partial_match`
  - `mismatch`
  - `not_sure`
- `query_defect_family`
- `neighbor_defect_family`
- `dominant_defect`
- `clock_position_match`
- `missed_major_defect`
- `retrieval_failure_mode`
- `next_action`
- `safe_comment`

`safe_comment`에는 보안 정보를 쓰지 않는다.

## 리뷰 후 요약

리뷰 CSV를 채운 뒤 실행한다.

```powershell
python scripts/summarize_expert_review.py `
  --review outputs/reports/real_unlabeled_expert_review_template.csv `
  --out outputs/reports/real_unlabeled_expert_review_summary.html `
  --metrics outputs/reports/real_unlabeled_expert_review_summary_metrics.json
```

## 나에게 주면 되는 결과

공유해도 되는 것:

- `outputs/reports/real_unlabeled_sanity.json`
- `outputs/reports/real_unlabeled_report.html`
- `outputs/reports/real_unlabeled_features.csv`
- `outputs/reports/real_unlabeled_neighbors.csv`
- 채운 `outputs/reports/real_unlabeled_expert_review_template.csv`
- `outputs/reports/real_unlabeled_expert_review_summary_metrics.json`
- `outputs/reports/real_unlabeled_expert_review_summary.html`

같이 알려주면 좋은 메모:

- real wafer sample 수
- 사용한 array key mapping
- `orientation` 값
- chip block width/height
- grid rows/cols
- sanity FAIL이 있었다면 어떤 error였는지

공유하면 안 되는 것:

- 실제 wafer raw image
- 실제 wafer raw array `.npz`
- 실제 file path가 들어간 manifest 원본
- lot/tool/recipe/chamber/wafer id

## 내일 판단 기준

우선 성공 기준:

- sanity error 0
- feature extraction 완료
- query별 top-k neighbor 생성
- reviewer가 봤을 때 top-k 안에 `same_family` 또는 `partial_match`가 일부 존재

주의 신호:

- 대부분 `mismatch`
- `parser_or_mask_issue`가 반복
- `missed_major_defect=yes`가 많음
- scratch/local에서 계속 놓침
- stby가 Grade 7처럼 처리된 흔적

리뷰 후 결정:

- accepted match가 충분하면 현재 feature 검색을 real triage용 최소 버전으로 유지
- 특정 family만 약하면 해당 feature 보강
- scratch/local이 약하면 segmentation 또는 scratch 전용 representation으로 이동
- parser/mask 문제가 있으면 AI/model보다 export 형식부터 수정

## AI 모델 구현 상태

현재 구현된 것:

- Feature extractor
- compact 50 feature 기반 global nearest-neighbor retrieval
- interest-conditioned retrieval 평가 경로
- feature selection / standardization / nearest-neighbor 공통 유틸
- 라벨 없는 실제 wafer sanity / drift / review template 절차
- expert review summary와 `next_action_queue`
- synthetic-label segmentation dataset helper
- NumPy-only 1x1 sigmoid segmentation smoke training

현재 segmentation smoke의 의미:

- 학습 성능 모델이 아니라 배관 검증이다.
- manifest, input tensor, synthetic mask target, weighted BCE loss가 연결되는지 확인한다.
- scratch/local/stby overlap을 향후 segmentation으로 넘길 수 있는 최소 준비 상태다.

아직 구현되지 않은 것:

- 실사용 U-Net / SegFormer / DINO embedding 모델
- real wafer로 검증된 supervised 또는 self-supervised AI model
- calibrated defect probability model
- scratch/local 전용 line/segmentation model

따라서 현재 솔루션의 중심은 AI deep model이 아니라 다음 기준선이다.

```text
feature extraction
-> nearest-neighbor retrieval
-> expert review
-> 필요한 feature 또는 모델 보강
```

AI 모델은 real wafer review에서 약한 defect family가 확인된 뒤, 필요한 곳에 붙이는 단계다.
