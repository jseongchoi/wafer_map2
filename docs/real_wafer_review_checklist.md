# Real Wafer 리뷰 체크리스트

## 결론

내일 real wafer로 리뷰해볼 수 있다.

단, 실제 wafer raw image/array를 repo에 넣는 방식이 아니다. 보안 환경에서 제품별 raw PNG 폴더를 준비하고, batch script로 feature와 sanity report를 만든 뒤 생성된 리뷰 양식을 채우는 방식이다. Semantic `.npz` export가 이미 준비된 경우에는 기존 manifest 방식도 사용할 수 있다.

```text
보안 환경의 제품별 raw PNG 폴더
-> manifest 자동 생성
-> feature / sanity / nearest-neighbor 생성
-> 전문가 리뷰 CSV 작성
-> 요약 지표 확인
```

## 내가 해야 할 일

1. real wafer sample 5~20장을 고른다.
   - 가능하면 정상/edge/local/shot/ring/scratch/stby가 섞이게 고른다.
   - sample id는 익명 id로 바꾼다.

2. 제품별 폴더에 raw PNG를 모은다.
   - 예: `raw_root/product_a/*.png`
   - PNG는 8-bit grayscale이어야 한다.
   - gray value 기준은 `0, 31, 151, 175, 191, 207, 223, 255`다.
   - chip 전체가 `255`인 경우만 stby fail chip으로 분리한다.

3. batch script를 실행한다.
   - 제품별 stby chip에서 chip size/grid를 먼저 추론한다.
   - stby가 없거나 STBY fail chip이 서로 붙어 geometry가 애매한 제품은 `--geometry-json`으로 제품별 geometry를 준다.
   - lot/tool/recipe/chamber/wafer id는 sample id에 넣지 않는다.

4. 라벨 없는 실제 wafer 처리 script를 실행한다.
   - PNG batch script는 기본적으로 이 단계까지 함께 실행한다.
   - `report.html`
   - `sanity.json`
   - `features.csv`
   - reference feature가 있으면 `neighbors.csv`, `review_template.csv`

5. sanity 결과를 먼저 본다.
   - FAIL이면 retrieval 리뷰보다 parser/export 문제를 먼저 고친다.
   - `PASS`이면 top-k 리뷰로 넘어간다.

6. `review_template.csv` 파일의 리뷰 항목을 채운다.
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

우선 제품별 raw PNG 폴더를 준비한다.

```text
D:/secure_fbm/raw_png/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_001.png
```

PNG 입력 기준:

- 정확한 8-bit grayscale PNG
- gray value: `0, 31, 151, 175, 191, 207, 223, 255`
- `255`는 chip 전체가 255일 때 stby, 일부 pixel이면 Grade 7

Semantic `.npz`를 쓰는 경우에는 각 wafer를 `.npz`로 export한다.

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

PNG 폴더 방식에서는 `scripts/analyze_png_raw_folders.py`가 manifest를 자동 생성한다. 이 manifest에는 실제 raw path가 들어가므로 공유하지 않는다.

템플릿:

- 표준 key: `configs/eval/real_unlabeled_manifest_template_standard.json`
- key mapping 필요: `configs/eval/real_unlabeled_manifest_template_keymap.json`
- raw PNG: `configs/eval/real_unlabeled_manifest_template_png.json`

반드시 확인할 것:

- `schema_version`: `real_unlabeled_manifest/v1`
- `feature_schema_version`: `observable_fbm_features/v1`
- `pseudonymized`: `true`
- `parser_name`, `parser_version`, `orientation` 기록
- `chip_blocks.width`, `chip_blocks.height`
- `grid.rows`, `grid.cols`
- `arrays_npz`는 보안 환경의 local path
- PNG batch manifest의 `png_path`도 보안 환경의 local path

금지:

- lot id, wafer id, tool id, recipe, chamber 정보
- 실제 raw file path가 들어간 manifest 원본 공유
- raw image/array repo 저장

## 실행 명령

제품별 raw PNG 폴더 운영 실행:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root D:/secure_fbm/raw_png `
  --production-run `
  --geometry-json D:/secure_fbm/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

실제 운영에서는 자동 geometry 추론을 성공 근거로 쓰지 않는다. `--production-run`은 제품별 `--geometry-json`과 양수 `actual_net_die`를 요구하고, debug용 workspace/input/output override를 막는다.
`actual_net_die=0`은 운영 승인값으로 보지 않는다.

Semantic `.npz` manifest를 직접 쓸 때:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest D:/secure_fbm/real_manifest.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --features-out outputs/reports/real_unlabeled_features.csv `
  --sanity-out outputs/reports/real_unlabeled_sanity.json `
  --report-out outputs/reports/real_unlabeled_report.html `
  --neighbors-out outputs/reports/real_unlabeled_neighbors.csv `
  --review-template-out outputs/reports/real_unlabeled_expert_review_template.csv
```

먼저 볼 것:

1. `report.html`
2. `sanity.json`
3. `features.csv`
4. reference feature가 있으면 `review_template.csv`

Sanity가 FAIL이면 retrieval 리뷰보다 parser/export 수정을 먼저 한다.

## 리뷰어가 채울 것

PNG batch 경로에서는 `outputs/reports/real_png_batch/review_template.csv`를 채운다.
Semantic `.npz` smoke 경로를 직접 쓴 경우에만 지정한 `--review-template-out` 경로를 채운다.

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
  --review outputs/reports/real_png_batch/review_template.csv `
  --out outputs/reports/real_png_batch/review_summary.html `
  --metrics outputs/reports/real_png_batch/review_summary_metrics.json
```

## 나에게 주면 되는 결과

공유해도 되는 것:

- `outputs/reports/real_png_batch/sanity.json`
- `outputs/reports/real_png_batch/batch_metadata.json`
- `outputs/reports/real_png_batch/report.html`
- `outputs/reports/real_png_batch/features.csv`
- `outputs/reports/real_png_batch/neighbors.csv`
- 채운 `outputs/reports/real_png_batch/review_template.csv`
- `outputs/reports/real_png_batch/review_summary_metrics.json`
- `outputs/reports/real_png_batch/review_summary.html`

같이 알려주면 좋은 메모:

- real wafer sample 수
- 제품별 `chip_blocks`, `grid`
- `stby_chip_count_est` 범위
- `grade_min`, `grade_max`
- sanity warning 종류
- 사용한 옵션: `--geometry-json`, `--wafer-mask-strategy`
- `.npz` 경로를 썼다면 사용한 array key mapping
- `orientation` 값
- sanity FAIL이 있었다면 어떤 error였는지

공유하면 안 되는 것:

- 실제 wafer raw image
- 실제 wafer raw array `.npz`
- 실제 file path가 들어간 manifest 원본
- `outputs/private/*_manifest.json`
- lot/tool/recipe/chamber/wafer id

공유 전 검사:

- `sample_id`가 `product_<10hex>_w<10hex>` 형식의 opaque alias인지 확인한다.
- 제품명, lot, wafer, tool, recipe, chamber 정보가 공유 산출물에 들어가면 안 된다.
- CSV에 `png_path`, `arrays_npz`, `metadata_json`, 실제 lot/tool/recipe/chamber/wafer id가 없어야 한다.
- 채운 review/comment 필드에 민감어가 들어가지 않았는지 확인한다.
- 폴더 전체를 압축해 공유하지 말고 필요한 파일만 골라 공유한다.

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

## 파일럿 성공 판정 기준

파일럿 성공 판정은 아래 조건을 모두 만족해야 한다. 이 기준을 만족하기 전에는 기업 판매용 성능 주장으로 쓰지 않는다.

- 실제 batch 기본 검사 오류: `0`
- 공유 산출물 보안 문제: `0`
- 실제 batch 실행 정보: `production_run=true`, `geometry_contract=explicit`, `manifest_location=outputs/private`, 제품 수와 양수 `actual_net_die` 제품 수 일치
- 실제 산출물 행 수: `features.csv`, `sanity.json`, `neighbors.csv`, `review_template.csv`가 `png_sample_count`와 모순되지 않음
- Reviewed query wafer: 최소 `20`, 권장 `50` 이상
- `accepted_match_rate`: `same_family` 또는 `partial_match` 비율 `70%` 이상
- `query_topk_accept_rate`: query별 top-k 안에 accept가 하나 이상 있는 비율 `80%` 이상
- `missed_major_defect_rate`: `5%` 이하
- `parser_or_mask_issue`: `0`
- `sensitive_comment` 또는 민감정보 포함 comment: `0`
- Scratch/local/stby 관련 실패가 반복되면 해당 family는 별도 개선 track으로 분리하고 판매 성능 범위에서 제외한다.
- CPU encoder 결과는 실제 리뷰 label로 별도 검증되기 전까지 `synthetic_label_hint`와 리뷰 우선순위로만 사용한다.

## AI 모델 구현 상태

현재 구현된 것:

- Feature extractor
- compact 50 feature 기반 global nearest-neighbor retrieval
- interest-conditioned retrieval 평가 경로
- feature selection / standardization / nearest-neighbor 공통 유틸
- 라벨 없는 실제 wafer sanity / drift / 리뷰 양식 절차
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
