# Real Wafer 리뷰 체크리스트

## 결론

실제 wafer 5~20장으로 바로 리뷰할 수 있다. 입력은 제품별 raw PNG 폴더 또는 semantic `.npz` manifest 중 편한 방식을 쓴다.

```text
제품별 raw PNG 폴더
-> manifest 자동 생성
-> feature / sanity / nearest-neighbor 생성
-> 전문가 리뷰 CSV 작성
-> 요약 지표 확인
```

## 내가 해야 할 일

1. real wafer sample 5~20장을 고른다.
   - 가능하면 정상/edge/local/shot/ring/scratch/stby가 섞이게 고른다.

2. 제품별 폴더에 raw PNG를 모은다.
   - 예: `data/raw/product_a/*.png`
   - PNG는 8-bit grayscale이어야 한다.
   - gray value 기준은 `0, 31, 151, 175, 191, 207, 223, 255`다.
   - chip 전체가 `255`인 경우만 stby fail chip으로 분리한다.

3. batch script를 실행한다.
   - 제품별 stby chip에서 chip size/grid를 먼저 추론한다.
   - stby가 없거나 geometry가 애매한 제품은 `--geometry-json`으로 제품별 geometry를 준다.

4. sanity 결과를 먼저 본다.
   - FAIL이면 retrieval 리뷰보다 parser/export 문제를 먼저 고친다.
   - PASS이면 top-k 리뷰로 넘어간다.

5. `review_template.csv` 파일의 리뷰 항목을 채운다.
   - query당 top-5 neighbor를 본다.
   - 전체 20~50 pair 정도를 먼저 채운다.
   - `reviewer_decision`, `dominant_defect`, `retrieval_failure_mode`, `next_action`은 가능하면 꼭 채운다.

6. review summary를 생성한다.
   - `same_family_rate`
   - `accepted_match_rate`
   - `missed_major_defect_rate`
   - `next_action_queue`

## 입력 준비

권장 수량:

- 첫 리뷰: real wafer 5~20장
- 가능하면 정상/edge/local/shot/ring/scratch/stby가 섞인 sample

폴더 예시:

```text
data/raw/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_101.png
```

인트라넷 공유 폴더를 그대로 써도 된다.

```text
Z:/fbm/raw_png/
  product_a/
    wafer_001.png
```

## 실행 명령

제품별 raw PNG 폴더 실행:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

Semantic `.npz` manifest를 직접 쓸 때:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest data/raw/real_manifest.json `
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

## 리뷰어가 채울 것

PNG batch 경로에서는 `outputs/reports/real_png_batch/review_template.csv`를 채운다.

최소 리뷰 권장:

- query당 top-5 neighbor
- 전체 20~50 pair

필수로 채우면 좋은 column:

- `reviewer_decision`
- `query_defect_family`
- `neighbor_defect_family`
- `dominant_defect`
- `clock_position_match`
- `missed_major_defect`
- `retrieval_failure_mode`
- `next_action`
- `review_comment`

## 리뷰 후 요약

```powershell
python scripts/summarize_expert_review.py `
  --review outputs/reports/real_png_batch/review_template.csv `
  --out outputs/reports/real_png_batch/review_summary.html `
  --metrics outputs/reports/real_png_batch/review_summary_metrics.json
```

## 나에게 주면 되는 결과

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

## 판단 기준

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

- 실제 batch 기본 검사 오류: `0`
- 실제 산출물 행 수: `features.csv`, `sanity.json`, `neighbors.csv`, `review_template.csv`가 `png_sample_count`와 모순되지 않음
- Reviewed query wafer: 최소 `20`, 권장 `50` 이상
- `accepted_match_rate`: `same_family` 또는 `partial_match` 비율 `70%` 이상
- `query_topk_accept_rate`: query별 top-k 안에 accept가 하나 이상 있는 비율 `80%` 이상
- `missed_major_defect_rate`: `5%` 이하
- `parser_or_mask_issue`: `0`
- Scratch/local/stby 관련 실패가 반복되면 해당 family는 별도 개선 track으로 분리한다.

## AI 모델 구현 상태

현재 deep-learning 주 경로는 pattern asset 기반 hybrid synthetic data와 multi-label segmentation이다. 실제 wafer 리뷰에서 반복적으로 놓치는 family를 확인한 뒤, 해당 family를 asset/label로 보강하고 U-Net 학습 데이터에 반영한다.
