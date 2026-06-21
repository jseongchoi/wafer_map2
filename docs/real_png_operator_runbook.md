# 실제 raw PNG 운영 안내서

## 결론

전제 조건을 만족하면 사용자는 **제품별 raw PNG 폴더와 제품별 geometry JSON을 준비하고, 실행 결과에서 안전한 산출물 또는 요약값만 공유**하면 된다.

공유하면 안 되는 것은 원본 PNG, 실제 경로가 들어간 manifest, lot/tool/recipe/chamber/wafer id다.
분석에 필요한 정보는 `outputs/reports/real_png_batch` 아래의 파생 산출물과 아래의 결과 공유 템플릿으로 충분해야 한다.

## 역할 분리

사용자가 하는 일:

1. 보안 환경에 제품별 raw PNG 폴더를 둔다.
2. 제품별 `product_geometry.json`을 만든다.
3. 운영 실행 명령을 실행한다.
4. `report.html`, `sanity.json`, `features.csv`, `neighbors.csv`, `review_template.csv`를 확인한다.
5. 리뷰 가능한 pair를 채우고 summary를 만든다.
6. 안전한 결과만 공유한다.

코드가 하는 일:

1. PNG gray value를 정해진 기준 그대로 읽는다.
2. full-255 chip을 stby fail chip으로 분리한다.
3. 실제 경로가 들어간 목록 파일은 `outputs/private`에 만든다.
4. feature, 기본 검사 결과, nearest-neighbor, 리뷰 양식을 만든다.
5. CPU encoder 결과는 리뷰 우선순위 참고값으로만 붙인다.
6. 자동 점검 스크립트로 단계별 산출물과 보안 누락을 검사한다.

## 사전 조건

아래 파일이 있어야 한다.

- `outputs/pre_real_readiness/reports/synthetic_reference_features.csv`
- `outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz`

없으면 먼저 실행한다.

```powershell
python scripts/run_pre_real_readiness.py `
  --config configs/synth/debug.json `
  --out-root outputs/pre_real_readiness `
  --count 20
```

CPU 환경에서는 시간이 걸릴 수 있다. 이 실행은 실제 성능 증명이 아니라 real PNG 실행 전 reference와 모델 산출물을 준비하는 단계다.

## 입력 폴더

보안 환경의 raw root는 workspace 밖에 둔다.

```text
D:/secure_fbm/raw_png/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_001.png
```

PNG 조건:

- 8-bit grayscale PNG
- 허용 gray value: `0, 31, 151, 175, 191, 207, 223, 255`
- wafer 밖도 `0`이어도 된다.
- chip block 전체가 `255`이면 stby fail chip이다.
- chip 일부 pixel만 `255`이면 Grade 7이다.

## Geometry JSON

운영 실행에서는 자동 geometry 추론을 성공 근거로 쓰지 않는다.
제품 담당자가 승인한 값을 `product_geometry.json`으로 둔다.
key는 제품 폴더명과 같아야 한다.

```json
{
  "product_a": {
    "chip_blocks": { "width": 100, "height": 50 },
    "grid": { "rows": 38, "cols": 20 },
    "actual_net_die": 600
  },
  "product_b": {
    "chip_blocks": { "width": 100, "height": 50 },
    "grid": { "rows": 38, "cols": 20 },
    "actual_net_die": 600
  }
}
```

필수 조건:

- `chip_blocks.width`와 `chip_blocks.height`는 양수다.
- `grid.rows`와 `grid.cols`는 양수다.
- `actual_net_die`는 양수다.
- `actual_net_die`는 `grid.rows * grid.cols`보다 클 수 없다.
- `actual_net_die=0`은 운영 승인값으로 보지 않는다.

## 운영 실행

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root D:/secure_fbm/raw_png `
  --production-run `
  --geometry-json D:/secure_fbm/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

`--production-run` 옵션은 실제 운영 실행이라는 뜻이며, 다음 조건을 강제한다.

- raw root는 workspace 밖이어야 한다.
- manifest는 `outputs/private` 아래에 있어야 한다.
- 리포트 출력은 `outputs/reports` 아래에 있어야 한다.
- `--geometry-json`이 있어야 한다.
- 제품별 양수 `actual_net_die`가 있어야 한다.
- `--reference-features`가 있어야 한다.
- 개발용 입력/출력 예외 옵션을 사용할 수 없다.

## 성공 시 산출물

공유 가능한 산출물:

- `outputs/reports/real_png_batch/batch_metadata.json`
- `outputs/reports/real_png_batch/features.csv`
- `outputs/reports/real_png_batch/sanity.json`
- `outputs/reports/real_png_batch/report.html`
- `outputs/reports/real_png_batch/neighbors.csv`
- `outputs/reports/real_png_batch/review_template.csv`
- `outputs/reports/real_png_batch/cpu_encoder_predictions.csv`
- `outputs/reports/real_png_batch/cpu_encoder_sanity.json`
- `outputs/reports/project_readiness_audit.json`
- `outputs/reports/project_readiness_audit.html`

공유 금지:

- 원본 raw PNG
- `outputs/private/*_manifest.json`
- 실제 file path가 들어간 manifest 원본
- 제품명, lot, wafer, tool, recipe, chamber 식별자
- 폴더 전체 압축본

## 실행 후 자동 점검

```powershell
python scripts/audit_project_readiness.py
```

기대 상태:

- 0~5단계: `PASS`
- 6~7단계: `PASS` 또는 이유가 명확한 `CHECK`
- 8단계: 실제 batch 산출물이 있으면 `PASS` 또는 `CHECK`

8단계가 `CHECK`이면 `notes` 항목을 우선 본다.
대표 원인은 기본 검사 오류, 산출물 행 수 불일치, 운영 실행 정보 불일치, 비공개 경로 노출이다.

## 리뷰 작성

`outputs/reports/real_png_batch/review_template.csv` 리뷰 양식을 채운다.

최소 작성 기준:

- query wafer 최소 20개, 권장 50개 이상
- query당 top-k neighbor 중 의미 있는 쌍을 우선 확인
- `reviewer_decision`은 가능하면 반드시 채움
- `missed_major_defect`, `retrieval_failure_mode`, `next_action`도 가능하면 채움
- `safe_comment`에는 민감정보를 쓰지 않음

리뷰 요약:

```powershell
python scripts/summarize_expert_review.py `
  --review outputs/reports/real_png_batch/review_template.csv `
  --out outputs/reports/real_png_batch/review_summary.html `
  --metrics outputs/reports/real_png_batch/review_summary_metrics.json
```

## 결과 공유 템플릿

보안상 파일을 직접 공유할 수 없으면 아래 항목만 손으로 옮겨 적는다.

```text
1. 실행 정보
- 실행 날짜:
- raw root는 workspace 밖이었는가: yes/no
- production_run: true/false
- 제품 수:
- PNG sample 수:
- geometry_contract:
- explicit_geometry_product_count:
- actual_net_die_product_count:
- wafer_mask_strategy:
- reference_features: true/false
- cpu_model_scoring: true/false

2. 제품별 geometry
- product alias 또는 익명 product index:
- chip_blocks: width=?, height=?
- grid: rows=?, cols=?
- actual_net_die:

3. sanity 요약
- sanity error sample 수:
- 전체 sanity error 수:
- warning 종류:
- grade_min 범위:
- grade_max 범위:
- stby_chip_count_est 범위:
- chip_index_die_count 범위:

4. output count
- features.csv row 수:
- sanity.json sample 수:
- neighbors.csv row 수:
- review_template.csv row 수:
- cpu_encoder_predictions.csv row 수:

5. 자동 점검 결과
- 전체 상태:
- 상태별 개수:
- 8단계 상태:
- 8단계 메모:
- 공유 산출물 보안 문제 수:

6. 리뷰 요약
- reviewed query wafer 수:
- reviewed pair 수:
- accepted_match_rate:
- query_topk_accept_rate:
- missed_major_defect_rate:
- parser_or_mask_issue 수:
- top failure modes:
- next_action_queue 상위 항목:

7. 민감정보 점검
- raw PNG 공유 없음: yes/no
- private manifest 공유 없음: yes/no
- 실제 path 공유 없음: yes/no
- lot/tool/recipe/chamber/wafer id 공유 없음: yes/no
- safe_comment 민감어 없음: yes/no
```

## 실패 시 판단

기본 검사 오류가 있으면 모델이나 검색 결과를 해석하지 않는다.
먼저 parser, gray value, geometry, mask 기준을 고친다.

대표 원인:

- unknown gray value가 있다.
- 제품 폴더명이 geometry JSON key와 다르다.
- PNG shape가 `grid * chip_blocks`와 맞지 않는다.
- `actual_net_die`가 0이거나 grid보다 크다.
- stby chip이 붙어 있어 자동 추론이 애매하다.
- wafer 밖 `0`과 in-wafer good `0`을 구분할 mask 기준이 부족하다.

8단계가 `PASS`가 되어도 즉시 판매 가능한 상태는 아니다.
전문가 리뷰 기준을 통과해야 제한된 파일럿 결과로 볼 수 있다.
