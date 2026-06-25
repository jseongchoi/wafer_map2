# 실제 raw PNG 운영 안내서

이 문서는 실제 FBM raw PNG를 프로젝트 파이프라인에 넣는 방법만 정리합니다. 경로 정책, 외부 반출, 접근 권한은 사용 환경에서 별도로 관리한다고 가정합니다.

## 입력 폴더

기본 개발 위치:

```text
data/raw/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_101.png
```

인트라넷 운영 위치는 원하는 경로를 그대로 쓸 수 있습니다.

```text
Z:/fbm/raw_png/
  product_a/
    wafer_001.png
    wafer_002.png
```

PNG 조건:

- 8-bit grayscale PNG
- 허용 gray value: `0, 31, 151, 175, 191, 207, 223, 255`
- chip block 전체가 `255`이면 stby fail chip
- chip 일부 pixel만 `255`이면 grade 7

## Geometry JSON

제품별 chip 크기와 grid 정보를 JSON으로 둡니다. 자동 추론도 가능하지만, 실제 제품에서는 geometry JSON을 쓰는 편이 재현성이 좋습니다.

```json
{
  "product_a": {
    "chip_blocks": { "width": 100, "height": 50 },
    "grid": { "rows": 38, "cols": 20 },
    "actual_net_die": 600
  }
}
```

## 실행

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

`--reference-features`와 `--cpu-model`은 선택입니다. 처음에는 빼고 parser와 sanity부터 확인해도 됩니다.

## 생성 산출물

```text
outputs/manifests/real_png_batch_manifest.json
outputs/reports/real_png_batch/batch_metadata.json
outputs/reports/real_png_batch/features.csv
outputs/reports/real_png_batch/sanity.json
outputs/reports/real_png_batch/report.html
outputs/reports/real_png_batch/neighbors.csv
outputs/reports/real_png_batch/review_template.csv
outputs/reports/real_png_batch/cpu_encoder_predictions.csv
outputs/reports/real_png_batch/cpu_encoder_sanity.json
```

## 실행 후 확인 순서

1. `sanity.json`에서 unknown gray value, shape mismatch, geometry 문제를 확인합니다.
2. `report.html`에서 wafer가 정상적으로 렌더링되는지 봅니다.
3. `features.csv` row 수가 입력 PNG 수와 맞는지 봅니다.
4. reference feature를 붙였다면 `neighbors.csv`에서 유사 wafer 후보가 생성됐는지 봅니다.
5. CPU model을 붙였다면 `cpu_encoder_predictions.csv`에서 확률값이 생성됐는지 봅니다.

## 리뷰 작성

`outputs/reports/real_png_batch/review_template.csv`를 열고 사람이 판단합니다.

권장 작성 기준:

- query wafer 최소 20개, 권장 50개 이상
- query당 top-k neighbor 중 의미 있는 쌍 확인
- `reviewer_decision` 입력
- `missed_major_defect`, `retrieval_failure_mode`, `next_action` 입력

리뷰 요약:

```powershell
python scripts/summarize_expert_review.py `
  --review outputs/reports/real_png_batch/review_template.csv `
  --out outputs/reports/real_png_batch/review_summary.html `
  --metrics outputs/reports/real_png_batch/review_summary_metrics.json
```

## 실패 시 판단

기본 검사 오류가 있으면 모델 결과를 해석하지 말고 parser/geometry부터 고칩니다.

대표 원인:

- unknown gray value가 있다.
- 제품 폴더명이 geometry JSON key와 다르다.
- PNG shape가 `grid * chip_blocks`와 맞지 않는다.
- `actual_net_die`가 0이거나 grid보다 크다.
- `actual_net_die=0`이면 제품 geometry가 비어 있거나 net die 산정 기준이 누락된 상태다.
- stby chip과 grade 7 chip 기준이 섞였다.
- wafer 밖 `0`과 in-wafer good `0`을 구분할 mask 기준이 부족하다.
