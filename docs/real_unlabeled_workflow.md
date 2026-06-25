# 라벨 없는 실제 Wafer 처리 절차

## 목적

라벨 없는 실제 wafer FBM raw image/array를 읽어서 feature, sanity report, nearest-neighbor 결과, 전문가 리뷰 양식을 만든다.

```text
제품별 raw PNG 폴더 또는 FBM 배열
-> manifest 자동 생성 또는 real_unlabeled_manifest/v1
-> feature 추출
-> sanity / drift report
-> nearest-neighbor CSV
-> 전문가 리뷰 양식
```

Schema 세부 기준은 [데이터 형식](data_schema.md)을 따른다. 실제 raw PNG 실행 순서는 [실제 raw PNG 운영 안내서](real_png_operator_runbook.md)을 우선 따른다.

## 입력 준비 A. 제품별 raw PNG 폴더

현재 실제 raw data가 8-bit grayscale PNG라면 이 경로를 우선 사용한다.

```text
data/raw/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_101.png
```

인트라넷 공유 폴더나 다른 드라이브 경로도 그대로 사용할 수 있다.

```text
Z:/fbm/raw_png/
  product_a/
    wafer_001.png
```

PNG gray value 기준:

| Gray value | 의미 |
| --- | --- |
| `0` | Grade 0, good |
| `31` | Grade 1 |
| `151` | Grade 2 |
| `175` | Grade 3 |
| `191` | Grade 4 |
| `207` | Grade 5 |
| `223` | Grade 6 |
| `255` | Grade 7 또는 stby 후보 |

Stby 판정:

- chip block 전체가 `255`이면 stby fail chip으로 본다.
- 일부 pixel만 `255`이면 Grade 7 fail로 본다.
- stby chip은 내부 배열에서 `stby_mask=1`, `valid_test_mask=0`, `severity=0`으로 변환된다.

제품별 chip size와 grid는 full-255 stby block에서 먼저 추론한다. 추론이 실패하거나 제품 기준을 고정하고 싶으면 `--geometry-json`으로 제품별 `chip_blocks`, `grid`, 필요 시 `actual_net_die`를 준다.

## 입력 준비 B. Semantic `.npz`

sample별 semantic array export가 가능하면 `.npz` manifest도 사용할 수 있다.

필수 array:

- `severity`: Grade 0~7, `[H, W]`
- `wafer_mask`: wafer 내부 1, wafer 밖 0
- `valid_test_mask`: 실제 test된 pixel 1
- `stby_mask`: stby fail chip 영역 1

권장 array:

- `chip_index`: die/chip id. wafer 밖은 -1

## Manifest 예시

```json
{
  "schema_version": "real_unlabeled_manifest/v1",
  "feature_schema_version": "observable_fbm_features/v1",
  "samples": [
    {
      "sample_id": "product_a_wafer_001",
      "source_type": "npz_semantic_arrays",
      "arrays_npz": "data/raw/product_a/wafer_001_arrays.npz",
      "parser_name": "fbm_npz_parser",
      "parser_version": "0.1.0",
      "orientation": "not_rotated",
      "chip_blocks": { "width": 100, "height": 50 },
      "grid": { "rows": 38, "cols": 20 },
      "actual_net_die": 600
    }
  ]
}
```

원본 key 이름이 다르면 `array_keys`로 매핑한다.

```json
"array_keys": {
  "severity": "grade",
  "wafer_mask": "in_wafer",
  "valid_test_mask": "valid",
  "stby_mask": "stby",
  "chip_index": "die_id"
}
```

Manifest 기준:

- `sample_id`는 비어 있지 않으면 된다.
- 상대경로와 절대경로를 모두 허용한다.
- PNG batch manifest 기본 위치는 `outputs/manifests/<out-dir-name>_manifest.json`이다.

## 실행

제품별 raw PNG 폴더 실행:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

Synthetic smoke:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --features-out outputs/reports/real_unlabeled_features.csv `
  --sanity-out outputs/reports/real_unlabeled_sanity.json `
  --report-out outputs/reports/real_unlabeled_report.html `
  --neighbors-out outputs/reports/real_unlabeled_neighbors.csv `
  --review-template-out outputs/reports/real_unlabeled_expert_review_template.csv
```

직접 만든 manifest:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest data/raw/real_manifest.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv
```

## 산출물

- `features.csv`
- `sanity.json`
- `batch_metadata.json`
- `neighbors.csv`
- `review_template.csv`
- `report.html`

## Sanity / Drift 해석

Sanity check는 parser와 입력 배열이 정해진 형식을 지키는지 확인하는 단계다.

확인 항목:

- PNG gray value가 허용 값만 포함하는지
- PNG stby chip이 full-255 chip 단위로 분리되는지
- 제품별 chip size/grid 추론 또는 명시값이 맞는지
- severity/mask shape 일치
- severity 0~7
- mask binary
- wafer 밖 severity 0
- invalid-test severity 0
- stby는 wafer 안, valid-test 밖, severity 0
- chip_index는 wafer 밖 -1, wafer 안 non-negative

Reference feature를 주면 drift summary를 만든다.

- query/reference compact feature 평균 차이
- reference 표준편차 기준 z-score shift
- 가장 크게 달라진 feature

Drift summary는 성능 metric이 아니다. 실제 wafer가 synthetic reference 분포와 얼마나 다른지 보는 sanity signal이다.

## 전문가 리뷰 연결

`review_template.csv`를 채운 뒤 아래 명령으로 요약한다.

```powershell
python scripts/summarize_expert_review.py `
  --review outputs/reports/real_png_batch/review_template.csv `
  --out outputs/reports/real_png_batch/review_summary.html `
  --metrics outputs/reports/real_png_batch/review_summary_metrics.json
```
