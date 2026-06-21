# 라벨 없는 실제 Wafer 처리 절차

## 목적

실제 wafer FBM raw image/array를 repo에 저장하지 않고, 보안 환경 안의 제품별 raw PNG 폴더 또는 semantic `.npz`를 참조해 feature와 리뷰 산출물을 만든다.

```text
보안 환경의 제품별 raw PNG 폴더 또는 FBM 배열
-> manifest 자동 생성 또는 real_unlabeled_manifest/v1
-> feature 추출
-> sanity / drift report
-> nearest-neighbor CSV
-> 전문가 리뷰 양식
```

Schema 세부 기준은 [데이터 형식](data_schema.md)을 따른다.
실제 raw PNG를 넣고 결과를 공유하는 실행 순서는 [실제 raw PNG 운영 안내서](real_png_operator_runbook.md)을 우선 따른다.

## 입력 준비 A. 제품별 raw PNG 폴더

현재 실제 raw data가 8-bit grayscale PNG라면 이 경로를 우선 사용한다.

폴더 구조:

```text
D:/secure_fbm/raw_png/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
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

제품별 chip size와 grid는 full-255 stby block에서 먼저 추론한다.
stby chip이 없는 제품이거나 추론이 실패하는 제품은 `--geometry-json`으로 제품별 `chip_blocks`, `grid`, 필요 시 `actual_net_die`를 준다. STBY fail chip이 서로 붙어 하나의 큰 255 직사각형처럼 보이면 chip size가 애매하므로 자동 추론을 중단하고 `--geometry-json`을 요구한다.

```json
{
  "product_a": {
    "chip_blocks": { "width": 100, "height": 50 },
    "grid": { "rows": 38, "cols": 20 },
    "actual_net_die": 600
  }
}
```

## 입력 준비 B. Semantic `.npz`

이미 보안 환경에서 sample별 semantic array export가 가능하면 `.npz` 경로도 계속 사용할 수 있다.

필수 array:

- `severity`: Grade 0~7, `[H, W]`
- `wafer_mask`: wafer 내부 1, wafer 밖 0
- `valid_test_mask`: 실제 test된 pixel 1
- `stby_mask`: stby fail chip 영역 1

권장 array:

- `chip_index`: die/chip id. wafer 밖은 -1

중요 의미:

- Stby는 Grade 7이 아니다.
- Stby는 `stby_mask=1`, `valid_test_mask=0`, `severity=0`이다.
- Wafer 밖 영역과 in-wafer Grade 0은 `wafer_mask`로 구분한다.

## Manifest 예시

복사해서 수정할 수 있는 템플릿:

- 표준 key: `configs/eval/real_unlabeled_manifest_template_standard.json`
- key mapping 필요: `configs/eval/real_unlabeled_manifest_template_keymap.json`
- raw PNG: `configs/eval/real_unlabeled_manifest_template_png.json`

표준 key를 쓰는 경우:

```json
{
  "schema_version": "real_unlabeled_manifest/v1",
  "feature_schema_version": "observable_fbm_features/v1",
  "samples": [
    {
      "sample_id": "product_aaaaaaaaaa_wbbbbbbbbbb",
      "source_type": "npz_semantic_arrays",
      "arrays_npz": "D:/secure_fbm/real_like_001_arrays.npz",
      "pseudonymized": true,
      "parser_name": "secure_fbm_parser",
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

보안 규칙:

- `sample_id`는 `product_<10hex>_w<10hex>` 형식의 opaque id만 사용한다.
- lot, wafer id, tool, recipe, chamber 같은 민감 정보를 넣지 않는다.
- `png_path`, `arrays_npz`, `metadata_json`은 기본적으로 workspace 밖 보안 경로여야 한다.
- 제품별 PNG batch가 만든 manifest에는 실제 path가 들어가므로 원본 manifest는 공유하지 않는다.
- PNG batch manifest 기본 위치는 `outputs/private/<out-dir-name>_manifest.json`이다. `outputs/reports/...` 공유용 report 폴더와 분리한다.
- 테스트 목적으로 workspace 안 파일을 쓰려면 `allow_workspace_input=true`를 명시한다.

## 실행

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

제품명이 sample id에 드러나면 안 되는 경우 기본값을 그대로 쓴다. 이때 sample id는 `product_<hash>_w<hash>`처럼 익명 alias로 생성된다.
제품 폴더명은 sample id에 포함하지 않는다. batch script는 공유 산출물에 제품명이 노출되지 않도록 opaque alias만 생성한다.

연습 실행에서는 `--production-run`을 빼고 자동 geometry 추론을 확인할 수 있다. 실제 운영 실행에서는 `--production-run`이 `--geometry-json`, 양수 `actual_net_die`, private manifest, `outputs/reports` 출력, reference feature를 강제한다.
`actual_net_die=0`은 운영 geometry 승인값으로 보지 않는다.

제품별 geometry JSON 예시:

```powershell
Get-Content D:/secure_fbm/product_geometry.json
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

실제 보안 path:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest D:/secure_fbm/real_manifest.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv
```

## 산출물

Repo에 남겨도 되는 산출물:

- feature CSV
- sanity JSON
- batch metadata JSON
- nearest-neighbor CSV
- 전문가 리뷰 양식 CSV
- HTML report

Repo에 남기지 않는 것:

- 실제 wafer raw image
- 실제 wafer raw array
- 실제 file path가 포함된 manifest 원본, 특히 `outputs/private/*_manifest.json`
- lot/process/tool/chamber/recipe 민감 정보

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

`--reference-features`를 넣으면 nearest-neighbor CSV와 reviewer 입력용 리뷰 양식 CSV가 함께 생성된다.

리뷰 양식에는 reviewer가 채울 빈 field만 들어간다.

- `reviewer_decision`
- `query_defect_family`
- `neighbor_defect_family`
- `dominant_defect`
- `clock_position_match`
- `missed_major_defect`
- `retrieval_failure_mode`
- `next_action`
- `safe_comment`

Reference의 `label_*` 컬럼은 reviewer bias를 막기 위해 리뷰 양식에 복사하지 않는다.

Review 절차는 [전문가 리뷰 절차](expert_review_protocol.md)을 따른다.

## 나에게 공유할 최소 정보

보안상 원본 PNG, 실제 path, manifest 원본은 공유하지 않는다.
문제가 생겼을 때는 아래 값만 옮겨 적어도 원인 파악이 가능하다.

- 전체 sample 수
- sanity error가 있는 sample 수
- warning 종류
- 제품별 `chip_blocks`, `grid`
- `stby_chip_count_est` 범위
- `grade_min`, `grade_max`
- `chip_index_die_count` 범위
- 사용한 옵션: `--geometry-json` 사용 여부, `--wafer-mask-strategy` 값
