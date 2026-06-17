# Real-Unlabeled Workflow

## 목적

실제 wafer FBM raw image/array를 repo에 저장하지 않고, 보안 환경 안의 semantic `.npz`만 참조해 feature와 review 산출물을 만든다.

```text
secure FBM arrays
-> real_unlabeled_manifest/v1
-> observable feature extraction
-> sanity / drift report
-> nearest-neighbor CSV
-> expert review template
```

Schema 세부 계약은 [Data Schema](data_schema.md)를 기준으로 한다.

## 입력 준비

보안 환경에서 sample별 `.npz`를 export한다.

필수 array:

- `severity`: Grade 0~7, `[H, W]`
- `wafer_mask`: wafer 내부 1, none-wafer 0
- `valid_test_mask`: 실제 test된 pixel 1
- `stby_mask`: stby fail chip 영역 1

권장 array:

- `chip_index`: die/chip id. wafer 밖은 -1

중요 의미:

- Stby는 Grade 7이 아니다.
- Stby는 `stby_mask=1`, `valid_test_mask=0`, `severity=0`이다.
- None-wafer와 in-wafer Grade 0은 `wafer_mask`로 구분한다.

## Manifest 예시

복사해서 수정할 수 있는 템플릿:

- 표준 key: `configs/eval/real_unlabeled_manifest_template_standard.json`
- key mapping 필요: `configs/eval/real_unlabeled_manifest_template_keymap.json`

표준 key를 쓰는 경우:

```json
{
  "schema_version": "real_unlabeled_manifest/v1",
  "feature_schema_version": "observable_fbm_features/v1",
  "samples": [
    {
      "sample_id": "real_like_001",
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

- `sample_id`는 익명 id만 사용한다.
- lot, wafer id, tool, recipe, chamber 같은 민감 정보를 넣지 않는다.
- `arrays_npz`와 `metadata_json`은 기본적으로 workspace 밖 보안 경로여야 한다.
- 테스트 목적으로 workspace 안 파일을 쓰려면 `allow_workspace_input=true`를 명시한다.

## 실행

Synthetic smoke:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --reference-features outputs/reports/fbm_grouping_scale_features.csv `
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
  --reference-features outputs/reports/fbm_grouping_scale_features.csv
```

## 산출물

Repo에 남겨도 되는 산출물:

- observable feature CSV
- sanity JSON
- nearest-neighbor CSV
- expert review template CSV
- HTML report

Repo에 남기지 않는 것:

- 실제 wafer raw image
- 실제 wafer raw array
- 실제 file path가 포함된 report
- lot/process/tool/chamber/recipe 민감 정보

## Sanity / Drift 해석

Sanity check는 parser와 semantic contract 검증이다.

확인 항목:

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
- 가장 크게 달라진 observable feature

Drift summary는 성능 metric이 아니다. 실제 wafer가 synthetic reference 분포와 얼마나 다른지 보는 sanity signal이다.

## Expert Review 연결

`--reference-features`를 넣으면 nearest-neighbor CSV와 reviewer 입력용 template CSV가 함께 생성된다.

Template에는 reviewer가 채울 빈 field만 들어간다.

- `reviewer_decision`
- `query_defect_family`
- `neighbor_defect_family`
- `dominant_defect`
- `clock_position_match`
- `missed_major_defect`
- `retrieval_failure_mode`
- `next_action`
- `safe_comment`

Reference의 `label_*` 컬럼은 reviewer bias 방지를 위해 template에 복사하지 않는다.

Review 절차는 [Expert Review Protocol](expert_review_protocol.md)을 따른다.
