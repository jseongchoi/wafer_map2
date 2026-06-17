# Real-Unlabeled Workflow

## 목적

실제 wafer FBM은 보안상 repo에 저장하지 않는다. 이 workflow는 보안 환경 안의 local path를 입력으로 받아, repo에는 원본 wafer가 아닌 파생 산출물만 남기기 위한 최소 구조다.

```text
secure real FBM arrays
-> semantic tensor manifest
-> observable feature extraction
-> sanity check
-> optional nearest-neighbor review
-> feature/report outputs only
```

## 핵심 원칙

- 실제 raw wafer image/array는 repo에 복사하지 않는다.
- 실제 inference feature는 observable-only다.
- synthetic oracle인 `*_mask_ratio`, `label_*`, `pattern_masks`는 real inference에 사용하지 않는다.
- Stby Fail Chip은 Grade 7이 아니다.
- Stby는 `stby_mask=1`, `valid_test_mask=0`, `severity=0`으로 semantic parsing한다.
- None-wafer와 in-wafer Grade 0은 모두 시각적으로 검정일 수 있지만, `wafer_mask`로 분리한다.

## Manifest 입력 계약

현재 스크립트:

```text
scripts/extract_real_unlabeled_features.py
```

지원 입력:

1. `npz_semantic_arrays`
   - 보안 환경에서 semantic array로 export한 `.npz`
   - 필수: `severity`
   - 필수: `wafer_mask`, `valid_test_mask`, `stby_mask`
   - 권장: `chip_index`
   - 필수 metadata: `chip_blocks`, `grid`

2. `synthetic_sample_dir`
   - real data 없이 workflow를 검증하기 위한 adapter
   - 기존 synthetic sample folder를 real-like input처럼 처리한다.

예시:

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

Array key가 다르면 다음처럼 매핑한다.

```json
{
  "samples": [
    {
      "sample_id": "real_like_002",
      "source_type": "npz_semantic_arrays",
      "arrays_npz": "D:/secure_fbm/real_like_002_arrays.npz",
      "pseudonymized": true,
      "parser_name": "secure_fbm_parser",
      "parser_version": "0.1.0",
      "orientation": "not_rotated",
      "array_keys": {
        "severity": "grade",
        "wafer_mask": "in_wafer",
        "stby_mask": "stby",
        "valid_test_mask": "valid"
      },
      "chip_blocks": { "width": 100, "height": 50 },
      "grid": { "rows": 38, "cols": 20 }
    }
  ]
}
```

필수 필드:

- top-level: `schema_version`, `feature_schema_version`, `samples`
- sample: `sample_id`, `source_type`, `arrays_npz`, `chip_blocks`, `grid`, `parser_name`, `parser_version`, `orientation`
- optional sample: `array_keys`, `metadata_json`, `actual_net_die`
- real sample은 `pseudonymized=true`를 명시해야 한다.
- `sample_id`는 익명 id만 허용한다. lot, wafer id, tool id, recipe 등 민감 정보를 넣지 않는다.

## 실행 예시

Synthetic sample로 workflow smoke test:

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

실제 보안 path 사용:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest D:/secure_fbm/real_manifest.json `
  --reference-features outputs/reports/fbm_grouping_scale_features.csv
```

## `.npz` Semantic Array Schema

보안 환경에서 export할 최소 `.npz`는 다음 semantic key를 가져야 한다.

| Key | Required | dtype | Shape | Meaning |
| --- | --- | --- | --- | --- |
| `severity` | yes | `uint8` 권장 | `[H, W]` | Grade 0~7 fail severity. wafer 밖과 stby는 0이어야 한다. |
| `wafer_mask` | yes | `uint8`/bool | `[H, W]` | wafer 내부 1, none-wafer 0. |
| `valid_test_mask` | yes | `uint8`/bool | `[H, W]` | 실제 test된 pixel 1. stby와 wafer 밖은 0. |
| `stby_mask` | yes | `uint8`/bool | `[H, W]` | stby fail chip 영역 1. Grade 7로 encode하지 않는다. |
| `chip_index` | recommended | `int32` | `[H, W]` | die/chip id. wafer 밖은 -1. 없으면 `grid`와 `chip_blocks`로 추정한다. |

Load 전 strict validation:

- `schema_version`은 `real_unlabeled_manifest/v1`이어야 한다.
- `feature_schema_version`은 `observable_fbm_features/v1`이어야 한다.
- sample entry에는 `source_type`이 명시되어야 한다.
- `severity`는 cast 전 2D, finite, integer-like, 0~7이어야 한다.
- mask는 cast 전 2D, severity와 같은 shape, binary 0/1 또는 bool이어야 한다.
- `chip_index`가 제공되면 cast 전 2D, severity와 같은 shape, integer-like여야 한다.
- `valid_test_mask=0`인 pixel의 `severity`는 0이어야 한다.

원본 key 이름이 다르면 manifest의 `array_keys`로 매핑한다.

```json
"array_keys": {
  "severity": "grade",
  "wafer_mask": "in_wafer",
  "valid_test_mask": "valid",
  "stby_mask": "stby",
  "chip_index": "die_id"
}
```

Real sample manifest 필수 원칙:

- `sample_id`는 익명 id만 사용한다.
- `pseudonymized=true`를 명시한다.
- `parser_name`, `parser_version`, `orientation`을 남겨 export logic 재현성을 확보한다.
- `chip_blocks.width`, `chip_blocks.height`, `grid.rows`, `grid.cols`를 반드시 넣는다.
- `arrays_npz`는 기본적으로 workspace 밖 보안 경로여야 한다.
- 테스트 목적 workspace 입력은 sample entry에 `allow_workspace_input=true`를 명시한 경우에만 허용한다.

## Sanity Check

현재 확인하는 항목:

- severity shape와 mask shape 일치
- severity가 0~7 범위
- mask가 binary
- wafer 밖 severity가 0
- stby가 wafer 안에 존재
- stby pixel은 valid test가 아님
- stby pixel severity는 0
- chip_index가 wafer 밖에서 -1
- `actual_net_die`와 `chip_index>=0` die count 일치
- stby area가 chip area의 정수배에 가까운지 확인
- valid tested pixel 존재
- measured Grade 0 존재 여부 warning
- stby 없음 warning

Reference feature CSV를 함께 입력하면 추가로 확인하는 항목:

- query feature와 reference feature의 평균 차이
- reference 표준편차 기준 z-score shift
- 가장 크게 달라진 compact observable feature 목록

이 drift summary는 성능 metric이 아니다. 실제 wafer가 synthetic reference 분포와 얼마나 다른지 빠르게 보는 sanity check다.
Global 비교에는 compact observable feature만 사용하며, `label_*`, `*_mask_ratio`, `polar_*`, `stby_polar_*`는 제외한다.

## 안전한 출력

Repo에 남겨도 되는 출력:

- observable feature CSV
- sanity JSON
- nearest-neighbor CSV
- expert review template CSV
- HTML report

주의:

- 실제 wafer image preview는 기본 출력하지 않는다.
- 필요하면 보안 환경 내부에서만 별도 gallery를 만든다.
- report에는 원본 파일 경로나 민감 lot/process metadata를 넣지 않는다.
- nearest-neighbor CSV는 기본적으로 reference의 `label_*` 컬럼을 복사하지 않는다.
- synthetic reference label을 검증용 neighbor CSV에 포함하려면 `--include-reference-labels`를 명시적으로 켠다. Expert review template은 reviewer bias 방지를 위해 label column을 복사하지 않는다.
- `npz_semantic_arrays` 입력은 기본적으로 workspace 밖 경로만 허용한다.
- 테스트 목적의 workspace 입력은 manifest에 `allow_workspace_input=true`를 둔 경우에만 허용한다.

## Expert Review 연결

`--reference-features`를 넣으면 workflow는 query feature를 reference feature store와 비교해 nearest-neighbor CSV를 만든다. 같은 실행에서 `--review-template-out` 경로에 reviewer 입력용 CSV도 생성한다.

Review template에는 다음만 들어간다.

- `query_sample_id`
- `neighbor_sample_id`
- `rank`
- `distance`
- 빈 reviewer 입력 필드
  - `reviewer_decision`
  - `query_defect_family`
  - `neighbor_defect_family`
  - `dominant_defect`
  - `clock_position_match`
  - `missed_major_defect`
  - `retrieval_failure_mode`
  - `next_action`
  - `safe_comment`

기본적으로 reference의 `label_*` 컬럼은 review template에 복사하지 않는다. `--include-reference-labels`를 켜면 neighbor CSV에는 label이 남을 수 있지만, template 생성 helper는 reviewer bias 방지를 위해 label column을 다시 제거한다.

## 다음 보강

- real feature aggregate와 synthetic feature aggregate 비교
- reference 대비 feature drift summary 추가 완료
- [expert review template](expert_review_protocol.md) 연결 완료
- top-k nearest-neighbor HTML table 강화
- class별 score threshold/calibration
- holdout synthetic stress test와 연결
