# 데이터 형식

변수 이름이 낯설면 [용어와 변수 설명](glossary.md)을 먼저 본다.

## 1. 좌표 기준

Wafer map은 2D array로 표현한다.

```text
y axis: top -> bottom
x axis: left -> right
wafer center: (cx, cy)
clock position:
  12:00 = negative y
  03:00 = positive x
  06:00 = positive y
  09:00 = negative x
```

위치 기반 report는 Cartesian coordinate와 polar coordinate를 함께 사용할 수 있다.

## 2. 분석용 배열

분석 내부에서는 시각화 색보다 의미가 명확한 배열을 우선한다.

필수 channel:

| Channel | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `severity` | `uint8` | `[H, W]` | Grade 0~7. wafer 밖과 stby는 0. |
| `wafer_mask` | `uint8`/bool | `[H, W]` | wafer 내부 1, wafer 밖 0. |
| `valid_test_mask` | `uint8`/bool | `[H, W]` | 실제 test된 pixel 1. |
| `stby_mask` | `uint8`/bool | `[H, W]` | stby fail chip 영역 1. |

권장 channel:

| Channel | dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `chip_index` | `int32` | `[H, W]` | die/chip id. wafer 밖은 -1. |

핵심 규칙:

- Stby는 Grade 7이 아니다.
- Stby는 `stby_mask=1`, `valid_test_mask=0`, `severity=0`으로 표현한다.
- Wafer 밖 영역과 in-wafer Grade 0은 `wafer_mask`로 구분한다.
- `pattern_masks`, `pattern_intensity`는 합성 데이터 검증용 label이며 실제 inference feature가 아니다.

## 3. 실제 raw PNG gray 기준

실제 raw PNG를 직접 읽는 경우 gray value는 아래 기준으로 해석한다.

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

`255`는 pixel 단독으로 바로 stby로 보지 않는다.
한 chip block 전체가 `255`이고 block 내부 `min=max=255`일 때만 stby fail chip으로 분리한다.
그 chip은 내부 배열에서 `stby_mask=1`, `valid_test_mask=0`, `severity=0`이 된다.
일부 pixel만 `255`인 경우는 Grade 7 fail로 유지한다.

PNG 밖 영역과 in-wafer Grade 0이 모두 `0`일 수 있으므로, PNG 입력은 기본적으로 제품별 chip geometry를 기준으로 centered ellipse wafer mask를 추정한다.
제품 die layout이 centered ellipse와 다르면 별도 geometry 또는 mask export가 필요하다.
`png_grayscale_raw` manifest는 기본적으로 `chip_blocks`와 `grid`를 포함해야 한다. geometry를 모르는 smoke/inference 경로만 `allow_geometry_inference=true`를 명시하고, STBY 255 직사각형이 붙어 있어 애매하면 `--geometry-json`으로 보완한다.

## 4. Geometry Metadata

제품별 chip size와 die layout은 metadata로 명시한다.

```json
{
  "chip_blocks": { "width": 100, "height": 50 },
  "grid": { "rows": 38, "cols": 20 },
  "actual_net_die": 600
}
```

`chip_index`가 없으면 `chip_blocks`와 `grid`로 추정할 수 있지만, 실제 작업에서는 export하는 편이 더 안전하다.

## 5. 합성 Sample 형식

Synthetic sample은 repo 밖에서 재생성 가능한 산출물이다.

```text
sample_id/
  arrays.npz
  metadata.json
  raw_grayscale.png
  preview.png
```

`arrays.npz`:

```text
severity
wafer_mask
valid_test_mask
stby_mask
chip_index
pattern_masks        # synthetic validation only
pattern_intensity    # synthetic validation only
```

`raw_grayscale.png`는 실제 raw PNG gray 기준으로 렌더링한 검증용 파일이다.
분석 내부의 `severity`는 계속 Grade 0~7을 사용한다.

## 6. 라벨 없는 실제 Wafer Manifest

실제 wafer는 raw PNG 또는 `.npz`를 manifest로 참조한다. 파일은 `data/raw`, 인트라넷 공유 폴더, 로컬 절대경로 중 편한 위치에 둘 수 있다.

Top-level:

```json
{
  "schema_version": "real_unlabeled_manifest/v1",
  "feature_schema_version": "observable_fbm_features/v1",
  "samples": []
}
```

`npz_semantic_arrays` sample:

```json
{
  "sample_id": "product_a_wafer_001",
  "source_type": "npz_semantic_arrays",
  "arrays_npz": "data/raw/product_a/wafer_001_arrays.npz",
  "parser_name": "fbm_npz_parser",
  "parser_version": "0.1.0",
  "orientation": "not_rotated",
  "chip_blocks": { "width": 100, "height": 50 },
  "grid": { "rows": 38, "cols": 20 }
}
```

`png_grayscale_raw` sample:

```json
{
  "sample_id": "product_a_wafer_001",
  "source_type": "png_grayscale_raw",
  "png_path": "data/raw/product_a/wafer_001.png",
  "parser_name": "png_raw_folder_batch",
  "parser_version": "0.1.0",
  "orientation": "not_rotated",
  "wafer_mask_strategy": "centered_ellipse_from_png",
  "chip_blocks": { "width": 100, "height": 50 },
  "grid": { "rows": 38, "cols": 20 }
}
```

제품별 PNG 폴더를 일괄 처리할 때는 manifest를 직접 만들지 않고 `scripts/analyze_png_raw_folders.py`가 생성한다.

```text
raw_root/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_001.png
```

원본 array key가 표준 key와 다르면 `array_keys`를 사용한다.

```json
{
  "array_keys": {
    "severity": "grade",
    "wafer_mask": "in_wafer",
    "valid_test_mask": "valid",
    "stby_mask": "stby",
    "chip_index": "die_id"
  }
}
```

Manifest 기준:

- `sample_id`는 비어 있지 않으면 된다. 사람이 읽기 쉬운 wafer 이름을 써도 된다.
- `png_path`, `arrays_npz`, optional `metadata_json`은 상대경로와 절대경로를 모두 허용한다.
- `scripts/analyze_png_raw_folders.py`의 기본 manifest 위치는 `outputs/manifests/<run_name>_manifest.json`이다.

## 7. Feature Table 기준

현재 feature table의 목적:

- similar wafer retrieval
- coarse grouping
- defect score report
- expert review triage
- downstream model input

기본 column:

```text
sample_id
actual_net_die
total_fail_density
grade_weighted_severity
stby_ratio
edge_density
center_density
edge_sector_peak_contrast
ring_radial_peak_contrast
scratch_angular_peak_contrast
local_hotspot_peak_contrast
shot_lower_left_contrast
shot_bottom_edge_contrast
shot_left_edge_contrast
...
```

전체 유사 wafer 검색 feature 기준:

```text
compact feature 50개
```

전체 유사 wafer 검색에서 제외:

```text
label_*
*_mask_ratio
pattern_masks
pattern_intensity
polar_*
stby_polar_*
```

`polar_*`, `stby_polar_*`는 `class_location`, `feature_key`처럼 위치가 중요한 검색에서만 조건부로 쓴다.

현재 repo feature table에는 민감한 lot/tool/recipe/chamber id를 넣지 않는다.
