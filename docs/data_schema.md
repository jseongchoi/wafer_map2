# Data Schema

## 1. Coordinate System

Wafer map은 2D array로 표현한다.

```text
y axis: top -> bottom
x axis: left -> right
wafer center: (cx, cy)
clock position:
  12:00 = negative y direction
  03:00 = positive x direction
  06:00 = positive y direction
  09:00 = negative x direction
```

위치 기반 report는 Cartesian coordinate와 polar coordinate를 함께 사용할 수 있어야 한다.

## 2. Geometry Units

```text
cell block: 가장 작은 Fail Bit Map 분석 단위
chip/die: cell block 직사각형 묶음
wafer: net die들이 원형/준원형으로 배열된 전체 map
```

초기 synthetic 기본값:

```yaml
target_net_die: 600
default_chip_blocks:
  width: 100
  height: 50
```

제품별 chip size와 die layout은 config에서 바꿀 수 있어야 한다.

## 3. Logical Values

사용자의 원본 parser mapping은 조정 가능하므로, 분석 내부에서는 시각화 값보다 semantic tensor를 우선한다.

권장 내부 표현:

```text
none wafer: wafer_mask = 0
in wafer: wafer_mask = 1
Grade 0, fail bit 0: severity = 0
Grade 1~7, fail bit count bucket: severity = 1~7
stby chip: stby_mask = 1, valid_test_mask = 0
measured chip: stby_mask = 0, valid_test_mask = 1
```

Stby는 Grade 7과 다르다. 시각화에서 흰색으로 표시하더라도 분석에서는 `stby_mask`로 분리한다.

## 4. Tensor Channels

최소 sample tensor:

```text
severity: uint8 [H, W], 0~7
wafer_mask: uint8 [H, W], 0/1
valid_test_mask: uint8 [H, W], 0/1
stby_mask: uint8 [H, W], 0/1
```

Synthetic 전용 validation tensor:

```text
pattern_masks: uint8 [C, H, W]
pattern_intensity: float32 [C, H, W]
chip_index: int32 [H, W]
```

`pattern_masks`는 synthetic validation label이다. 실제 wafer inference feature에 섞으면 label leakage가 되므로 report와 검증용으로만 사용한다.

선택 channel:

```text
radius_norm: [H, W]
theta_sin/theta_cos: [H, W]
die_boundary_mask: [H, W]
edge_distance: [H, W]
local_density: [H, W]
```

## 5. Synthetic Sample Files

저장 형식은 numpy와 JSON을 기본으로 한다.

```text
sample_id/
  arrays.npz
  metadata.json
  preview.png
```

`arrays.npz`:

```text
severity: uint8 [H, W]
wafer_mask: uint8 [H, W]
valid_test_mask: uint8 [H, W]
stby_mask: uint8 [H, W]
pattern_masks: uint8 [C, H, W]
pattern_intensity: float32 [C, H, W]
chip_index: int32 [H, W]
```

`metadata.json`:

```json
{
  "sample_id": "synth_000001",
  "target_net_die": 600,
  "chip_blocks": {"width": 100, "height": 50},
  "image_shape": {"height": 0, "width": 0},
  "patterns": [
    {
      "type": "scratch",
      "instance_id": 1,
      "clock_position": "12:00",
      "severity": 0.8,
      "parameters": {"mode": "spin_arc"}
    }
  ],
  "grade_thresholds": [0.055, 0.115, 0.205, 0.34, 0.53, 0.76, 1.05]
}
```

## 6. Feature Table

현재 feature table의 1차 목적은 FBM 정보추출, 유사 wafer 검색, coarse grouping, defect score report다.

예시 column:

```text
sample_id
product_id
lot_id
wafer_id
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
radial_zone_00_severity ... radial_zone_NN_severity
angular_sector_00_severity ... angular_sector_11_severity
```

향후 공정/설비/lot/recipe metadata가 조인되면 이 feature table을 ANOVA와 통계 검정에도 사용할 수 있다.

실제 보안 데이터는 repo에 저장하지 않고, feature schema와 extraction code만 관리한다.

## 7. Real-Unlabeled Manifest Schema

실제 wafer는 repo에 저장하지 않고, 보안 환경의 semantic `.npz`를 manifest로 참조한다.

Top-level 필수 field:

```json
{
  "schema_version": "real_unlabeled_manifest/v1",
  "feature_schema_version": "observable_fbm_features/v1",
  "samples": []
}
```

`npz_semantic_arrays` sample 필수 field:

```json
{
  "sample_id": "real_like_001",
  "source_type": "npz_semantic_arrays",
  "arrays_npz": "D:/secure_fbm/real_like_001_arrays.npz",
  "pseudonymized": true,
  "parser_name": "secure_fbm_parser",
  "parser_version": "0.1.0",
  "orientation": "not_rotated",
  "chip_blocks": {"width": 100, "height": 50},
  "grid": {"rows": 38, "cols": 20}
}
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

보안 규칙:

- `sample_id`에는 lot, wafer id, tool, recipe, chamber 같은 민감 정보를 넣지 않는다.
- `arrays_npz`와 optional `metadata_json`은 기본적으로 workspace 밖 경로여야 한다.
- real inference feature에는 `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`를 넣지 않는다.
- global nearest-neighbor에는 compact observable feature만 쓰며 `polar_*`, `stby_polar_*`는 제외한다.

Real `.npz` ingest validation:

- `severity`, `wafer_mask`, `valid_test_mask`, `stby_mask`는 required다.
- `chip_index`는 recommended이며 없으면 grid/chip block metadata로 추정한다.
- `array_keys`는 원본 key 이름이 표준 key와 다를 때만 필요하다.
- 배열은 semantic cast 전에 검사한다. float mask `0.5`, severity `1.5`, NaN/inf, 잘못된 shape는 reject한다.
- stby와 기타 invalid-test 영역은 `severity=0`이어야 한다. stby는 Grade 7이 아니다.
