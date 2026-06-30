# 학습 데이터 규격

이 문서는 현재 multi-label U-Net 학습 코드가 기대하는 데이터 규격을
정의합니다.

다음 질문에 답하고 싶을 때 이 문서를 보면 됩니다.

```text
train_unet_segmentation.py를 돌리려면 어떤 파일이 필요한가?
arrays.npz 안에는 어떤 array가 들어가야 하나?
U-Net input channel은 무엇인가?
U-Net target channel은 무엇인가?
manual label, parametric label, synthetic label은 어떻게 학습 데이터가 되는가?
```

## 1. 큰 그림

학습 데이터는 사람이 작업하는 원천 라벨 폴더가 아닙니다. 학습 데이터는
asset, parametric rule, procedural pattern을 base wafer 위에 합성하고
rasterize한 최종 결과입니다.

```text
manual mask / parametric rule / pattern asset
-> base wafer 위에 합성
-> full-size family mask 생성
-> data/synthetic/<dataset>/<sample_id>/arrays.npz
-> asset_segmentation_manifest.csv
-> train_unet_segmentation.py
```

U-Net은 `bbox_xywh`나 parametric rule JSON을 직접 읽지 않습니다. 학습 전에
모든 결함은 `pattern_masks` 안의 full-size binary mask로 들어가 있어야
합니다.

## 2. dataset 폴더 구조

학습 dataset은 여러 sample folder로 구성됩니다.

```text
data/synthetic/<dataset_name>/
  synth_000001/
    arrays.npz
    metadata.json
    preview.png          # 선택
  synth_000002/
    arrays.npz
    metadata.json
    preview.png          # 선택
```

readiness 단계에서 manifest를 만듭니다.

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
```

이 manifest가 U-Net 학습의 직접 입력입니다.

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
```

## 3. arrays.npz 필수 key

각 sample folder의 `arrays.npz`에는 아래 key가 있어야 합니다.

| Key | dtype | Shape | 필수 | 의미 |
|---|---|---:|---|---|
| `severity` | `uint8` | `[H, W]` | 예 | 최종 wafer grade map, 값 0-7 |
| `wafer_mask` | `uint8` 또는 bool | `[H, W]` | 예 | wafer 내부 pixel |
| `valid_test_mask` | `uint8` 또는 bool | `[H, W]` | 예 | 실제 측정/관측 가능한 pixel |
| `stby_mask` | `uint8` 또는 bool | `[H, W]` | 예 | STBY 또는 missing-test context |
| `chip_index` | `int32` | `[H, W]` | 예 | die/chip id, 모르면 wafer 밖은 `-1` |
| `pattern_masks` | `uint8` | `[7, H, W]` | 예 | family별 binary segmentation mask |
| `pattern_intensity` | `float32` | `[7, H, W]` | 예 | family별 결함 강도, 0-1 |

모든 2D array는 같은 `H, W`를 가져야 합니다.

## 4. pattern_masks channel 순서

`pattern_masks`는 `wafermap.data.schema.PATTERN_CLASSES` 순서를 따릅니다.

```text
0 scratch
1 ring
2 edge
3 local
4 random
5 shot_grid
6 stby_pattern
```

하지만 현재 U-Net target으로 쓰는 순서는 아래 여섯 개입니다.

```text
0 local
1 scratch
2 ring
3 edge
4 shot_grid
5 random
```

`stby_pattern`은 현재 primary defect target이 아닙니다. STBY는
`stby_mask`와 `valid_test_mask`로 context 처리합니다.

## 5. U-Net input tensor

`wafermap.training.segmentation.sample_to_input_tensor()`는 sample 하나를
아래 tensor로 바꿉니다.

```text
X.shape = [12, output_size, output_size]
```

input channel 순서:

```text
0  severity_mean
1  severity_max
2  fail_density
3  wafer_mask
4  valid_test_mask
5  stby_mask
6  x_norm
7  y_norm
8  radial_norm
9  angle_sin
10 angle_cos
11 edge_distance_norm
```

resize 정책:

- `severity_mean`: normalized severity의 mean pooling
- `severity_max`: normalized severity의 max pooling
- `fail_density`: `severity > 0`의 mean pooling
- `wafer_mask`, `valid_test_mask`, `stby_mask`: max pooling
- 좌표 channel: mean pooling

왜 이렇게 하냐면, 평균만 쓰면 작은 결함이 사라지고 max만 쓰면 한 pixel이
너무 과장될 수 있기 때문입니다.

## 6. U-Net target tensor

`wafermap.training.segmentation.sample_to_target_tensor()`는 `pattern_masks`를
아래 tensor로 바꿉니다.

```text
Y.shape = [6, output_size, output_size]
```

target channel 순서:

```text
0 local
1 scratch
2 ring
3 edge
4 shot_grid
5 random
```

resize 전에 모든 target은 아래처럼 잘립니다.

```text
target = pattern_mask & wafer_mask & valid_test_mask
```

target mask는 max pooling으로 resize합니다. 그래야 작은 blob이나 얇은
scratch가 낮은 해상도에서 사라지지 않습니다.

## 7. metadata.json 예시

`metadata.json`은 이 sample이 어떻게 만들어졌는지 설명합니다. loader가
반드시 필요로 하는 값은 `sample_id`이고, 나머지는 review와 debugging을
위한 정보입니다.

```json
{
  "schema_version": "synthetic_wafer_training_sample/v1",
  "sample_id": "synth_000001",
  "source": "hybrid_pattern_asset_composer",
  "base_sample_id": "base_wafer_0003",
  "image_shape": {
    "height": 1024,
    "width": 1024
  },
  "pattern_classes": [
    "scratch",
    "ring",
    "edge",
    "local",
    "random",
    "shot_grid",
    "stby_pattern"
  ],
  "target_channels": [
    "local",
    "scratch",
    "ring",
    "edge",
    "shot_grid",
    "random"
  ],
  "multi_label": true,
  "stby_target_excluded": true,
  "composition_rule": "max",
  "placed_assets": [
    {
      "family": "local",
      "asset_id": "WAFER_0001_local_0001",
      "source_sample_id": "WAFER_0001",
      "placed_xy": [140, 720],
      "bbox_xywh": [140, 720, 80, 70],
      "mask_pixel_count": 530,
      "label_type": "manual_mask"
    }
  ],
  "parametric_instances": [
    {
      "family": "shot_grid",
      "instance_id": "synth_000001_shot_grid_0001",
      "label_type": "parametric_mask",
      "parameters": {
        "shot_rows": 3,
        "shot_cols": 3,
        "shot_row_offset": 0,
        "shot_col_offset": 0,
        "affected_slot": [2, 0],
        "anchor_region": "lower_left",
        "intra_die_region": {
          "x_min": 0.0,
          "x_max": 0.35,
          "y_min": 0.65,
          "y_max": 1.0
        },
        "severity_threshold": 1
      },
      "rasterized_to_pattern_mask": true
    }
  ]
}
```

중요한 점은 `parametric_instances` 자체를 학습하지 않는다는 것입니다.
그 규칙으로 생성된 pixel이 `pattern_masks`에 들어가 있어야 학습됩니다.

## 8. manifest CSV 규격

readiness script는 sample folder를 보고 manifest를 만듭니다.

```powershell
python scripts/build_segmentation_readiness.py `
  --data data/synthetic/<dataset_name> `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
```

필수 column:

| Column | 의미 |
|---|---|
| `sample_id` | `metadata.json`의 sample id |
| `split` | `train` 또는 `val` |
| `arrays_path` | `arrays.npz` 경로 |
| `metadata_path` | `metadata.json` 경로 |
| `input_channels` | pipe(`|`)로 연결한 input channel 이름 |
| `target_channels` | pipe(`|`)로 연결한 target channel 이름 |
| `has_<family>` | valid-test target에 해당 family가 있으면 1 |
| `<family>_mask_ratio` | `wafer_mask & valid_test_mask` 안에서 target pixel 비율 |
| `active_target_count` | 이 sample에 들어있는 positive target family 개수 |

예시:

```csv
sample_id,split,arrays_path,metadata_path,input_channels,target_channels,has_local,local_mask_ratio,has_scratch,scratch_mask_ratio,has_ring,ring_mask_ratio,has_edge,edge_mask_ratio,has_shot_grid,shot_grid_mask_ratio,has_random,random_mask_ratio,active_target_count
synth_000001,train,data/synthetic/my_dataset/synth_000001/arrays.npz,data/synthetic/my_dataset/synth_000001/metadata.json,severity_mean|severity_max|fail_density|wafer_mask|valid_test_mask|stby_mask|x_norm|y_norm|radial_norm|angle_sin|angle_cos|edge_distance_norm,local|scratch|ring|edge|shot_grid|random,1,0.0021,1,0.0042,0,0.0,0,0.0,1,0.0065,1,0.0080,4
```

## 9. 합성 규칙

asset이나 parametric label을 base wafer에 합성할 때 기본 규칙은 이렇습니다.

```text
severity[pixel] = max(base_severity[pixel], defect_grade[pixel])
pattern_masks[family_index, pixel] = 1
pattern_intensity[family_index, pixel] = max(existing_intensity, defect_intensity)
```

모든 target mask는 반드시 valid 영역으로 잘라야 합니다.

```text
mask &= wafer_mask > 0
mask &= valid_test_mask > 0
```

severity 조건이 있는 경우:

```text
mask &= severity >= severity_threshold
```

multi-family overlap은 허용합니다. 모델은 softmax가 아니라 sigmoid target을
사용합니다.

## 10. parametric label이 학습 mask가 되는 과정

parametric label은 mask를 만드는 recipe입니다.

```text
shot_grid rule
-> full-size binary mask 생성
-> pattern_masks[shot_grid]에 기록
-> manifest 생성
-> U-Net 학습
```

예시:

| Family | 사람이 입력하는 값 | 최종 rasterized mask |
|---|---|---|
| `shot_grid` | shot rows/cols, affected slot, intra-die region | 반복 die region pixel |
| `ring` | center, radius, width, angle range | annulus 또는 arc pixel |
| `edge` | radial range, angle sector | edge band/sector pixel |
| `scratch` | polyline points, width | 선 주변 tube pixel |
| `local` | center/radius, lasso, brush | blob pixel |

사람이 라벨을 어떻게 저장하는지는 [라벨 데이터 가이드](label_data_guidelines.md)를
보면 됩니다.

## 11. readiness 판단 기준

파일이 열리기만 한다고 학습 가능한 dataset은 아닙니다. 아래를 확인해야
합니다.

- 모든 target family가 train split에 충분히 있는가?
- validation split에도 해석할 만큼 positive sample이 있는가?
- target pixel이 `wafer_mask & valid_test_mask` 안에만 있는가?
- `mixed_unknown`이 target에 섞이지 않았는가?
- mask ratio가 너무 넓거나 너무 희박하지 않은가?
- gallery에서 사람이 봐도 family 정의가 맞는가?

`train_unet_segmentation.py`는 target family가 train split에 없으면 기본적으로
학습을 막습니다. `--allow-incomplete-target-coverage`는 wiring/debug용으로만
써야 합니다.

## 12. 이 규격을 담당하는 코드

| 계약 | 코드 |
|---|---|
| array dataclass와 `PATTERN_CLASSES` | `src/wafermap/data/schema.py` |
| sample folder 읽기/쓰기 | `src/wafermap/data/io.py` |
| asset 합성 | `scripts/compose_synthetic_from_assets.py` |
| procedural pattern rasterization | `src/wafermap/synth/procedural_patterns.py` |
| readiness manifest와 metrics | `scripts/build_segmentation_readiness.py` |
| U-Net input/target tensor | `src/wafermap/training/segmentation.py` |
| train coverage check | `scripts/train_unet_segmentation.py` |
