# FBM 데이터 흐름 운영 가이드

이 문서는 실제 FBM 파일을 어디에 두고, 누끼 딴 defect 이미지는 어디에 저장되며, 합성된 multi-defect label 데이터가 어디에 만들어지는지 정리합니다.

핵심은 아래 한 줄입니다.

```text
real FBM -> manifest -> pattern asset -> hybrid synthetic sample -> segmentation manifest -> U-Net
```

## 1. 전체 폴더 계약

| 단계 | 폴더/파일 | 역할 | Git 포함 여부 |
|---|---|---|---|
| 실제 FBM 입력 | `data/raw/<product>/*.png` 또는 인트라넷 경로 | 원본 raw PNG 위치 | `.gitignore`로 제외 |
| 개발용 FBM 입력 | `data/raw/<product>/*.png` | 로컬 smoke test용 원본 PNG 위치 | `.gitignore`로 제외 |
| manifest | `outputs/manifests/<run_name>_manifest.json` | 실제 PNG 경로와 geometry가 들어 있는 실행 manifest | `.gitignore`로 제외 |
| 공유 가능 실제 분석 결과 | `outputs/reports/<run_name>/` | sanity, feature, neighbor, review report | `.gitignore`로 제외 |
| 사람이 누끼 딴 defect asset | `data/pattern_assets/<family>/<asset_id>/` | 합성 데이터에 붙일 defect 조각 | `.gitignore`로 제외 |
| 합성 학습 sample | `data/synthetic/asset_composed/<sample_id>/` | 라벨 있는 synthetic wafer sample | `.gitignore`로 제외 |
| 학습 manifest/report | `outputs/pattern_asset_pipeline/` | train/val manifest, gallery, smoke, U-Net report | `.gitignore`로 제외 |
| 모델 파일 | `outputs/models/*.pt` | 학습된 PyTorch 모델 | `.gitignore`로 제외 |

중요한 원칙은 원본 wafer와 사람이 만든 asset, 합성 sample, 모델 파일 모두 재생성 가능하거나 민감할 수 있으므로 Git에 올리지 않는다는 것입니다. Git에는 코드, 문서, `.gitkeep`만 남깁니다.

## 2. 실제 FBM은 어디에 두는가

기본 개발 위치는 workspace 안의 `data/raw`입니다. 인트라넷 운영에서는 원하는 로컬/네트워크 경로를 그대로 써도 됩니다.

```text
data/raw/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_101.png
```

인트라넷 운영 폴더 예시는 아래처럼 둘 수 있습니다.

```text
Z:/fbm/raw_png/
  product_a/
    wafer_001.png
    wafer_002.png
```

`data/raw/**`는 `.gitignore` 대상입니다. 따라서 로컬에 넣어도 Git에는 올라가지 않습니다.

PNG 조건은 다음과 같습니다.

| 항목 | 기준 |
|---|---|
| 파일 형식 | 8-bit grayscale PNG |
| 값 범위 | `0, 31, 151, 175, 191, 207, 223, 255` |
| 실제 입력 의미 | grade 0~7 wafer map |
| stby 처리 | chip block 전체가 `255`이면 stby fail chip |
| grade 7 처리 | chip 일부 pixel만 `255`이면 grade 7 |

## 3. 실제 FBM을 에디터가 읽게 만드는 manifest

Pattern Asset Editor는 원본 폴더를 직접 받지 않고 manifest를 받습니다. 먼저 raw PNG 폴더를 manifest로 바꿉니다.

실행:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

생성 위치:

```text
outputs/manifests/real_png_batch_manifest.json
outputs/reports/real_png_batch/batch_metadata.json
outputs/reports/real_png_batch/features.csv
outputs/reports/real_png_batch/sanity.json
outputs/reports/real_png_batch/report.html
```

PNG를 빠르게 manifest만 만들 때:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --manifest-only `
  --out-dir outputs/reports/local_raw_batch
```

생성 위치:

```text
outputs/manifests/local_raw_batch_manifest.json
```

## 4. 누끼 딴 defect 이미지는 어디에 저장되는가

에디터 실행:

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <보고 싶은 sample_id> `
  --assets-root data/pattern_assets
```

브라우저에서 family를 고르고 mask를 칠한 뒤 저장하면 아래 구조로 저장됩니다.

```text
data/pattern_assets/
  local/
    <asset_id>/
      grade.png
      mask.png
      preview.png
      metadata.json
  scratch/
    <asset_id>/
      grade.png
      mask.png
      preview.png
      metadata.json
  ring/
    <asset_id>/
      grade.png
      mask.png
      preview.png
      metadata.json
```

각 파일의 의미는 다음입니다.

| 파일 | 의미 |
|---|---|
| `grade.png` | 합성 wafer에 실제로 붙일 grade 0~7 defect patch |
| `mask.png` | 딥러닝 정답으로 쓸 binary defect mask |
| `preview.png` | 사람이 검수하기 쉬운 시각화 |
| `metadata.json` | family, 원본 sample, bbox, 저장 방식, source 위치 |

사람이 우선 누끼 따야 하는 family는 `local`, `scratch`, `ring`입니다. `random`은 사람이 누끼 따는 대상이 아니라 코드가 sparse fail baseline으로 만듭니다. `edge`, `shot_grid`도 기본은 코드 생성이며, 특이한 실제 모양이 있을 때만 optional asset으로 저장합니다.

## 5. 누끼 asset 검수 위치

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

생성 위치:

```text
outputs/reports/pattern_asset_library_report.html
```

검수할 내용은 아래입니다.

| 질문 | 문제일 때 피드백 |
|---|---|
| defect만 잘 칠해졌는가? | "mask가 배경까지 먹었다" |
| family가 맞는가? | "이건 local이 아니라 scratch다" |
| ring 하나가 하나의 asset으로 저장됐는가? | "ring이 여러 개로 쪼개졌다" |
| scratch가 너무 두껍거나 끊겼는가? | "scratch line assist가 방향을 잘못 잡았다" |

## 6. 합성된 multi-defect label 데이터는 어디에 생기는가

단일 명령으로 전체 파이프라인을 실행합니다.

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --count 200 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

생성되는 학습 sample:

```text
data/synthetic/asset_composed/
  asset_composed_000000/
    arrays.npz
    metadata.json
  asset_composed_000001/
    arrays.npz
    metadata.json
```

주의할 점은 합성된 label은 일반 PNG 한 장으로 저장되는 것이 아니라 `arrays.npz` 안에 tensor로 저장된다는 것입니다.

`arrays.npz` 내부 핵심 배열:

| 배열 | 의미 |
|---|---|
| `severity` | 최종 합성 wafer grade 0~7 map |
| `wafer_mask` | wafer 내부 영역 |
| `valid_test_mask` | 실제 test 가능한 chip/pixel 영역 |
| `stby_mask` | stby fail chip 영역 |
| `pattern_masks` | family별 multi-label 정답 mask |
| `pattern_intensity` | family별 defect intensity |
| `chip_index` | chip 좌표 index |

`metadata.json` 내부 핵심 정보:

| 필드 | 의미 |
|---|---|
| `placed_assets` | 어떤 human asset을 어디에 붙였는지 |
| `procedural_patterns` | 코드가 만든 scratch/edge/shot_grid/random 정보 |
| `composition_rule` | 현재는 `max` |
| `multi_label` | 한 wafer 또는 한 pixel에 여러 family가 겹칠 수 있음 |

즉 모델 학습에서 정답은 다음처럼 정의됩니다.

```text
input  = severity + wafer/valid/stby mask + coordinate channels
target = pattern_masks[local, scratch, ring, edge, shot_grid, random]
```

## 7. 학습 manifest와 리포트는 어디에 생기는가

`run_pattern_asset_pipeline.py`는 합성 sample을 만든 뒤 학습용 manifest와 검수 리포트를 같이 만듭니다.

```text
outputs/pattern_asset_pipeline/
  asset_segmentation_manifest.csv
  asset_segmentation_readiness.html
  asset_segmentation_readiness_metrics.json
  asset_segmentation_gallery.png
  asset_segmentation_smoke.html
  asset_embedding_smoke.html
  asset_embedding_vectors.csv
  asset_unet_segmentation.html
  asset_unet_segmentation_metrics.json

outputs/reports/
  pattern_asset_project_report.html
```

가장 먼저 볼 파일:

```text
outputs/reports/pattern_asset_project_report.html
```

U-Net이 직접 먹는 파일:

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
```

## 8. U-Net 학습은 어떻게 실행하는가

PyTorch 설치:

```powershell
pip install -e .[train]
```

U-Net 학습:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

생성 위치:

```text
outputs/models/asset_unet_segmentation.pt
outputs/pattern_asset_pipeline/asset_unet_segmentation.html
outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json
```

## 9. U-Net이 적합한가

현재 단계에서는 U-Net이 적합합니다.

이유는 FBM 목표가 image classification이 아니라 pixel 단위 defect mask 예측이기 때문입니다. U-Net은 encoder가 전체 wafer 문맥을 보고, decoder가 위치별 mask를 복원하는 구조라 `local`, `scratch`, `ring`, `edge`처럼 위치와 모양이 모두 중요한 문제에 잘 맞습니다.

다만 그냥 U-Net만 쓰면 부족합니다. FBM은 절대좌표가 중요하므로 입력에 위치 채널을 넣어야 합니다. 현재 학습 tensor는 다음 입력 채널을 사용합니다.

```text
severity
wafer_mask
valid_test_mask
stby_mask
x_norm
y_norm
radial_norm
angle_sin
angle_cos
edge_distance_norm
```

처음 모델은 small U-Net으로 시작하는 것이 맞습니다. 이유는 다음입니다.

| 선택 | 판단 |
|---|---|
| small U-Net | 지금 바로 학습/검증 가능한 1차 기준선 |
| U-Net++/Attention U-Net | scratch/ring 경계가 부족할 때 확장 후보 |
| SegFormer/Mask2Former | 데이터가 충분히 쌓인 뒤 확장 후보 |
| 단순 CNN classifier | 위치별 mask와 수치화가 필요하므로 1차 목표에는 부적합 |

따라서 현재 정답은 이렇습니다.

```text
지금은 coordinate-aware small U-Net이 맞다.
성능 한계가 보이면 U-Net++ 또는 SegFormer로 확장한다.
유사 wafer 검색은 U-Net encoder embedding을 저장해서 cosine/FAISS로 붙인다.
```

## 10. 최소 운영 순서

1. 실제 FBM PNG를 `data/raw/<product>/` 또는 인트라넷의 원하는 raw PNG 폴더에 둡니다.
2. `scripts/analyze_png_raw_folders.py`로 manifest를 만듭니다.
3. `scripts/run_pattern_asset_editor.py`로 `local`, `scratch`, `ring` asset을 저장합니다.
4. `scripts/build_pattern_asset_report.py`로 asset 품질을 봅니다.
5. `scripts/run_pattern_asset_pipeline.py`로 합성 sample과 segmentation manifest를 만듭니다.
6. `outputs/reports/pattern_asset_project_report.html`을 보고 family별 부족한 asset을 확인합니다.
7. PyTorch 환경에서 `scripts/train_unet_segmentation.py`를 실행합니다.
8. 예측 mask를 다시 에디터에 불러와 사람이 고치고, 그 결과를 asset/label로 누적합니다.
