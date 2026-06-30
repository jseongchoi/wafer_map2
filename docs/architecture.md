# 설계 구조

이 문서는 새 코드를 어디에 넣고, 각 폴더가 무엇을 책임지는지 설명합니다.
목표는 단순합니다. “문서에 적힌 데이터 흐름”과 “실제 코드 위치”가 서로
어긋나지 않게 만드는 것입니다.

## 1. 제품 경계

WaferMap의 중심 기능은 아래 네 가지입니다.

1. 실제 wafer sample을 읽고 manifest로 정리합니다.
2. 사람이 segmentation tool에서 대표 불량 mask를 저장합니다.
3. pattern asset과 parametric rule을 합성해 학습 sample을 만듭니다.
4. small U-Net을 학습하고 prediction을 다시 사람이 고칠 수 있게 내보냅니다.

이 경계를 벗어난 실험성 retrieval, historical benchmark, report는 보조 기능입니다.

## 2. 최상위 폴더

| 폴더 | 역할 | 예시 |
|---|---|---|
| `scripts/` | 사용자가 직접 실행하는 command | `run_segmentation_tool.py` |
| `src/wafermap/` | 재사용 가능한 실제 구현 | `assets/`, `synth/`, `training/` |
| `docs/` | 작업자/개발자 문서 | `index.html`, `label_data_guidelines.md` |
| `configs/` | manifest, geometry, eval 설정 | `configs/eval/*.json` |
| `tests/` | 문서/데이터/학습 계약 검증 | `tests/test_pattern_asset_pipeline.py` |
| `data/` | 로컬 입력과 생성 데이터 | git에 넣지 않는 산출물 |
| `outputs/` | report, model, prediction output | git에 넣지 않는 산출물 |

## 3. 패키지 경계

| 패키지 | 책임 |
|---|---|
| `src/wafermap/assets/` | pattern asset 저장, metadata, library report |
| `src/wafermap/synth/` | synthetic wafer 생성, procedural pattern |
| `src/wafermap/training/` | U-Net dataset, tensor 변환, 학습 loop |
| `src/wafermap/real/` | 실제 raw PNG ingestion과 manifest 생성 |
| `src/wafermap/reporting/` | HTML/report 생성 |
| `src/wafermap/data/` | 공통 schema와 family 정의 |

## 4. 기능별 코드 지도

### 실제 wafer 입력

사용자가 제품별 raw PNG 폴더를 가지고 있으면 먼저 manifest로 만듭니다.

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/product_A_manifest.json
```

관련 코드:

- `scripts/analyze_png_raw_folders.py`
- `src/wafermap/real/`

### 직접 mask 생성

작업자가 실제 wafer를 보고 명확한 defect를 mask로 저장합니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

관련 코드:

- `scripts/run_segmentation_tool.py`
- `scripts/run_pattern_asset_editor.py`
- `src/wafermap/assets/`

### 합성 데이터 생성

asset과 procedural pattern을 base wafer 위에 합성합니다.

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20
```

관련 코드:

- `scripts/compose_synthetic_from_assets.py`
- `src/wafermap/synth/`

### 학습 준비와 U-Net

합성 sample을 검사하고 manifest를 만든 뒤 학습합니다.

```powershell
python scripts/build_segmentation_readiness.py `
  --dataset-dir data/synthetic/asset_composed `
  --out-manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv

python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
```

관련 코드:

- `scripts/build_segmentation_readiness.py`
- `scripts/train_unet_segmentation.py`
- `src/wafermap/training/segmentation.py`

## 5. 새 코드를 넣는 위치

| 새 요구사항 | 넣을 위치 | 이유 |
|---|---|---|
| 새 defect family | `src/wafermap/data/schema.py`와 관련 test | family 순서가 학습 target에 직접 영향 |
| 새 asset metadata | `src/wafermap/assets/` | 저장/검수/report가 한곳에서 관리됨 |
| 새 procedural defect | `src/wafermap/synth/procedural_patterns.py` | 합성 mask 생성 책임 |
| 새 training channel | `src/wafermap/training/segmentation.py` | input/target tensor 계약 |
| 새 사용자 command | `scripts/`와 `scripts/README.md` | 사용자가 직접 실행 |
| 새 운영 문서 | `docs/operator_manual.md` 또는 관련 문서 | 작업자가 찾기 쉬움 |

## 6. 기술 부채

현재 일부 파일명에는 과거 호환성을 위한 이름이 남아 있습니다.
예를 들어 `run_pattern_asset_editor.py`는 기존 사용자를 위해 유지하지만,
실제 방향은 in-repo segmentation tool 중심입니다.

또한 historical retrieval/benchmark 문서는 남아 있지만 현재 1순위 제품 흐름은
segmentation dataset factory입니다.

## 7. 리팩터링 규칙

리팩터링은 아래 기준을 만족할 때만 합니다.

- 문서에 적힌 command 흐름이 더 단순해진다.
- 테스트가 있는 데이터 계약을 더 명확히 만든다.
- pattern asset, synthetic dataset, U-Net training 사이의 중복을 줄인다.
- 사용자-facing command 이름이 깨지지 않는다.

단순히 코드가 보기 싫다는 이유만으로 넓은 리팩터링을 하지 않습니다.
