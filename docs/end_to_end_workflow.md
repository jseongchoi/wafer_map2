# 전체 실행 흐름

이 문서는 WaferMap을 처음부터 끝까지 어떻게 사용하는지 설명합니다.
한 줄로 요약하면 아래와 같습니다.

```text
실제 wafer 준비
-> 대표 불량 mask 저장
-> pattern asset 검수
-> 합성 wafer 생성
-> 학습 manifest 생성
-> U-Net 학습
-> 예측 export
-> 사람이 수정
```

## 1. 입력 데이터 준비

입력은 두 종류가 가능합니다.

| 입력 | 언제 사용 | 예시 |
|---|---|---|
| raw PNG 폴더 | 실제 제품 wafer image가 있을 때 | `data/raw/product_A/*.png` |
| semantic `.npz` | 이미 array와 metadata가 준비되어 있을 때 | `data/synthetic/.../arrays.npz` |

raw PNG를 쓴다면 먼저 manifest를 만듭니다.

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/product_A_manifest.json
```

manifest에는 sample id, image path, shape, geometry 정보가 들어갑니다.

## 2. 실제 wafer에서 pattern asset 추출

segmentation tool을 열고 명확한 불량만 mask로 저장합니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

작업 규칙:

- 경계가 명확한 `local` blob은 brush/lasso로 mask를 저장합니다.
- 긴 `scratch`는 가능하면 polyline + width 같은 parametric 표현을 사용합니다.
- `ring`, `edge`, `shot_grid`는 손으로 다 칠하지 말고 rule 기반 mask를 우선합니다.
- 애매하면 `mixed_unknown`으로 남기고 학습 target에서는 제외합니다.

## 3. Defect family 선택

현재 학습 target은 아래 여섯 family입니다.

```text
local
scratch
ring
edge
shot_grid
random
```

예시:

```text
왼쪽 아래에 작은 덩어리      -> local
중앙을 가로지르는 긴 선       -> scratch
동심원처럼 둥근 band          -> ring
wafer 오른쪽 edge만 두꺼움    -> edge
shot마다 왼쪽 아래 die fail   -> shot_grid
구조 없이 드문드문 fail       -> random
```

## 4. asset 검수

저장된 asset이 학습에 쓸 만한지 report로 확인합니다.

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

검수 기준:

- mask가 wafer 밖을 많이 포함하지 않는가?
- family가 맞는가?
- 너무 애매한 패턴을 억지로 넣지 않았는가?
- bbox는 crop 힌트로만 쓰이고 실제 target mask가 존재하는가?

## 5. 합성 wafer 생성

대표 pattern asset을 base wafer 위에 합성합니다.

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

합성 결과 sample은 아래처럼 생깁니다.

```text
data/synthetic/asset_composed/
  asset_composed_000001/
    arrays.npz
    metadata.json
    preview.png
```

`arrays.npz` 안의 핵심은 `pattern_masks`입니다.

```text
pattern_masks[family_index, y, x] = 1 또는 0
```

## 6. 학습 준비 상태 확인

합성 sample이 학습 가능한지 확인하고 manifest를 만듭니다.

```powershell
python scripts/build_segmentation_readiness.py `
  --dataset-dir data/synthetic/asset_composed `
  --out-manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-report outputs/reports/segmentation_readiness.html
```

확인할 것:

- `arrays.npz`에 필수 key가 있는가?
- target family가 하나 이상 있는 sample이 충분한가?
- family별 mask ratio가 0만 나오지 않는가?
- train/val split이 만들어졌는가?

## 7. U-Net 학습

readiness manifest를 사용해 작은 U-Net을 학습합니다.

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-model outputs/models/asset_unet_segmentation.pt
```

이 모델의 목적은 완벽한 자동 판정이 아닙니다.
실제 wafer에 대해 사람이 고칠 수 있는 첫 prediction을 만드는 것입니다.

## 8. 예측 export와 수정 loop

학습된 모델로 validation 또는 실제 wafer prediction을 export합니다.

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

이 JSON의 schema marker는 `fbm_prediction_masks/v1`입니다.
segmentation tool은 이 형식을 correction seed로 읽습니다.

그 다음 prediction을 segmentation tool에 seed로 넣습니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0002 `
  --assets-root data/pattern_assets `
  --prediction-json outputs/predictions/fbm_prediction_masks.json
```

사람은 예측 mask를 수정하고, 수정된 좋은 예시를 다시 asset으로 저장합니다.
이것이 active learning loop의 시작입니다.

## 9. 한 번에 점검하는 명령

asset 합성, readiness, report 생성을 한 번에 확인하려면 아래 command를 씁니다.

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

## 10. 테스트 정책

빠른 검증:

```powershell
python -m pytest tests/test_documentation_quality.py -q
python -m pytest tests/test_pattern_asset_pipeline.py -q
python -m pytest tests/test_segmentation_training.py -q
```

느린 end-to-end 검증까지 포함:

```powershell
python -m pytest -q --run-slow --basetemp .pytest_tmp_full
```

## 11. 자주 생기는 실수

| 실수 | 왜 문제인가 | 대안 |
|---|---|---|
| bbox만 저장 | U-Net target이 없음 | full-size binary mask 저장 |
| 애매한 패턴을 억지 family로 저장 | 모델 family 경계가 흐려짐 | `mixed_unknown`으로 보관 |
| shot_grid를 손으로 전부 칠함 | 시간이 너무 많이 듦 | shot layout + affected slot 사용 |
| 합성 데이터만 만들고 correction loop를 안 돌림 | 실제 wafer gap이 줄지 않음 | prediction을 사람이 고쳐 asset으로 재저장 |
