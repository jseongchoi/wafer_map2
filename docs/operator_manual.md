# 작업자 매뉴얼

이 문서는 실제로 WaferMap 파이프라인을 돌리는 사람이 보는 짧은 실행 안내서입니다.
자세한 설계 설명보다 “무엇을 실행하고 무엇을 확인해야 하는가”에 집중합니다.

## 1. 목표

작업자의 목표는 좋은 training sample을 만드는 것입니다.

```text
좋은 실제 wafer 예시를 찾는다
-> 명확한 불량만 mask로 저장한다
-> 합성 데이터셋을 만든다
-> readiness report를 본다
-> U-Net prediction을 다시 수정한다
```

## 2. 역할

| 역할 | 하는 일 |
|---|---|
| 작업자 | wafer를 열고 불량 mask/asset을 저장 |
| 리뷰어 | family가 맞는지, mask가 학습에 적합한지 확인 |
| 개발자 | command, schema, training pipeline 유지 |

작업자와 리뷰어가 다르지 않아도 됩니다. 다만 저장할 때와 검수할 때의 관점을
분리하는 것이 좋습니다.

## 3. 사전 준비

필요한 입력:

- raw wafer PNG 또는 이미 준비된 sample manifest
- wafer geometry 정보가 있으면 더 좋음
- 저장할 asset 폴더: `data/pattern_assets`
- 출력 폴더: `outputs/`

폴더 예시:

```text
data/raw/product_A/
  WAFER_0001.png
  WAFER_0002.png

data/pattern_assets/
outputs/manifests/
outputs/reports/
```

## 4. manifest 준비

raw PNG 폴더에서 시작할 때:

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/product_A_manifest.json
```

실행 후 확인:

- manifest 파일이 생성됐는가?
- sample id가 사람이 구분 가능한가?
- image shape가 예상과 맞는가?

## 5. segmentation tool 열기

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

tool에서 할 일:

1. wafer 전체를 먼저 훑습니다.
2. 확실한 불량만 family를 고릅니다.
3. `local`처럼 경계가 명확한 것은 mask로 칠합니다.
4. `ring`, `edge`, `shot_grid`는 가능하면 규칙 기반으로 저장합니다.
5. 애매한 것은 `mixed_unknown` 또는 review-only로 남깁니다.

## 6. 라벨링 규칙

| 상황 | 권장 행동 |
|---|---|
| 작은 blob이 선명함 | `local`로 mask 저장 |
| 긴 선형 결함 | `scratch`로 저장하되 polyline + width 고려 |
| 둥근 band | `ring` parametric label 사용 |
| wafer edge 근처만 이상함 | `edge` sector rule 사용 |
| shot마다 같은 상대 위치 문제 | `shot_grid` rule 사용 |
| family가 애매함 | 학습 target에 넣지 말고 review-only |

가장 중요한 원칙:

```text
bbox는 위치 힌트다.
U-Net이 배우는 정답은 mask다.
```

## 7. asset 저장 후 report 확인

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

report에서 볼 것:

- family별 asset 수
- mask가 비어 있지 않은지
- 너무 큰 mask가 아닌지
- 같은 wafer에서 중복 저장된 asset이 많은지
- preview가 사람이 보기에 납득되는지

## 8. 합성 데이터 만들기

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

처음에는 `--count 20` 정도로 작게 돌려 preview와 mask가 맞는지 확인합니다.

## 9. readiness 확인

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

생성되는 핵심 파일:

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
outputs/reports/pattern_asset_project_report.html
```

## 10. U-Net 학습과 prediction export

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-model outputs/models/asset_unet_segmentation.pt
```

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

## 11. Troubleshooting

| 증상 | 확인할 것 | 조치 |
|---|---|---|
| manifest가 비어 있음 | 입력 폴더 경로 | `--input-root` 재확인 |
| tool에서 wafer가 안 보임 | image path | manifest의 path가 실제 존재하는지 확인 |
| asset report가 비어 있음 | save 여부 | tool에서 asset 저장 버튼/출력 폴더 확인 |
| readiness에서 target 없음 | `pattern_masks` | 합성 sample의 `arrays.npz` 확인 |
| U-Net 학습 실패 | manifest column | `asset_segmentation_manifest.csv` schema 확인 |
| prediction이 엉뚱함 | threshold, family coverage | threshold 조정, asset 추가 수집 |

## 12. Release Checklist

작업 결과를 공유하기 전 아래를 확인합니다.

- [ ] `data/pattern_assets`에 family별 asset이 저장됨
- [ ] pattern asset report가 열림
- [ ] 합성 sample preview가 이상하지 않음
- [ ] `asset_segmentation_manifest.csv`가 생성됨
- [ ] 문서 링크가 깨지지 않음
- [ ] `python -m pytest tests/test_documentation_quality.py -q` 통과
- [ ] 관련 pipeline 테스트 통과
