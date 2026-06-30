# 검증 프로토콜

이 문서는 WaferMap에서 무엇을 검증해야 하는지 단계별로 정리합니다.
목표는 “모델 점수 하나”가 아니라 데이터 계약, mask 품질, 합성 품질, correction loop가
끊기지 않는지 확인하는 것입니다.

## 1. 검증 철학

현재 프로젝트는 production model 성능을 바로 주장하는 단계가 아닙니다.
먼저 아래가 안정적이어야 합니다.

```text
실제 wafer를 읽을 수 있다.
사람이 mask asset을 저장할 수 있다.
합성 sample이 정답 mask를 가진다.
manifest가 학습 코드와 맞는다.
모델 prediction을 다시 사람이 수정할 수 있다.
```

## 2. 입력 검증

raw PNG ingestion 후 확인합니다.

| 항목 | 확인 방법 |
|---|---|
| 파일 존재 | manifest의 `image_path`가 실제 파일인지 |
| shape | 모든 sample의 image shape가 예상 범위인지 |
| geometry | wafer center/radius가 말이 되는지 |
| intensity | raw gray 값이 뒤집히거나 깨지지 않았는지 |

예시:

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/product_A_manifest.json
```

## 3. Pattern asset 검증

asset 하나가 학습에 쓸 만한지 확인합니다.

좋은 asset 조건:

- family가 명확함
- mask가 비어 있지 않음
- mask가 wafer 밖을 많이 포함하지 않음
- bbox와 mask가 같은 defect를 가리킴
- source sample metadata가 남아 있음

report 생성:

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

## 4. Synthetic dataset 검증

합성 sample은 preview만 보고 끝내면 안 됩니다.
반드시 `arrays.npz`의 target mask를 확인해야 합니다.

필수 key:

```text
severity
wafer_mask
valid_test_mask
stby_mask
chip_index
pattern_masks
pattern_intensity
```

실패 예:

```text
preview에는 scratch가 보이는데 pattern_masks[scratch]가 전부 0
```

이 경우 모델은 scratch를 배울 수 없습니다.

## 5. Readiness 검증

```powershell
python scripts/build_segmentation_readiness.py `
  --dataset-dir data/synthetic/asset_composed `
  --out-manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-report outputs/reports/segmentation_readiness.html
```

확인 항목:

- sample 수
- train/val split
- family별 `has_*` 값
- family별 `*_mask_ratio`
- target이 비어 있는 sample 비율

## 6. Segmentation model 검증

U-Net 학습 검증은 두 단계로 봅니다.

1. 코드/데이터 계약 검증
2. prediction이 사람이 수정할 수 있는 수준인지 검증

명령:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-model outputs/models/asset_unet_segmentation.pt
```

확인:

- 학습이 shape 오류 없이 시작되는가?
- loss가 NaN이 아닌가?
- validation prediction이 전부 0 또는 전부 1이 아닌가?
- family별로 최소한의 반응이 있는가?

## 7. Prediction correction 검증

prediction export:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

검증:

- JSON schema가 `fbm_prediction_masks/v1`인지
- sample id가 manifest와 맞는지
- family별 mask가 tool에서 열리는지
- 사람이 수정 후 asset으로 저장할 수 있는지

## 8. Human review loop

모델 성능은 숫자만 보지 말고 실제 수정 시간을 봐야 합니다.

질문:

- prediction이 완전 blank보다 나은가?
- 사람이 지워야 할 false positive가 너무 많은가?
- 놓친 defect가 특정 family에 집중되는가?
- correction 후 새 asset으로 저장할 만한 예시가 생기는가?

## 9. 테스트 명령

문서와 링크:

```powershell
python -m pytest tests/test_documentation_quality.py -q --basetemp .pytest_tmp_docs
```

pattern asset과 학습 데이터 계약:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py tests/test_segmentation_training.py -q
```

전체 빠른 테스트:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

느린 테스트 포함:

```powershell
python -m pytest -q --run-slow --basetemp .pytest_tmp_full
```

## 10. 통과 기준

1차 통과 기준:

- 문서 링크가 깨지지 않음
- 핵심 문서가 한국어 설명과 예시를 포함함
- asset report가 생성됨
- synthetic sample에 `pattern_masks`가 존재함
- readiness manifest가 생성됨
- training dataset loader가 sample을 읽음
- prediction export가 correction tool 입력으로 사용 가능함
