# 라벨 없는 실제 Wafer 처리 절차

이 문서는 label이 없는 실제 wafer batch를 받았을 때 어떻게 처리할지 설명합니다.
핵심은 “모델 학습용 정답이 없다”는 점을 인정하고, 먼저 triage와 asset 후보를
만드는 것입니다.

## 1. 목적

라벨 없는 실제 wafer는 바로 supervised training에 넣을 수 없습니다.
대신 아래 용도로 씁니다.

- 대표 불량 후보 찾기
- family별 asset 후보 수집
- 모델 prediction correction 대상 만들기
- synthetic-real gap 확인

## 2. 입력 준비 A: 제품별 raw PNG 폴더

```text
data/raw/product_A/
  WAFER_0001.png
  WAFER_0002.png
```

manifest 생성:

```powershell
python scripts/analyze_png_raw_folders.py `
  --input-root data/raw/product_A `
  --out-manifest outputs/manifests/product_A_manifest.json `
  --out-report-dir outputs/reports/product_A_raw
```

## 3. 입력 준비 B: Semantic `.npz`

이미 wafer array가 준비되어 있다면 sample folder를 사용할 수 있습니다.

```text
data/real_unlabeled/product_A/
  WAFER_0001/
    arrays.npz
    metadata.json
```

이 경우에도 sample id, path, shape가 manifest로 정리되어야 합니다.

## 4. Manifest 예시

```json
{
  "samples": [
    {
      "sample_id": "WAFER_0001",
      "image_path": "data/raw/product_A/WAFER_0001.png",
      "product_id": "product_A",
      "split": "unlabeled"
    }
  ]
}
```

## 5. Feature 추출

retrieval이나 sanity check가 필요하면 feature를 추출합니다.

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --out outputs/features/product_A_real_features.parquet
```

이 단계는 현재 주력 segmentation workflow의 보조입니다.
좋은 asset 후보를 찾는 데 사용할 수 있습니다.

## 6. Segmentation tool 연결

라벨 없는 wafer에서 실제로 중요한 작업은 대표 불량을 골라 mask로 저장하는 것입니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

작업자는 명확한 패턴만 저장합니다.

```text
확실한 local blob       -> asset 저장
확실한 scratch          -> asset 또는 parametric 저장
shot 반복 위치 불량     -> shot_grid rule 후보
애매한 diffuse          -> mixed_unknown/review-only
```

## 7. 모델 prediction을 이용하는 경우

이미 학습된 모델이 있으면 prediction을 seed로 열 수 있습니다.

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json
```

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets `
  --prediction-json outputs/predictions/fbm_prediction_masks.json
```

## 8. 산출물

라벨 없는 wafer batch에서 기대하는 산출물:

```text
outputs/manifests/product_A_manifest.json
outputs/reports/product_A_raw/
data/pattern_assets/<family>/<asset_id>/
outputs/reviews/product_A_review_template.csv
```

## 9. Sanity / Drift 해석

실제 wafer와 합성 wafer가 다르면 prediction 품질이 낮을 수 있습니다.
이때는 모델이 틀렸다고만 보지 말고 아래를 봅니다.

- 실제 wafer에 없는 family를 합성에서 과하게 만들었는가?
- 실제 wafer의 intensity 분포가 합성과 다른가?
- shot layout이 합성 rule과 다른가?
- edge/ring 위치 기준이 제품마다 다른가?

## 10. 다음 의사결정

| 상황 | 다음 행동 |
|---|---|
| 명확한 local이 많이 보임 | local asset을 우선 저장 |
| 반복 shot 문제가 보임 | shot layout metadata 확보 |
| 모델 prediction이 blank | synthetic coverage 부족 확인 |
| 모델 false positive가 많음 | threshold 조정, negative/random context 보강 |
| family 정의가 애매함 | 전문가 review protocol로 넘김 |
