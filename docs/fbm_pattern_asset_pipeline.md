# Pattern Asset과 합성 데이터 파이프라인

이 문서는 실제 wafer에서 뽑은 불량 pattern asset을 어떻게 합성 데이터셋과
U-Net 학습으로 연결하는지 설명합니다.

## 1. 핵심 계약

파이프라인의 계약은 단순합니다.

```text
pattern asset 또는 parametric rule
-> base wafer 위에 배치
-> severity map 업데이트
-> family별 full-size mask 생성
-> arrays.npz에 저장
```

중요한 점:

- 사람이 만든 asset이 우선입니다. 이를 `human asset primary` 원칙으로 둡니다.
- 부족한 family는 `procedural fallback`으로 보강합니다.
- 합성 sample은 반드시 정답 `pattern_masks`를 가져야 합니다.

## 2. Pattern asset 형식

asset 하나는 보통 아래 구조를 가집니다.

```text
data/pattern_assets/local/local_000001/
  image.png
  mask.png
  metadata.json
```

metadata 예시:

```json
{
  "asset_id": "local_000001",
  "family": "local",
  "source_sample_id": "WAFER_0001",
  "label_type": "manual_mask",
  "bbox_xywh": [120, 720, 80, 70],
  "quality": "usable"
}
```

`mask.png`는 asset crop 내부의 binary mask입니다.
합성할 때 이 mask를 base wafer 좌표로 옮겨 `pattern_masks`에 기록합니다.

## 3. Parametric asset 형식

손으로 따기 어려운 패턴은 rule로 저장합니다.

```json
{
  "family": "edge",
  "label_type": "parametric_mask",
  "params": {
    "theta_range_deg": [300, 40],
    "radial_range": [0.88, 1.0],
    "intensity": 0.75
  }
}
```

이 rule은 합성 시점에 full-size mask로 rasterize됩니다.

## 4. Family별 권장 합성 방식

| Family | 권장 source | 합성 예시 |
|---|---|---|
| `local` | 실제 asset | random 위치에 blob crop 배치 |
| `scratch` | 실제 asset + procedural line | polyline을 랜덤 위치/각도로 배치 |
| `ring` | parametric rule | radius와 width를 바꿔 ring 생성 |
| `edge` | parametric rule | edge sector angle을 바꿔 생성 |
| `shot_grid` | parametric rule | shot layout과 affected slot 반복 |
| `random` | procedural fallback | sparse fail noise 생성 |

## 5. 합성 명령

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

처음에는 작은 count로 preview를 확인하고, 문제가 없을 때 count를 늘립니다.

## 6. 배치 정책

asset을 배치할 때 확인할 것:

- wafer 밖으로 너무 많이 나가지 않게 합니다.
- `valid_test_mask` 밖에 target을 만들지 않습니다.
- 여러 family가 겹칠 수 있지만 metadata에 composition 기록을 남깁니다.
- 너무 과도한 defect density를 만들지 않습니다.

예시:

```text
base wafer는 정상/약한 random fail만 있음
-> local asset 1개를 왼쪽 아래에 배치
-> scratch procedural line 1개를 중앙에 생성
-> edge sector 1개를 오른쪽 edge에 생성
```

결과:

```text
pattern_masks[local] = local asset 위치
pattern_masks[scratch] = line 위치
pattern_masks[edge] = edge sector 위치
```

## 7. readiness manifest

합성 후 readiness를 돌립니다.

```powershell
python scripts/build_segmentation_readiness.py `
  --dataset-dir data/synthetic/asset_composed `
  --out-manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-report outputs/reports/segmentation_readiness.html
```

readiness가 확인하는 것:

- `arrays.npz` 필수 key 존재
- shape 일치
- family별 mask ratio
- train/val split 가능 여부
- target이 전부 0인 sample 과다 여부

## 8. 모델 보조 correction

학습된 U-Net은 바로 최종 판정기가 아닙니다.
실제 목적은 correction 시간을 줄이는 것입니다.

```text
U-Net prediction
-> segmentation tool에서 prefill
-> 사람이 틀린 부분 수정
-> 수정 mask를 새 asset으로 저장
-> 다음 합성/학습에 반영
```

prediction export 예시:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

## 9. 검증 명령

전체 pipeline smoke:

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

테스트:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py -q
python -m pytest tests/test_segmentation_training.py -q
```

## 10. 실패 예시

| 실패 | 원인 | 해결 |
|---|---|---|
| 합성 sample은 있는데 target이 없음 | mask 기록 누락 | `pattern_masks` 확인 |
| shot_grid가 부자연스러움 | shot layout metadata 부족 | layout/rule 보강 |
| U-Net loss가 줄지 않음 | family coverage 부족 | asset/rule 수집 확대 |
| 실제 wafer prediction이 약함 | synthetic-real gap | correction loop 반복 |
