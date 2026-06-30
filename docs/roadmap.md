# 로드맵

이 문서는 WaferMap을 어떤 순서로 구현하고 검증할지 정리합니다.
현재 중심은 “대표 불량 패턴을 모아 합성 segmentation dataset을 만드는 것”입니다.

## Phase 0. 범위 고정

상태: 완료

결정:

- wafer classification이 아니라 multi-label segmentation을 목표로 합니다.
- target은 family별 binary mask입니다.
- `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`을 primary family로 둡니다.
- 애매한 불량은 억지로 학습 target에 넣지 않습니다.

산출물:

- [핵심 방향](core_direction.md)
- [불량 Family 정의](pattern_taxonomy.md)

## Phase 1. Direct Segmentation Tool

상태: 현재

목표:

- 실제 wafer를 열고 사람이 mask를 만들 수 있어야 합니다.
- prediction JSON을 prefill로 불러와 수정할 수 있어야 합니다.
- 수정된 mask를 pattern asset으로 저장할 수 있어야 합니다.

예시 명령:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/product_A_manifest.json `
  --sample-id WAFER_0001 `
  --assets-root data/pattern_assets
```

완료 기준:

- `local` blob을 수동 mask로 저장 가능
- `scratch`, `ring`, `edge`, `shot_grid`는 parametric workflow 방향이 문서화됨
- asset report에서 저장 결과 확인 가능

## Phase 2. Pattern Asset Library

상태: 현재

목표:

- family별 좋은 대표 예시를 모읍니다.
- asset마다 source wafer, family, label_type, bbox, quality가 남아야 합니다.

처음 목표량:

| Family | 1차 목표 |
|---|---|
| `local` | 50개 이상 |
| `scratch` | 30개 이상 |
| `ring` | 20개 이상 또는 rule 중심 |
| `edge` | 20개 이상 또는 rule 중심 |
| `shot_grid` | 10개 이상 또는 rule 중심 |
| `random` | baseline sample 충분히 |

## Phase 3. Hybrid Synthetic Data

상태: 현재

목표:

- 실제 asset과 procedural fallback을 섞어 합성 sample을 만듭니다.
- 각 sample은 `arrays.npz`와 `metadata.json`을 가져야 합니다.
- `pattern_masks`가 family별 target을 포함해야 합니다.

예시 명령:

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20
```

## Phase 4. Readiness와 Smoke Validation

상태: 현재

목표:

- synthetic dataset이 U-Net 학습에 들어갈 수 있는지 검사합니다.
- family coverage, mask ratio, split을 확인합니다.

예시 명령:

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

완료 기준:

- `asset_segmentation_manifest.csv` 생성
- report 생성
- pipeline test 통과

## Phase 5. Small U-Net Training

상태: 다음

목표:

- coordinate-aware small U-Net을 학습합니다.
- 모델 prediction을 correction tool seed로 export합니다.

예시 명령:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out-model outputs/models/asset_unet_segmentation.pt
```

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json
```

성공 기준:

- 모델이 shape 오류 없이 학습됨
- prediction이 전부 blank가 아님
- segmentation tool에서 prediction을 수정 가능

## Phase 6. Active Learning

상태: 다음

목표:

```text
모델 예측
-> 사람이 수정
-> 수정 mask를 새 asset으로 저장
-> 합성 데이터 업데이트
-> 재학습
```

이 단계부터 실제 wafer와 synthetic wafer 사이의 차이를 줄입니다.

## Phase 7. Retrieval과 유사 wafer 검색

상태: 보조

retrieval/embedding 계열 실험은 남겨두지만 현재 주력 workflow는 아닙니다.
나중에 아래 용도로 사용할 수 있습니다.

- 비슷한 wafer 찾기
- 리뷰 우선순위 정하기
- 새 defect 후보 추천

## 현재 우선순위

1. 실제 wafer에서 명확한 family별 예시를 모읍니다.
2. `shot_grid`, `ring`, `edge` parametric label을 더 실용적으로 만듭니다.
3. 합성 sample의 `pattern_masks`를 계속 검증합니다.
4. U-Net prediction correction loop를 실제 wafer에서 반복합니다.
