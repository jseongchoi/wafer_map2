# Pattern Asset And Hybrid Synthetic Data Pipeline

이 문서는 직접 만든 defect mask를 reusable pattern asset으로 저장하고, 그 asset을 real/base wafer에 합성해 segmentation 학습 데이터를 만드는 방법을 설명합니다.

목표는 wafer-level class label이 아니라 pixel-level multi-label mask입니다.

```text
local segmentation mask
-> data/pattern_assets
-> data/synthetic/asset_composed
-> asset_segmentation_manifest.csv
-> train_unet_segmentation.py
```

## 1. Pattern Asset Library

Segmentation tool이 저장하는 asset format:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

| File | Meaning |
|---|---|
| `grade.png` | source wafer crop grade map |
| `mask.png` | segmentation target으로 쓸 binary mask |
| `preview.png` | source crop RGB preview |
| `metadata.json` | source sample, bbox, family, composition metadata |

`metadata.json`은 `fbm_pattern_asset/v1` schema를 사용합니다. Synthetic composer는 이 schema와 `mask.png`를 직접 소비합니다.

## 2. Asset Families

| Family | Source policy |
|---|---|
| `local` | human asset primary |
| `scratch` | human asset primary + procedural fallback |
| `ring` | human asset primary |
| `edge` | procedural primary + optional human asset |
| `shot_grid` | procedural primary + optional human asset |
| `random` | procedural only |

`edge`, `shot_grid`, `random`은 규칙 기반 합성이 빠르고 일관됩니다. 다만 실제 edge band나 shot pattern이 특정 공정 signature를 가지면 직접 asset으로 보강합니다.

## 3. Compose Synthetic Dataset

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 200 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

합성 원칙:

```text
severity[pixel] = max(base severity, pasted asset grade)
pattern_masks[family][pixel] = 1 where defect mask is active
```

한 wafer에 여러 family가 겹칠 수 있으므로 model target은 sigmoid multi-label mask입니다.

## 4. Run Readiness Pipeline

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

주요 산출물:

| Output | Meaning |
|---|---|
| `asset_segmentation_manifest.csv` | training/validation sample manifest |
| `asset_segmentation_readiness_metrics.json` | family coverage, overlap, split metrics |
| `asset_segmentation_gallery.png` | quick visual gallery |
| `asset_segmentation_smoke.html` | input/target/loss wiring smoke test |
| `asset_embedding_smoke.html` | simple embedding diagnostic |
| `asset_unet_segmentation.html` | PyTorch dependency or training entry report |
| `pattern_asset_project_report.html` | project-level summary |

Segmentation smoke test는 성능 평가가 아니라 data wiring 검증입니다. 실제 성능 판단은 PyTorch 환경에서 `train_unet_segmentation.py`를 돌린 뒤 family별 IoU/recall로 봅니다.

## 5. Train U-Net Entry Point

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

Before training, the entrypoint checks train-split target coverage. Each target family needs at least `--min-positive-samples-per-class` positive train samples. Validation-only gaps do not block training, but they are reported because that class metric is not meaningful without validation positives. Use `--allow-incomplete-target-coverage` only for wiring/debug runs.

입력 채널:

```text
severity_mean
severity_max
fail_density
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

출력:

```text
multi-label segmentation mask
local, scratch, ring, edge, shot_grid, random
```

## 6. Model-Assisted Correction

나중에 model-assisted annotation을 붙일 때 목표 흐름:

```text
trained model
-> prediction masks
-> run_segmentation_tool.py --prediction-json
-> human correction
-> updated pattern assets
-> retraining
```

기존 local tool의 `Load Prediction`, lasso, smart fit, proposal overlay는 이 단계의 핵심 UX입니다.

## 7. Validation

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py tests/test_segmentation_readiness.py tests/test_segmentation_training.py -q
```
