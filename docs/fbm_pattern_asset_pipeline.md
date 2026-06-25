# CVAT Pattern Asset And Hybrid Synthetic Data Pipeline

이 문서는 CVAT에서 만든 defect annotation을 reusable pattern asset으로 바꾸고, 그 asset을 real/base wafer에 합성해 segmentation 학습 데이터를 만드는 방법을 설명합니다.

## Goal

목표는 wafer-level class label이 아니라 pixel-level multi-label mask입니다.

```text
input:
  grade 0-7 wafer map

target:
  local mask
  scratch mask
  ring mask
  edge mask
  shot_grid mask
  random mask
```

합성 데이터가 필요한 이유는 실제 wafer에 defect가 보여도 pixel mask 정답이 거의 없기 때문입니다. CVAT로 실제 defect 조각을 라벨링하고, 이를 pattern asset으로 저장한 뒤 다양한 base wafer에 붙여 정답 mask가 있는 데이터를 만듭니다.

## Main Pipeline

```text
CVAT annotation
-> import_cvat_annotations.py
-> data/pattern_assets/<family>/<asset_id>
-> compose_synthetic_from_assets.py
-> data/synthetic/cvat_asset_composed
-> run_pattern_asset_pipeline.py
-> asset_segmentation_manifest.csv
-> train_unet_segmentation.py
```

## 1. Pattern Asset Library

CVAT importer와 legacy editor는 같은 asset format을 저장합니다.

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

파일 의미:

| File | Meaning |
|---|---|
| `grade.png` | 합성 wafer에 붙일 defect grade patch |
| `mask.png` | segmentation target으로 쓸 binary mask |
| `preview.png` | 사람 검수용 RGB crop |
| `metadata.json` | source sample, bbox, family, annotation source |

`metadata.json`의 `annotation_source.tool`이 `CVAT`이면 CVAT import에서 온 asset입니다.

## 2. Extensible Label Mapping

CVAT label은 [../configs/cvat/wafer_defect_labels.json](../configs/cvat/wafer_defect_labels.json)에서 관리합니다.

```json
{
  "name": "stby_blob",
  "display_name": "STBY / missing-test blob",
  "asset_family": "local",
  "grade_override": 7,
  "aliases": ["stby_fail", "missing_test_blob"]
}
```

이 구조 덕분에 CVAT label은 계속 늘릴 수 있고, 현재 composer가 이해하는 asset family로 매핑할 수 있습니다. 나중에 모델 target class를 늘릴 때는 label schema와 `wafermap.data.schema`를 같이 확장합니다.

## 3. Human Asset And Procedural Fallback

모든 defect를 사람이 직접 라벨링하지 않습니다.

| Family | Current strategy |
|---|---|
| `local` | CVAT/human asset primary |
| `scratch` | CVAT/human asset primary + procedural fallback |
| `ring` | CVAT/human asset primary |
| `edge` | procedural primary + optional CVAT asset |
| `shot_grid` | procedural primary + optional CVAT asset |
| `random` | procedural only |

`edge`, `shot_grid`, `random`은 규칙 기반 합성이 빠르고 일관됩니다. 다만 실제 edge band나 shot pattern이 특정 공정 signature를 가지면 CVAT asset으로 보강합니다.

## 4. Compose Synthetic Samples

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/cvat_asset_composed `
  --count 200 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

합성 규칙:

```text
output_grade[pixel] = max(base_grade[pixel], defect_grade[pixel])
pattern_masks[family][pixel] = 1 where defect mask is active
```

`source_jitter` placement는 원래 wafer 위치 signature를 유지하면서 약간만 흔듭니다. 공정 좌표가 중요한 wafer map 문제에서는 완전 random placement보다 이 기본값이 더 안전합니다.

## 5. Readiness And Smoke Validation

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/cvat_asset_composed `
  --work-dir outputs/cvat_pattern_asset_pipeline `
  --report-out outputs/reports/cvat_pattern_asset_project_report.html
```

검증 산출물:

| Output | Purpose |
|---|---|
| `asset_segmentation_manifest.csv` | training/validation sample manifest |
| `asset_segmentation_readiness_metrics.json` | family coverage, overlap, split metrics |
| `asset_segmentation_gallery.png` | quick visual gallery |
| `asset_segmentation_smoke.html` | input/target/loss wiring smoke test |
| `asset_embedding_smoke.html` | retrieval-style smoke diagnostic |
| `asset_unet_segmentation.html` | PyTorch dependency or training entry report |

Segmentation smoke test는 성능 평가가 아니라 data wiring 검증입니다. 실제 성능 판단은 PyTorch 환경에서 `train_unet_segmentation.py`를 돌린 뒤 family별 IoU/recall로 봅니다.

## 6. Model Definition

권장 1차 모델은 coordinate-aware small U-Net입니다.

Input channels:

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

Output heads:

```text
multi-label segmentation mask
wafer-level family score
optional encoder embedding
```

학습 entrypoint:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/cvat_pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/cvat_pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/cvat_pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

## 7. Active Learning Loop

나중에 model-assisted annotation을 붙일 때 목표 흐름:

```text
trained model
-> prediction masks
-> CVAT pre-annotation or review package
-> human correction
-> updated pattern assets / labels
-> retraining
```

기존 local editor의 `Load Prediction`, lasso, smart fit 코드는 이 단계의 UX reference로 보존합니다.

## 8. Legacy Local Editor

로컬 editor 실행:

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

이 경로는 CVAT를 대체하지 않습니다. 빠른 실험, single-wafer emergency correction, custom interaction reference로만 사용합니다.
