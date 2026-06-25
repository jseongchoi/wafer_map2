# Operator Manual

이 문서는 WaferMap을 실제로 운영하는 사람이 따라 하는 절차서입니다. 목표는 “CVAT에서 라벨링한 defect를 model 학습 가능한 synthetic segmentation dataset으로 변환”하는 것입니다.

## Roles

| Role | Responsibility |
|---|---|
| Data operator | raw wafer 위치, manifest 생성, CVAT package export |
| Annotator | CVAT에서 defect label 작성 |
| Dataset owner | CVAT export import, pattern asset QC, synthetic dataset 생성 |
| ML owner | readiness report 확인, U-Net 학습, prediction feedback loop 운영 |

한 사람이 모두 할 수 있지만, 산출물 기준은 위 역할로 나누어 생각합니다.

## 0. Preconditions

Python package 설치:

```powershell
pip install -e .[dev]
```

학습까지 할 때:

```powershell
pip install -e .[dev,train]
```

확인:

```powershell
python -m pytest tests/test_cvat_annotation_workflow.py -q
```

## 1. Prepare Real Wafer Manifest

raw PNG folder에서 manifest를 만듭니다.

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root data/raw `
  --geometry-json data/raw/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

주요 산출물:

```text
outputs/manifests/real_png_batch_manifest.json
outputs/reports/real_png_batch/report.html
```

빠른 로컬 확인에는 synthetic smoke manifest도 사용할 수 있습니다.

```text
configs/eval/real_unlabeled_synthetic_smoke.json
```

## 2. Export CVAT Task Package

```powershell
python scripts/export_cvat_wafer_images.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --out-dir data/cvat_exports/real_png_task `
  --limit 100
```

산출물:

```text
data/cvat_exports/real_png_task/
  images/
  labels.json
  manifest.json
```

CVAT에는 `images/` 안의 PNG를 업로드합니다. `labels.json`은 label 생성 기준으로 사용합니다.

## 3. Configure CVAT Labels

원본 label schema:

```text
configs/cvat/wafer_defect_labels.json
```

기본 label:

| Label | Meaning |
|---|---|
| `local` | compact blob or local cluster |
| `scratch` | line, arc, or scratch-like defect |
| `ring` | full/partial ring or annulus |
| `edge` | edge band or edge sector |
| `shot_grid` | repeated shot-relative defect |
| `random` | sparse unstructured fail pattern |
| `stby_blob` | STBY/missing-test mosaic blob |

새 label이 필요하면 CVAT에서 임의로만 추가하지 말고 `configs/cvat/wafer_defect_labels.json`에 먼저 추가합니다.

## 4. Annotation Rules

### Local

- defect blob 또는 compact cluster만 감쌉니다.
- 주변 정상 background까지 크게 포함하지 않습니다.
- 여러 blob이 하나의 공정 defect로 보이면 하나의 polygon으로 묶어도 됩니다.

### Scratch

- visible scratch thickness를 따라갑니다.
- 너무 얇아서 polygon이 어려우면 중심선을 기준으로 약간 넓게 잡습니다.
- scratch 주변의 unrelated blob은 별도 `local`로 분리합니다.

### Ring

- ring 내부 전체 disk를 칠하지 않습니다.
- annulus 또는 arc 두께만 라벨링합니다.
- 끊긴 ring도 같은 physical ring이면 하나의 `ring` object로 유지합니다.

### Edge

- wafer edge 전체가 아니라 abnormal edge band/sector만 잡습니다.
- edge가 너무 넓고 규칙적이면 human asset보다 procedural rule 조정 후보입니다.
- 특별한 실제 edge shape가 있을 때만 asset으로 축적합니다.

### Shot Grid

- 반복 위치가 shot-relative로 보이면 `shot_grid`로 라벨링합니다.
- 단일 blob이면 `local`이 우선입니다.

### STBY Blob

- missing-test mosaic block은 `stby_blob`으로 라벨링합니다.
- importer가 grade 7 override를 적용하고 현재 asset family는 `local`로 저장합니다.
- 실제 물리 defect origin을 숨긴 것 같으면 review note에 남깁니다.

## 5. Export From CVAT

권장 형식:

```text
CVAT for images 1.1
```

현재 importer 지원 shape:

- polygon
- box

Brush/mask annotation을 주 workflow로 쓰려면 CVAT native mask RLE import를 추가해야 합니다.

## 6. Import CVAT Annotations

```powershell
python scripts/import_cvat_annotations.py `
  --cvat-xml data/cvat_exports/real_png_task/annotations.xml `
  --cvat-manifest data/cvat_exports/real_png_task/manifest.json `
  --assets-root data/pattern_assets
```

산출물:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json

data/pattern_assets/cvat_import_report.json
```

## 7. Pattern Asset QC

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

검수 기준:

| Check | Bad sign |
|---|---|
| mask tightness | background까지 크게 먹음 |
| family correctness | ring을 local로 저장함 |
| ring continuity | 하나의 ring이 여러 asset으로 쪼개짐 |
| STBY handling | missing-test blob인데 grade가 7로 유지되지 않음 |
| edge quality | 전체 wafer boundary를 무의미하게 칠함 |

QC에서 심각한 문제가 있으면 CVAT annotation을 고치고 다시 export/import합니다.

## 8. Build Synthetic Dataset

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/cvat_asset_composed `
  --count 200 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

주의:

- `random`은 human asset 대상이 아닙니다.
- `edge`, `shot_grid`은 procedural primary입니다.
- `scratch`는 human asset이 부족할 때 procedural fallback을 같이 씁니다.

## 9. Readiness Validation

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/cvat_asset_composed `
  --work-dir outputs/cvat_pattern_asset_pipeline `
  --report-out outputs/reports/cvat_pattern_asset_project_report.html
```

반드시 확인할 파일:

```text
outputs/cvat_pattern_asset_pipeline/asset_segmentation_manifest.csv
outputs/cvat_pattern_asset_pipeline/asset_segmentation_readiness.html
outputs/cvat_pattern_asset_pipeline/asset_segmentation_readiness_metrics.json
outputs/reports/cvat_pattern_asset_project_report.html
```

합격 기준:

- target family별 positive sample이 충분합니다.
- mask overlap이 의도한 multi-label 범위를 벗어나지 않습니다.
- gallery에서 defect가 wafer 안에 자연스럽게 보입니다.
- `stby_blob`이 local asset으로 들어가도 의도한 grade/mask가 유지됩니다.

## 10. Train Model

PyTorch 환경에서:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/cvat_pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/cvat_pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/cvat_pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

모델 평가에서 먼저 볼 것:

- `local` small blob recall
- `scratch` continuity
- `ring` arc/ring continuity
- `edge` false positives near normal boundary
- `stby_blob`을 local로 처리했을 때의 side effect

## 11. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `unknown CVAT label` | schema에 없는 label을 CVAT에서 사용 | `configs/cvat/wafer_defect_labels.json`에 추가하거나 CVAT label 수정 |
| imported asset count is 0 | CVAT XML image names do not match manifest | `data/cvat_exports/<task>/manifest.json`과 XML image name 확인 |
| STBY asset grade is 0 | label이 `stby_blob`이 아니거나 grade override 누락 | label schema와 annotation label 확인 |
| synthetic sample has no assets | `data/pattern_assets`가 비어 있음 | CVAT import 또는 asset report 먼저 실행 |
| edge/ring labels look too broad | polygon rule too loose | CVAT annotation guideline 재적용 |
| full tests fail after docs edit | documentation quality test catches stale direction | `tests/test_documentation_quality.py`와 docs links 확인 |

## 12. Release Checklist

작업 완료 전:

```powershell
python -m pytest -q --basetemp .pytest_tmp_release
git status --short
```

push 전 확인:

- README main commands가 동작 가능한 경로를 가리킨다.
- `docs/architecture.md`와 `docs/operator_manual.md`가 현재 코드 구조와 맞다.
- 새 script를 만들었다면 `scripts/README.md`에 등록했다.
- 새 label을 만들었다면 `configs/cvat/wafer_defect_labels.json`에 등록했다.
