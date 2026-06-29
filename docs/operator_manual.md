# Operator Manual

이 문서는 WaferMap을 실제로 운영하는 사람이 따라 하는 절차서입니다. 목표는 “직접 작성한 defect mask를 model 학습 가능한 synthetic segmentation dataset으로 변환”하는 것입니다.

## Roles

| Role | Responsibility |
|---|---|
| Data operator | raw wafer 위치, manifest 생성 |
| Annotator | `run_segmentation_tool.py`에서 defect mask 작성 |
| Dataset owner | pattern asset QC, synthetic dataset 생성 |
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
python -m pytest tests/test_pattern_asset_pipeline.py -q
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

## 2. Open The Segmentation Tool

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --assets-root data/pattern_assets
```

빠른 smoke sample:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

모델 예측 mask를 correction seed로 쓸 때:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest outputs/manifests/real_png_batch_manifest.json `
  --sample-id <sample_id> `
  --prediction-json outputs/predictions/fbm_prediction_masks.json `
  --assets-root data/pattern_assets
```

## 3. Segmentation Families

| Family | Meaning |
|---|---|
| `local` | compact blob or local cluster |
| `scratch` | line, arc, or scratch-like defect |
| `ring` | full/partial ring or annulus |
| `edge` | abnormal edge band or edge sector |
| `shot_grid` | repeated shot-relative defect |
| `random` | sparse unstructured fail pattern |

## 4. Annotation Rules

### Local

- defect blob 또는 compact cluster만 감쌉니다.
- 주변 정상 background까지 크게 포함하지 않습니다.
- 여러 blob이 하나의 공정 defect로 보이면 하나의 mask로 묶어도 됩니다.

### Scratch

- visible scratch thickness를 따라갑니다.
- 너무 얇으면 중심선을 기준으로 약간 넓게 잡습니다.
- scratch 주변의 unrelated blob은 별도 `local`로 분리합니다.

### Ring

- ring 내부 전체 disk를 칠하지 않습니다.
- annulus 또는 arc 두께만 라벨링합니다.
- 끊긴 ring도 같은 physical ring이면 하나의 `ring` mask로 유지합니다.

### Edge

- wafer edge 전체가 아니라 abnormal edge band/sector만 잡습니다.
- edge가 너무 넓고 규칙적이면 human asset보다 procedural rule 조정 후보입니다.
- 특별한 실제 edge shape가 있을 때만 asset으로 축적합니다.

### Shot Grid

- 반복 위치가 shot-relative로 보이면 `shot_grid`로 라벨링합니다.
- 단일 blob이면 `local`이 우선입니다.

## 5. Save Pattern Assets

툴에서 `Save Assets`를 누르면 family별 asset이 저장됩니다.

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

## 6. Pattern Asset QC

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
| STBY handling | missing-test 영역을 물리 defect와 섞음 |
| edge quality | 전체 wafer boundary를 무의미하게 칠함 |

QC에서 심각한 문제가 있으면 segmentation tool에서 mask를 다시 저장합니다.

## 7. Build Synthetic Dataset

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 200 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

주의:

- `random`은 human asset 대상이 아닙니다.
- `edge`, `shot_grid`은 procedural primary입니다.
- `scratch`는 human asset이 부족할 때 procedural fallback을 같이 씁니다.

## 8. Readiness Validation

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

반드시 확인할 파일:

```text
outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv
outputs/pattern_asset_pipeline/asset_segmentation_readiness.html
outputs/pattern_asset_pipeline/asset_segmentation_readiness_metrics.json
outputs/reports/pattern_asset_project_report.html
```

합격 기준:

- target family별 positive sample이 충분합니다.
- mask overlap이 의도한 multi-label 범위를 벗어나지 않습니다.
- gallery에서 defect가 wafer 안에 자연스럽게 보입니다.
- `stby_pattern`이 target channel에 섞이지 않습니다.

## 9. Train Model

PyTorch 환경에서:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

Training checks train-split positive coverage before running. If a target family has no positive train sample, rebuild the synthetic dataset or use `--allow-incomplete-target-coverage` only for a wiring/debug run. If validation positives are missing, training may run but the report marks that family metric as uninformative.

모델 평가에서 먼저 볼 것:

- `local` small blob recall
- `scratch` continuity
- `ring` arc/ring continuity
- `edge` false positives near normal boundary
- prediction mask를 tool에 다시 불러왔을 때 correction이 쉬운지

## 10. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| saved asset count is 0 | tool에서 저장할 mask가 비어 있음 | active family와 mask pixel count 확인 |
| synthetic sample has no assets | `data/pattern_assets`가 비어 있음 | segmentation tool 저장 또는 asset report 먼저 실행 |
| edge/ring labels look too broad | mask rule too loose | tool에서 visible defect band/arc만 다시 지정 |
| prediction masks do not load | schema or sample id mismatch | `fbm_prediction_masks/v1`와 `sample_id` 확인 |
| full tests fail after docs edit | documentation quality test catches stale direction | `tests/test_documentation_quality.py`와 docs links 확인 |

## 11. Release Checklist

작업 완료 전:

```powershell
python -m pytest -q --basetemp .pytest_tmp_release
git status --short
```

push 전 확인:

- README main commands가 동작 가능한 경로를 가리킨다.
- `docs/architecture.md`와 `docs/operator_manual.md`가 현재 코드 구조와 맞다.
- 새 script를 만들었다면 `scripts/README.md`에 등록했다.
- 새 mask family를 만들었다면 schema, asset, training target 테스트를 같이 업데이트했다.
