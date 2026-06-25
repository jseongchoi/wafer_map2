# WaferMap Defect Intelligence

WaferMap은 wafer Fail Bit Map(FBM)에서 defect segmentation model을 만들기 위한 데이터셋 제작 도구입니다. 현재 1순위는 모델 학습 자체가 아니라, 실제 wafer에서 defect 영역을 안정적으로 라벨링하고 그 라벨을 합성 데이터 생성 파이프라인으로 연결하는 것입니다.

## Current Direction

메인 워크플로우는 CVAT-first입니다.

```text
real/unlabeled wafer manifest
-> CVAT image package
-> CVAT annotation export
-> reusable pattern assets
-> real/base wafer synthetic dataset composition
-> segmentation readiness and smoke validation
```

모델 학습과 평가(`train_unet_segmentation.py`)는 데이터셋 생성 경로가 안정화된 뒤 붙입니다. 지금은 1) Defect Segmentation Model용 데이터셋 제작, 2) 실제 defect pattern을 real/base wafer에 붙여 합성 데이터를 만드는 경로에 집중합니다.

## Main Commands

CVAT에 올릴 wafer preview package를 만듭니다.

```powershell
python scripts/export_cvat_wafer_images.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --out-dir data/cvat_exports/smoke_task `
  --limit 10
```

CVAT에서 `CVAT for images 1.1` 형식으로 annotation을 export한 뒤 pattern asset으로 가져옵니다.

```powershell
python scripts/import_cvat_annotations.py `
  --cvat-xml data/cvat_exports/smoke_task/annotations.xml `
  --cvat-manifest data/cvat_exports/smoke_task/manifest.json `
  --assets-root data/pattern_assets
```

생성된 pattern asset을 base wafer에 합성합니다.

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/cvat_asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

합성 데이터가 segmentation 학습 입력으로 쓸 수 있는지 점검합니다.

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/cvat_asset_composed `
  --work-dir outputs/cvat_pattern_asset_pipeline `
  --report-out outputs/reports/cvat_pattern_asset_project_report.html
```

## Extensible Labels

CVAT label 정의는 [configs/cvat/wafer_defect_labels.json](configs/cvat/wafer_defect_labels.json)에 있습니다. 새 label을 추가할 때는 코드에 하드코딩하지 말고 이 JSON에 추가합니다.

현재 핵심 매핑:

| CVAT label | Asset family | Notes |
|---|---|---|
| `local` | `local` | blob/local defect |
| `scratch` | `scratch` | scratch defect |
| `ring` | `ring` | ring or partial ring defect |
| `edge` | `edge` | edge band/sector defect |
| `shot_grid` | `shot_grid` | repeated shot-relative defect |
| `random` | `random` | sparse baseline fail pattern |
| `stby_blob` | `local` | STBY/missing-test mosaic blob, grade override 7 |

`stby_blob`은 CVAT에서는 별도 label로 보이지만 현재 synthetic composer에서는 `local` asset family로 소비합니다. 나중에 모델 output channel을 따로 만들기로 결정하면 `wafermap.data.schema`의 model class와 이 mapping을 같이 확장합니다.

## Legacy Local Editor

[scripts/run_pattern_asset_editor.py](scripts/run_pattern_asset_editor.py)는 유지합니다. 다만 이제 메인 UI가 아니라 CVAT로 처리하기 어려운 빠른 실험, fallback correction, UX reference 용도입니다.

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

새 annotation 기능은 기본적으로 CVAT workflow에 먼저 붙입니다.

## Docs

- [CVAT Wafer Defect Annotation Workflow](docs/cvat_wafer_annotation_workflow.md)
- [Project Overview](docs/project_overview.md)
- [Data Flow Guide](docs/fbm_data_flow_guide.md)
- [Hybrid Synthetic Data Pipeline](docs/fbm_pattern_asset_pipeline.md)
- [Pattern Taxonomy](docs/pattern_taxonomy.md)
- [Roadmap](docs/roadmap.md)
- [Glossary](docs/glossary.md)

## Validation

전체 테스트:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

CVAT workflow만 빠르게 확인:

```powershell
python -m pytest tests/test_cvat_annotation_workflow.py -q
```

## Repository Policy

원본 wafer, 생성된 pattern assets, synthetic samples, model checkpoints, reports는 로컬 산출물입니다. Git에는 code, configs, schemas, docs, tests만 넣습니다.
