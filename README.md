# WaferMap Defect Intelligence

WaferMap은 wafer Fail Bit Map(FBM)에서 defect segmentation model을 만들기 위한 데이터셋 제작 도구입니다. 현재 1순위는 모델 학습 자체가 아니라, 실제 wafer에서 defect mask를 직접 만들고 그 mask를 합성 데이터 생성 파이프라인으로 연결하는 것입니다.

## Current Direction

Core direction:

```text
FBM maps
-> defect generation
-> multi-defect synthetic maps
-> multi-defect segmentation training and validation
-> local segmentation tool for real-data pattern asset extraction
```

메인 워크플로우는 repo 안의 직접 세그멘테이션 툴을 기준으로 합니다.

```text
real/unlabeled wafer manifest
-> in-repo segmentation tool
-> reusable pattern assets
-> real/base wafer synthetic dataset composition
-> segmentation readiness and smoke validation
```

모델 학습과 평가(`train_unet_segmentation.py`)는 데이터셋 생성 경로가 안정화된 뒤 붙입니다. 지금은 1) Defect Segmentation Model용 mask asset 제작, 2) 실제 defect pattern을 real/base wafer에 붙여 합성 데이터를 만드는 경로에 집중합니다.

## Main Commands

wafer manifest에서 직접 세그멘테이션 툴을 엽니다.

```powershell
python scripts/run_segmentation_tool.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

저장된 pattern asset을 검수합니다.

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

생성된 pattern asset을 base wafer에 합성합니다.

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

합성 데이터가 segmentation 학습 입력으로 쓸 수 있는지 점검합니다.

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

## Segmentation Families

현재 핵심 mask family:

| Family | Use |
|---|---|
| `local` | blob/local defect |
| `scratch` | scratch defect |
| `ring` | ring or partial ring defect |
| `edge` | abnormal edge band/sector |
| `shot_grid` | repeated shot-relative defect |
| `random` | sparse baseline fail pattern |

`stby_pattern`은 valid-test mask를 설명하는 보조 패턴이며, 현재 primary segmentation target에서는 제외합니다.

## Docs

- [Core Direction](docs/core_direction.md)
- [Architecture](docs/architecture.md)
- [Operator Manual](docs/operator_manual.md)
- [Segmentation Tool Workflow](docs/segmentation_tool_workflow.md)
- [Project Overview](docs/project_overview.md)
- [Data Flow Guide](docs/fbm_data_flow_guide.md)
- [Hybrid Synthetic Data Pipeline](docs/fbm_pattern_asset_pipeline.md)
- [Semiconductor AI Data Science Review](docs/semiconductor_ai_review.md)
- [Scripts Command Map](scripts/README.md)
- [Pattern Taxonomy](docs/pattern_taxonomy.md)
- [Roadmap](docs/roadmap.md)
- [Glossary](docs/glossary.md)

## Validation

기본 테스트는 느린 end-to-end 테스트를 건너뜁니다.

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

느린 pipeline/training/report 테스트까지 포함할 때:

```powershell
python -m pytest -q --run-slow --basetemp .pytest_tmp_full
```

직접 세그멘테이션 툴과 pattern asset pipeline만 빠르게 확인:

```powershell
python -m pytest tests/test_pattern_asset_pipeline.py -q
```

## Repository Policy

원본 wafer, 생성된 pattern assets, synthetic samples, model checkpoints, reports는 로컬 산출물입니다. Git에는 code, configs, schemas, docs, tests만 넣습니다.
