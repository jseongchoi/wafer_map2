# WaferMap Defect Intelligence

WaferMap은 wafer Fail Bit Map(FBM)에서 defect segmentation model을 만들기 위한 데이터셋 제작 도구입니다. 현재 1순위는 모델 학습 자체가 아니라, 실제 wafer에서 defect mask를 직접 만들고 그 mask를 합성 데이터 생성 파이프라인으로 연결하는 것입니다.

## 현재 방향

핵심 방향:

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

## 주요 명령

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

학습된 U-Net 결과를 segmentation tool correction seed로 내보냅니다.

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --split val `
  --threshold 0.5
```

## 세그멘테이션 Family

현재 핵심 mask family:

| Family | 용도 |
|---|---|
| `local` | 작은 국소 blob 또는 cluster |
| `scratch` | 길게 이어지는 선형/곡선형 긁힘 |
| `ring` | 원형 또는 부분 원형 band |
| `edge` | wafer edge 근처 band/sector |
| `shot_grid` | shot 상대 위치가 반복되는 불량 |
| `random` | 뚜렷한 구조 없는 sparse baseline fail |

`stby_pattern`은 valid-test mask를 설명하는 보조 패턴이며, 현재 primary segmentation target에서는 제외합니다.

## 문서

- [문서 표지판](docs/index.html)
- [핵심 방향](docs/core_direction.md)
- [전체 실행 흐름](docs/end_to_end_workflow.md)
- [설계 구조](docs/architecture.md)
- [작업자 매뉴얼](docs/operator_manual.md)
- [세그멘테이션 도구 흐름](docs/segmentation_tool_workflow.md)
- [프로젝트 개요](docs/project_overview.md)
- [데이터 흐름 가이드](docs/fbm_data_flow_guide.md)
- [합성 데이터 파이프라인](docs/fbm_pattern_asset_pipeline.md)
- [반도체 AI 설계 검토](docs/semiconductor_ai_review.md)
- [실행 명령 지도](scripts/README.md)
- [불량 family 정의](docs/pattern_taxonomy.md)
- [라벨 데이터 가이드](docs/label_data_guidelines.md)
- [학습 데이터 규격](docs/training_data_contract.md)
- [모델 이해 가이드](docs/modeling_strategy.md)
- [로드맵](docs/roadmap.md)
- [용어 사전](docs/glossary.md)
- [문서 품질 검증표](docs/documentation_quality_audit.md)

## 검증

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

## 저장소 정책

원본 wafer, 생성된 pattern assets, synthetic samples, model checkpoints, reports는 로컬 산출물입니다. Git에는 code, configs, schemas, docs, tests만 넣습니다.
