# WaferMap Defect Intelligence

## Current Direction: CVAT-First Dataset Pipeline

The primary annotation workflow is moving to CVAT. The in-repo Pattern Asset Editor is now a legacy fallback/reference tool, not the main UI we keep expanding.

Near-term scope:

```text
real/unlabeled wafer manifest
-> CVAT image package
-> CVAT annotation export
-> pattern asset import
-> real/base wafer synthetic dataset composition
```

Model training and model evaluation are intentionally deferred until the dataset creation path is stable.

New CVAT workflow entry points:

```powershell
python scripts/export_cvat_wafer_images.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --out-dir data/cvat_exports/smoke_task `
  --limit 10

python scripts/import_cvat_annotations.py `
  --cvat-xml data/cvat_exports/smoke_task/annotations.xml `
  --cvat-manifest data/cvat_exports/smoke_task/manifest.json `
  --assets-root data/pattern_assets
```

Label mapping is managed in `configs/cvat/wafer_defect_labels.json`, so CVAT labels can be added without hard-coding UI changes. See [CVAT Wafer Defect Annotation Workflow](docs/cvat_wafer_annotation_workflow.md).

이 프로젝트의 목표는 Wafer Fail Bit Map(FBM)에서 defect pattern을 pixel 단위로 뽑아내고, 그 패턴을 합성 데이터와 딥러닝 학습 데이터로 연결해 실제 wafer의 불량 위치/형태/강도/유사도를 수치화하는 것입니다.

단순히 “이 wafer는 ring이다”처럼 이름 하나를 맞히는 프로젝트가 아닙니다. 최종적으로는 아래 네 가지가 같이 필요합니다.

```text
1. defect family별 pixel mask 검출
2. defect 위치와 형태 수치화
3. wafer-level family score 산출
4. embedding vector 기반 유사 wafer 검색
```

## 현재 중심 구조

지금 프로젝트의 중심축은 `Pattern Asset Builder -> Hybrid Synthetic Composer -> Multi-label Segmentation/Embedding Model`입니다.

```text
실제 wafer map
-> Pattern Asset Builder에서 local/scratch/ring 같은 human asset 생성
-> scratch cold-start/edge/shot_grid/random은 코드가 procedural mask로 생성
-> Hybrid Synthetic Composer가 human asset과 procedural defect를 wafer에 합성
-> 합성 위치와 mask가 자동 label이 됨
-> segmentation/score/embedding 모델 학습
-> 모델 prediction을 다시 에디터에 띄워 사람이 보정
```

중요한 기준은 family별로 다릅니다. `random`은 사람이 누끼 따는 대상이 아니라 산발 fail/noise baseline으로 코드 생성합니다. `edge`와 `shot_grid`도 기본은 코드 생성이고, 특이한 실제 모양이 있을 때만 optional asset으로 저장합니다. `scratch`는 사람이 딴 asset이 최종 기준이지만, 라벨이 0개인 cold-start 상태를 피하려고 procedural fallback도 생성합니다. 사람이 우선 모아야 하는 asset은 `local`, `scratch`, `ring`입니다.

기존 compact feature, sanity report, nearest-neighbor 코드는 완전히 버리는 것이 아니라 보조 진단 도구로 둡니다. 하지만 현재 제품 방향의 1순위는 pixel mask 기반 딥러닝 파이프라인입니다.

## 핵심 문서

- [FBM 데이터 흐름 운영 가이드](docs/fbm_data_flow_guide.md)
- [FBM Pattern Asset 기반 딥러닝 파이프라인](docs/fbm_pattern_asset_pipeline.md)
- [실제 raw PNG 운영 안내서](docs/real_png_operator_runbook.md)
- [라벨 없는 실제 wafer 처리 절차](docs/real_unlabeled_workflow.md)
- [전문가 리뷰 절차](docs/expert_review_protocol.md)
- [Real Wafer 리뷰 체크리스트](docs/real_wafer_review_checklist.md)

## Pattern Asset Builder 실행

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

브라우저에서 defect family를 고르고 mask를 칠한 뒤 저장합니다.

- `Smart Fit`: 대충 칠한 seed에서 grade 기준으로 연결 영역을 자동 확장합니다.
- `Trace Scratch Line`: 대충 그은 scratch seed의 방향을 잡아 선형 mask를 확장합니다.
- `Load Prediction`: 모델 prediction mask를 불러와 사람이 수정할 수 있게 합니다.
- `One Family Asset`: 기본 저장 모드입니다. ring처럼 끊긴 패턴도 하나의 asset으로 저장합니다.
- `Split Components`: 독립 blob 여러 개를 일부러 따로 저장할 때만 사용합니다.

## 저장된 asset 검수

```powershell
python scripts/build_pattern_asset_report.py `
  --assets-root data/pattern_assets `
  --out outputs/reports/pattern_asset_library_report.html
```

검수할 때는 아래를 보면 됩니다.

```text
preview.png  : 사람이 보기 쉬운 defect crop
mask.png     : 실제 label로 쓸 binary mask
grade.png    : 합성에 쓸 grade 0~7 값
metadata.json: family, bbox, source_sample_id, composition rule
```

사용자 피드백은 “ring 하나인데 나뉘었다”, “mask가 배경까지 먹었다”, “family가 scratch가 아니라 local이다”처럼 주면 됩니다.

## Hybrid synthetic wafer 생성

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

현재 합성 규칙은 `max(base_grade, defect_grade)`입니다. multi-label mask를 보존하므로 한 pixel이 여러 family에 속할 수 있습니다. Procedural family도 사람이 딴 asset과 동일하게 `pattern_masks[family]`에 들어가므로 딥러닝 학습 라벨로 사용할 수 있습니다.

## 보조 진단 도구

기존 feature/retrieval 기반 해석 리포트는 아직 남겨둡니다. 실제 입력 파싱, sanity, nearest-neighbor가 깨졌는지 보는 smoke diagnostic 용도입니다.

```powershell
python scripts/interpret_fbm.py `
  --features-csv outputs/reports/real_png_batch/features.csv `
  --sanity-json outputs/reports/real_png_batch/sanity.json `
  --neighbors-csv outputs/reports/real_png_batch/neighbors.csv `
  --out outputs/interpretation
```

제품별 raw PNG 폴더에서 바로 실행할 때는 아래처럼 사용합니다.

```powershell
python scripts/interpret_fbm.py `
  --input data/raw `
  --geometry-json data/raw/product_geometry.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --out outputs/interpretation
```

## Small U-Net 학습

현재 환경에 PyTorch가 설치되어 있으면 아래 명령으로 multi-label segmentation 모델을 학습합니다.

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

현재 Codex 작업 환경에는 PyTorch가 없을 수 있으므로, 전체 pipeline은 먼저 dependency check report를 생성합니다.

## 테스트

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

현재 Pattern Asset 파이프라인의 핵심 테스트는 아래를 확인합니다.

- mask 저장
- disconnected ring을 기본값으로 하나의 family asset으로 저장
- `Split Components` 선택 시 component 분리
- asset library report 생성
- 저장된 asset을 이용한 max 합성

## 데이터 파일 원칙

- 실제 wafer raw image/array, 사람이 딴 pattern asset, 합성 sample, model checkpoint는 대용량/로컬 데이터라 repo에 저장하지 않습니다.
- 로컬 개발 입력은 `data/raw/`에 둘 수 있고, 인트라넷 운영 입력은 원하는 로컬/네트워크 경로를 그대로 쓸 수 있습니다.
- batch manifest는 기본적으로 `outputs/manifests/`에 생성됩니다.
- 결과 리포트는 기본적으로 `outputs/reports/` 또는 명령에서 지정한 폴더에 생성됩니다.
- Repo에는 code, config, schema, 문서, 테스트만 남깁니다.
- 합성 데이터의 oracle label/mask는 검증용이며, 실제 inference feature에 섞지 않습니다.
