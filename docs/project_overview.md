# Project Overview

WaferMap의 목표는 wafer Fail Bit Map(FBM)에서 defect segmentation model을 만들 수 있는 데이터셋 제작 파이프라인을 완성하는 것입니다. 현재 핵심은 모델 성능 경쟁이 아니라, 실제 wafer defect mask를 빠르게 직접 만들고 합성 데이터로 확장할 수 있는 안정적인 데이터 공장입니다.

## Current Product Shape

현재 메인 구조는 in-repo segmentation tool 기반 dataset pipeline입니다.

```text
raw or real-like wafer samples
-> real_unlabeled_manifest/v1
-> local segmentation tool
-> pattern asset library
-> hybrid synthetic data
-> segmentation readiness / smoke validation
-> train_unet_segmentation.py when training data is stable
-> export_unet_predictions.py
-> local segmentation tool correction loop
```

## Why Direct Tooling

wafer FBM은 단순 이미지 라벨링보다 wafer mask, valid-test mask, STBY, grade map, family별 multi-label mask가 함께 움직입니다. 그래서 이 repository 안에 wafer-specific segmentation tool을 두고, 저장 결과를 바로 pattern asset과 synthetic dataset으로 연결합니다.

| Layer | Responsibility |
|---|---|
| `run_segmentation_tool.py` | wafer manifest에서 샘플을 열고 family별 mask를 직접 작성 |
| `run_pattern_asset_editor.py` | 기존 사용자 명령을 위한 호환용 엔진 파일 |
| `build_pattern_asset_report.py` | 저장된 asset의 mask, preview, metadata 검수 |
| `compose_synthetic_from_assets.py` | real/base wafer에 pattern asset과 procedural fallback 합성 |
| `run_pattern_asset_pipeline.py` | readiness, smoke validation, model dependency check |
| `export_unet_predictions.py` | trained U-Net output을 segmentation tool prediction JSON으로 변환 |

## Defect Families

현재 model target 후보는 다음 family입니다.

| Family | Current source | Notes |
|---|---|---|
| `local` | human asset primary | blob, cluster 포함 |
| `scratch` | human asset primary + procedural fallback | 실제 scratch asset이 쌓이면 fallback 비중을 줄입니다. |
| `ring` | human asset primary | partial ring, broken ring 포함 |
| `edge` | procedural primary + optional human asset | edge band/sector는 코드 생성이 빠르지만 실제 예외 shape는 asset으로 보강합니다. |
| `shot_grid` | procedural primary + optional human asset | shot-relative repeated defect |
| `random` | procedural only | 구조 없는 sparse fail baseline |

기존 synthetic generator의 `stby_pattern`은 valid-test mask를 설명하는 보조 패턴이며, segmentation target으로 직접 학습할지는 별도 결정 사항입니다.

## What Stays

- `src/wafermap/data`, `src/wafermap/real`, `src/wafermap/synth`, `src/wafermap/assets`, `src/wafermap/training`
- local segmentation tool and pattern asset scripts
- synthetic composition scripts
- readiness/smoke/model entrypoint scripts
- report generation and validation tests

## Compatibility

`scripts/run_pattern_asset_editor.py`는 삭제하지 않습니다. 기존 테스트와 사용자 명령을 깨지 않기 위한 엔진/호환 파일로 유지하고, operator-facing 문서와 새 작업은 `scripts/run_segmentation_tool.py`를 기준으로 합니다.

## Current Validation State

최근 기준:

- pattern asset pipeline 테스트 통과
- segmentation readiness/smoke 테스트 통과
- PyTorch가 없는 환경에서는 `train_unet_segmentation.py --check-deps`가 dependency report를 생성합니다.

## Next Implementation Order

1. 실제 wafer manifest에서 `run_segmentation_tool.py`로 mask asset을 만듭니다.
2. `local`, `scratch`, `ring`, `edge` 실제 asset을 충분히 쌓습니다.
3. asset quality report에서 mask leakage, wrong family, split/merge issue를 바로 확인합니다.
4. hybrid synthetic data 품질 report를 보고 procedural fallback realism을 조정합니다.
5. PyTorch 환경에서 `train_unet_segmentation.py`를 실제 학습으로 돌립니다.
6. `export_unet_predictions.py`로 model prediction을 내보내고 다시 segmentation tool로 불러와 correction loop를 붙입니다.

## Practical Definition Of Done

이 프로젝트가 다음 단계로 넘어가려면 아래가 먼저 안정화되어야 합니다.

- annotator가 tool 안에서 family를 헷갈리지 않는다.
- edge band, ring 같은 global pattern을 일관되게 라벨링할 수 있다.
- 저장 결과가 `data/pattern_assets`에서 preview/mask/metadata로 검증 가능하다.
- 합성 sample의 `pattern_masks`가 model target으로 바로 쓰일 수 있다.
- smoke validation에서 family coverage와 overlap 문제가 즉시 보인다.
