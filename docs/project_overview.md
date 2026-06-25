# Project Overview

WaferMap의 목표는 wafer Fail Bit Map(FBM)에서 defect segmentation model을 만들 수 있는 데이터셋 제작 파이프라인을 완성하는 것입니다. 현재 핵심은 모델 성능 경쟁이 아니라, 실제 wafer defect를 빠르게 라벨링하고 합성 데이터로 확장할 수 있는 안정적인 데이터 공장입니다.

## Current Product Shape

현재 메인 구조는 CVAT-first dataset pipeline입니다.

```text
raw or real-like wafer samples
-> real_unlabeled_manifest/v1
-> CVAT image package
-> CVAT annotation export
-> pattern asset library
-> hybrid synthetic data
-> segmentation readiness / smoke validation
-> train_unet_segmentation.py when training data is stable
```

## Why CVAT First

직접 만든 browser editor는 빠른 실험에는 좋지만, 실제 라벨링 제품으로 커지면 task 관리, label 관리, reviewer flow, 대량 image annotation, audit trail을 다시 만들어야 합니다. CVAT는 이 영역을 이미 해결합니다.

이 repository에서 직접 책임질 부분은 CVAT 바깥의 wafer-specific pipeline입니다.

| Layer | Responsibility |
|---|---|
| CVAT | annotation UI, task/review workflow, polygon/brush labeling |
| `export_cvat_wafer_images.py` | wafer manifest를 CVAT image package로 변환 |
| `configs/cvat/wafer_defect_labels.json` | 확장 가능한 label schema |
| `import_cvat_annotations.py` | CVAT XML을 reusable pattern asset으로 변환 |
| `compose_synthetic_from_assets.py` | real/base wafer에 pattern asset과 procedural fallback 합성 |
| `run_pattern_asset_pipeline.py` | readiness, smoke validation, model dependency check |

## Defect Families

현재 model target 후보는 다음 family입니다.

| Family | Current source | Notes |
|---|---|---|
| `local` | CVAT/human asset primary | blob, cluster, STBY-derived mosaic asset 포함 |
| `scratch` | CVAT/human asset primary + procedural fallback | 실제 scratch asset이 쌓이면 fallback 비중을 줄입니다. |
| `ring` | CVAT/human asset primary | partial ring, broken ring 포함 |
| `edge` | procedural primary + optional CVAT asset | edge band/sector는 코드 생성이 빠르지만 실제 예외 shape는 asset으로 보강합니다. |
| `shot_grid` | procedural primary + optional CVAT asset | shot-relative repeated defect |
| `random` | procedural only | 구조 없는 sparse fail baseline |

`stby_blob`은 CVAT label로는 별도 관리하지만 현재 asset family는 `local`입니다. 기존 synthetic generator의 `stby_pattern`은 valid-test mask를 설명하는 보조 패턴이며, segmentation target으로 직접 학습할지는 별도 결정 사항입니다.

## What Stays

- `src/wafermap/data`, `src/wafermap/real`, `src/wafermap/synth`, `src/wafermap/assets`, `src/wafermap/training`
- CVAT export/import scripts
- synthetic composition scripts
- readiness/smoke/model entrypoint scripts
- report generation and validation tests

## What Becomes Legacy

`scripts/run_pattern_asset_editor.py`는 삭제하지 않고 [Legacy Pattern Asset Editor](legacy_pattern_asset_editor.md)로 격리합니다. 이미 구현한 lasso, smart fit, low-resolution interaction, proposal overlay는 reference 가치가 있습니다. 다만 새 annotation product 기능은 기본적으로 CVAT workflow에 붙입니다.

## Current Validation State

최근 기준:

- CVAT export/import workflow 테스트 통과
- pattern asset pipeline 테스트 통과
- 전체 테스트 `111 passed`
- PyTorch가 없는 환경에서는 `train_unet_segmentation.py --check-deps`가 dependency report를 생성합니다.

## Next Implementation Order

1. 실제 wafer manifest에서 CVAT task package를 생성합니다.
2. CVAT label schema를 제품/공정별 label 증가에 견딜 수 있게 운영합니다.
3. CVAT polygon/box import를 안정화하고, brush mask가 주 workflow가 되면 CVAT mask RLE import를 추가합니다.
4. `local`, `scratch`, `ring`, `edge` 실제 asset을 충분히 쌓습니다.
5. hybrid synthetic data 품질 report를 보고 procedural fallback realism을 조정합니다.
6. PyTorch 환경에서 `train_unet_segmentation.py`를 실제 학습으로 돌립니다.
7. model prediction을 다시 CVAT 또는 review tool로 되돌리는 active learning loop를 붙입니다.

## Practical Definition Of Done

이 프로젝트가 다음 단계로 넘어가려면 아래가 먼저 안정화되어야 합니다.

- annotator가 CVAT에서 label을 헷갈리지 않는다.
- `stby_blob`, edge band, ring 같은 global pattern을 일관되게 라벨링할 수 있다.
- CVAT import 결과가 `data/pattern_assets`에서 preview/mask/metadata로 검증 가능하다.
- 합성 sample의 `pattern_masks`가 model target으로 바로 쓰일 수 있다.
- smoke validation에서 family coverage와 overlap 문제가 즉시 보인다.
