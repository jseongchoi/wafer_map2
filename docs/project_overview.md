# 프로젝트 개요

## 목표

WaferMap 프로젝트의 목표는 FBM(Fail Bit Map) wafer map을 해석하는 딥러닝 기반 시스템을 만드는 것입니다. 단순히 wafer 하나에 defect 이름을 붙이는 분류기가 아니라, 다음 기능을 동시에 달성해야 합니다.

```text
1. defect family별 pixel mask 검출
2. defect 위치, 면적, 중심, edge 거리, component 수 수치화
3. wafer-level family score 산출
4. encoder embedding 기반 유사 wafer top-k 검색
5. 모델 prediction을 사람이 수정해 다시 학습 데이터로 넣는 active learning loop
```

## 현재 핵심 구조

현재 프로젝트의 중심은 **라벨 있는 hybrid synthetic data 생성 파이프라인**입니다.

```text
Pattern Asset Builder
  사람이 local/scratch/ring 같은 실제 defect를 pixel mask로 누끼 저장

Procedural Generator
  scratch cold-start, edge, shot_grid, random을 코드로 생성

Hybrid Synthetic Composer
  human asset과 procedural defect를 base wafer에 합성
  합성 위치와 mask를 알고 있으므로 자동 label 생성

Segmentation / Embedding Pipeline
  readiness manifest, segmentation smoke, embedding smoke, U-Net entrypoint 생성
```

## Family별 데이터 생성 책임

| Family | 현재 데이터 생성 방식 | 판단 |
|---|---|---|
| `local` | human asset primary | 실제 blob texture와 군집 형태가 중요합니다. |
| `scratch` | human asset primary + procedural fallback | 실제 scratch asset이 최종 기준이지만, cold-start 학습을 위해 radial/spin-arc scratch를 코드로 보강합니다. |
| `ring` | human asset primary | partial ring, 끊긴 ring, 두께와 반경이 실제 공정 signature를 담습니다. |
| `edge` | procedural primary, asset optional | wafer edge 거리와 angular sector rule로 안정적으로 합성할 수 있습니다. |
| `shot_grid` | procedural primary, asset optional | chip/shot 반복 좌표로 합성하는 편이 자연스럽습니다. |
| `random` | procedural only | 구조 없는 산발 fail baseline이므로 사람이 누끼 따는 대상이 아닙니다. |

## 구현된 주요 산출물

| 영역 | 산출물 | 상태 |
|---|---|---|
| Pattern annotation | `scripts/run_pattern_asset_editor.py` | 구현됨 |
| Asset library report | `scripts/build_pattern_asset_report.py` | 구현됨 |
| Hybrid synthetic data | `scripts/compose_synthetic_from_assets.py` | 구현됨 |
| End-to-end pipeline | `scripts/run_pattern_asset_pipeline.py` | 구현됨 |
| Segmentation readiness | `scripts/build_segmentation_readiness.py` | 구현됨 |
| Segmentation smoke | `scripts/train_segmentation_smoke.py` | 구현됨 |
| Embedding smoke | `scripts/train_embedding_smoke.py` | 구현됨 |
| Small U-Net entrypoint | `scripts/train_unet_segmentation.py` | 구현됨, 현재 환경은 PyTorch 미설치 |
| 이해용 HTML report | `outputs/reports/pattern_asset_project_report.html` | 생성됨 |

## 현재 검증 결과

최근 pipeline 실행 기준:

- `local`: positive sample 8개
- `scratch`: positive sample 11개
- `ring`: positive sample 20개
- `edge`: positive sample 10개
- `shot_grid`: positive sample 9개
- `random`: positive sample 20개
- embedding top-1 lift: 약 `1.53x`
- 전체 테스트: `111 passed`

## 현재 남은 병목

1. 현재 Codex 작업 환경에는 PyTorch가 없어 small U-Net 실제 학습은 실행하지 못했습니다.
2. `scratch`는 procedural fallback으로 label을 만들 수 있지만, 최종 realism을 위해 실제 scratch human asset이 필요합니다.
3. 실제 wafer batch에서 모델 prediction을 검토하고 수정하는 active learning loop를 더 돌려야 합니다.
4. 실제 제품별 geometry/shot layout 정보가 들어오면 `edge`와 `shot_grid` procedural generator를 더 정확하게 보정해야 합니다.

## 결론

프로젝트는 이제 “막연한 feature 실험” 단계가 아니라, **라벨 있는 synthetic data를 만들고, multi-label segmentation/embedding 모델로 연결하는 구조**로 정렬되어 있습니다. 다음 핵심 작업은 PyTorch 학습 환경에서 `scripts/train_unet_segmentation.py`를 실행하고, 실제 wafer 기반 human asset과 prediction 수정 루프를 반복하는 것입니다.
