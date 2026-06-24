# 로드맵

이 로드맵은 현재 WaferMap 프로젝트의 주 경로인 **FBM pattern asset 기반 hybrid synthetic data -> multi-label segmentation -> embedding retrieval**을 기준으로 정리합니다.

## Phase 0. 목표와 데이터 계약 고정

상태: 완료

완료 기준:

- 입력 wafer는 grade 0~7 map입니다.
- target은 `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random` multi-label mask입니다.
- `stby_pattern`은 모델 target에서 제외합니다.
- 절대좌표와 wafer 방향성은 중요한 공정 signature로 취급합니다.

핵심 문서:

- [FBM Pattern Asset 기반 Hybrid Synthetic Data 파이프라인](fbm_pattern_asset_pipeline.md)
- [프로젝트 개요](project_overview.md)

## Phase 1. Pattern Asset Builder

상태: 구현됨

완료된 기능:

- 브라우저 기반 wafer map annotation
- family별 mask 칠하기/지우기
- `Smart Fit`
- `Trace Scratch Line`
- `Load Prediction`
- `One Family Asset` / `Split Components`
- asset 저장 구조: `grade.png`, `mask.png`, `preview.png`, `metadata.json`

다음 보강:

- 실제 scratch wafer에서 human asset을 추가합니다.
- prediction mask를 불러와 사람이 수정하는 active learning 사용성을 더 검증합니다.

## Phase 2. Hybrid Synthetic Composer

상태: 구현됨

현재 방식:

```text
human asset:
  local, scratch, ring

procedural generator:
  scratch fallback, edge, shot_grid, random

composition:
  max(base_grade, defect_grade)
  pattern_masks[family]에 정답 mask 저장
```

최근 결과:

- `scratch` positive sample 11개
- `edge` positive sample 10개
- `shot_grid` positive sample 9개
- `random` positive sample 20개

다음 보강:

- product별 shot layout 정보가 있으면 `shot_grid` generator를 제품별로 보정합니다.
- edge defect의 angular sector, radial width, sparsity를 실제 wafer 피드백으로 보정합니다.

## Phase 3. Readiness / Smoke / Embedding 검증

상태: 구현됨

구현된 산출물:

- segmentation manifest
- family coverage metrics
- overlap summary
- segmentation smoke loss report
- embedding smoke top-k retrieval report

최근 결과:

- embedding top-1 lift 약 `1.53x`
- 모든 target family에 positive sample 존재
- 전체 테스트 `111 passed`

주의:

- segmentation smoke는 실제 모델 성능이 아니라 input/target/loss 연결 검증입니다.
- 성능 판단은 PyTorch U-Net 학습 이후에 해야 합니다.

## Phase 4. Small U-Net Segmentation

상태: entrypoint 구현됨, 현재 환경은 PyTorch 미설치

구현된 파일:

```text
scripts/train_unet_segmentation.py
```

실행 명령:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

현재 차단점:

- Codex 작업 환경에는 PyTorch가 설치되어 있지 않습니다.
- `pyproject.toml`에 optional dependency `train = ["torch>=2.3"]`를 추가했습니다.

완료 기준:

- family별 validation IoU/recall이 계산됩니다.
- 특히 `scratch recall`, `local small-blob recall`, `ring continuity`를 봅니다.
- 모델 output을 `fbm_prediction_masks/v1`로 export할 수 있어야 합니다.

## Phase 5. Prediction Export와 Active Learning

상태: 부분 구현

구현됨:

- 에디터가 `fbm_prediction_masks/v1` JSON을 불러올 수 있습니다.

남은 작업:

- U-Net prediction을 `fbm_prediction_masks/v1`로 export합니다.
- 사람이 prediction을 수정한 뒤 다시 asset library 또는 training label로 넣습니다.
- 수정 전후 mask 품질을 report에 기록합니다.

## Phase 6. Embedding 기반 유사 Wafer 검색

상태: smoke 구현됨

현재 방식:

- PCA baseline embedding으로 top-k retrieval smoke 검증
- cosine-style similarity 평가
- label Jaccard 기반 smoke metric 계산

다음 보강:

- U-Net encoder embedding을 저장합니다.
- `pattern_vector`와 `process_vector`를 분리합니다.
- 데이터가 커지면 FAISS index를 붙입니다.

## Phase 7. 실제 Wafer 검증

상태: 대기

필요한 사용자 입력:

- 실제 wafer 5~20장
- 특히 scratch/local/ring이 보이는 wafer
- 제품별 geometry 또는 shot layout 정보가 있으면 더 좋습니다.

검증 기준:

- 모델 mask가 실제 defect 위치를 잘 잡는가
- family score가 눈으로 보는 주 defect와 맞는가
- 비슷한 wafer top-k가 공정적으로 납득되는가
- 틀린 결과를 에디터에서 수정하고 다시 학습 데이터로 넣을 수 있는가

## 현재 우선순위

1. PyTorch 학습 환경에서 small U-Net을 실제로 학습합니다.
2. 실제 scratch human asset을 추가합니다.
3. U-Net prediction export를 구현합니다.
4. active learning loop를 실제 wafer 기준으로 반복합니다.
5. encoder embedding search를 U-Net backbone 기반으로 교체합니다.
