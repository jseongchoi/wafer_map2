# FBM Pattern Asset 기반 Hybrid Synthetic Data 파이프라인

이 문서는 WaferMap 프로젝트의 현재 목표와 구현 방향을 설명합니다. 핵심은 **실제 FBM wafer map을 보고 defect pattern을 해석할 수 있는 딥러닝 모델을 만들기 위해, 라벨이 있는 합성 데이터를 안정적으로 생성하는 것**입니다.

최종 모델은 단순히 “이 wafer는 ring이다”처럼 하나의 class만 맞히는 모델이 아닙니다. 우리가 원하는 출력은 다음에 가깝습니다.

```text
입력:
  grade 0~7 wafer map

출력:
  family별 defect probability mask
  family별 score
  defect 면적, 중심 위치, edge 거리, radial 위치, component 수
  embedding vector 기반 유사 wafer top-k 검색
```

즉 목표는 **wafer map 해석 모델**입니다. 이를 만들려면 먼저 모델이 배울 수 있는 정답 mask가 필요합니다.

## 1. 왜 라벨 있는 합성 데이터가 필요한가

실제 wafer map에는 defect가 보이지만, 대부분은 pixel 단위 정답 mask가 없습니다. 딥러닝 segmentation 모델을 학습하려면 다음 정답이 필요합니다.

```text
이 pixel은 ring이다.
이 pixel은 scratch다.
이 pixel은 edge다.
이 pixel은 local blob이다.
```

실제 데이터를 전부 사람이 pixel 단위로 라벨링하는 것은 너무 느립니다. 그래서 이 프로젝트는 아래 전략을 씁니다.

```text
1. 실제 wafer에서 중요한 defect 모양만 사람이 누끼 딴다.
2. 누끼 딴 defect 조각을 family별 asset으로 저장한다.
3. 코드로 만들 수 있는 defect는 코드가 직접 mask와 grade를 생성한다.
4. human asset과 procedural defect를 base wafer 위에 합성한다.
5. 합성 위치와 mask를 알고 있으므로 자동으로 정답 라벨이 생긴다.
6. 이 데이터로 multi-label segmentation/embedding 모델을 학습한다.
```

이 방식의 본질은 **에디터를 만드는 것**이 아니라 **딥러닝 학습용 라벨 공장**을 만드는 것입니다.

## 2. 모든 family를 사람이 누끼 따면 안 되는 이유

처음에는 모든 family를 에디터에서 누끼 따는 방식처럼 보일 수 있습니다. 하지만 이것은 맞지 않습니다.

`random`은 사람이 누끼 따기 어렵습니다. 왜냐하면 random은 특정 모양이 아니라 산발적인 fail bit 분포입니다. 사람이 어디까지를 random defect로 칠할지 기준이 흔들립니다.

`edge`는 wafer edge 거리와 angular sector rule로 만들 수 있습니다. 사람이 edge를 매번 누끼 따는 것보다 코드가 edge band, local edge sector, edge intensity를 바꿔가며 생성하는 것이 더 빠르고 일관됩니다.

`shot_grid`도 실제 shot layout 정보가 있으면 코드로 만드는 편이 자연스럽습니다. 반복되는 chip/shot 상대좌표 defect이기 때문입니다.

따라서 family별 라벨 생성 책임을 나눕니다.

| Family | 라벨 생성 방식 | 이유 |
|---|---|---|
| `local` | human asset primary | 실제 blob 모양, blob 개수, texture가 중요합니다. |
| `scratch` | human asset primary + procedural fallback | 얇고 끊긴 선형 패턴이라 실제 wafer 누끼가 최종 기준입니다. 다만 cold-start 학습이 멈추지 않도록 radial/spin-arc scratch를 코드로도 생성합니다. |
| `ring` | human asset primary | partial ring, 끊긴 ring, 두께와 반경이 실제 공정 signature를 담습니다. |
| `edge` | procedural primary, asset optional | wafer edge 거리와 sector rule로 합성 가능합니다. |
| `shot_grid` | procedural primary, asset optional | chip/shot 반복 좌표로 합성 가능합니다. |
| `random` | procedural only | 산발 fail/noise baseline이라 사람이 누끼 따는 대상이 아닙니다. |

중요한 점은 procedural family도 학습 라벨에서 빠지는 것이 아니라는 점입니다. 코드가 만든 mask도 똑같이 `pattern_masks[family]`에 들어갑니다.

## 3. 현재 구현된 파이프라인

현재 구현은 다음 구조입니다.

```text
Pattern Asset Builder
  -> 사람이 local/scratch/ring 같은 defect를 pixel mask로 저장

Hybrid Synthetic Composer
  -> 저장된 human asset을 base wafer에 합성
  -> edge/shot_grid/random은 코드로 mask와 grade 생성
  -> multi-label pattern_masks 생성

Segmentation Readiness
  -> family별 positive sample 수, mask 비율, train/val split 확인

Segmentation Smoke
  -> 실제 모델 전 단계에서 input/target/loss 배선 검증

Embedding Smoke
  -> encoder embedding이 유사 family wafer를 찾을 수 있는지 최소 검증

Project Report
  -> 사용자가 현재 상태와 다음 작업을 이해할 수 있는 HTML 보고서 생성
```

## 4. Pattern Asset Builder의 역할

에디터는 사람이 필요한 family만 보강하는 도구입니다.

현재 에디터 기능은 다음과 같습니다.

- `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random` family 선택
- brush로 mask 칠하기/지우기
- `Smart Fit`: seed 주변의 grade 기준 연결 영역 확장
- `Trace Scratch Line`: scratch seed의 주방향을 계산해 선형 mask 확장
- `Load Prediction`: 모델이 예측한 mask를 불러와 사람이 수정
- `One Family Asset`: ring처럼 끊긴 패턴도 하나의 asset으로 저장
- `Split Components`: 여러 blob을 일부러 각각 다른 asset으로 저장

하지만 새 방법론에서는 `random`은 에디터로 누끼 따지 않는 것이 원칙입니다. `edge`와 `shot_grid`도 기본은 코드 생성이고, 실제 wafer에서 특이한 edge/shot-grid shape가 있을 때만 optional asset으로 저장합니다.

## 5. Hybrid Synthetic Composer

합성기는 두 종류의 defect를 함께 붙입니다.

### 5.1 Human asset 합성

사람이 저장한 asset은 다음 파일을 가집니다.

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

합성 규칙은 기본적으로 `max`입니다.

```text
output_grade[pixel] = max(base_grade[pixel], defect_grade[pixel])
output_mask[family][pixel] = 1 where defect mask is 1
```

배치는 기본적으로 `source_jitter`입니다. 즉 실제 wafer에서 누끼 딴 위치를 완전히 버리지 않고, 그 주변으로 조금 흔들어 합성합니다. 절대좌표 signature가 중요한 wafer map 특성 때문입니다.

### 5.2 Procedural defect 합성

코드로 생성하는 family는 현재 다음입니다.

```text
scratch
edge
shot_grid
random
```

`scratch`는 radial line과 spin-arc 두 가지 형태로 cold-start mask를 만듭니다. 이것은 실제 scratch asset을 대체하는 최종 정답이 아니라, scratch 라벨이 0개인 상태에서 segmentation 학습 루프가 멈추지 않도록 하는 임시 보강입니다.

`edge`는 wafer 중심에서의 radius와 angle을 이용해 edge sector mask를 만듭니다.

`shot_grid`는 chip 크기와 shot 반복 layout을 이용해 특정 shot-relative 위치에 반복 defect를 만듭니다.

`random`은 낮은 확률의 sparse impulse fail bit로 만듭니다. 이것은 사람이 누끼 따는 defect가 아니라, 구조 없는 fail baseline입니다.

이 procedural defect들도 최종 sample에는 아래처럼 들어갑니다.

```text
severity
pattern_masks[scratch]
pattern_masks[edge]
pattern_masks[shot_grid]
pattern_masks[random]
pattern_intensity[family]
metadata["procedural_patterns"]
```

따라서 딥러닝 모델은 사람이 만든 라벨과 코드가 만든 라벨을 같은 방식으로 학습할 수 있습니다.

## 6. 최종 모델 정의

추천 모델 구조는 다음입니다.

```text
Backbone:
  small U-Net / ConvNeXt-UNet / SegFormer 계열

Input channels:
  grade 0~7 normalized
  wafer_mask
  valid_test_mask
  x coordinate
  y coordinate
  radial distance
  angle encoding
  edge distance

Heads:
  multi-label segmentation head
  wafer-level family score head
  embedding head
```

출력은 다음 구조를 목표로 합니다.

```json
{
  "family_scores": {
    "local": 0.91,
    "scratch": 0.64,
    "edge": 0.31
  },
  "masks": {
    "local": "HxW probability map",
    "scratch": "HxW probability map"
  },
  "measurements": [
    {
      "family": "local",
      "area_ratio": 0.012,
      "centroid_xy": [512, 381],
      "radial_norm": 0.64,
      "angle_deg": 83,
      "edge_distance_norm": 0.18,
      "component_count": 2,
      "confidence": 0.91
    }
  ],
  "embedding": {
    "pattern_vector": "shape-similarity vector",
    "process_vector": "absolute-position-aware vector"
  }
}
```

## 7. 유사 wafer 검색

유사 wafer 검색은 segmentation 모델의 encoder embedding을 사용합니다.

초기에는 cosine similarity로 충분합니다.

```text
query wafer -> encoder -> embedding vector
database wafer embeddings와 cosine similarity 계산
top-k nearest wafer 반환
```

데이터가 커지면 FAISS index를 붙입니다.

embedding은 두 종류로 나누는 것이 좋습니다.

| Embedding | 목적 |
|---|---|
| `pattern_vector` | defect shape 중심 검색 |
| `process_vector` | 절대좌표/공정 위치 signature 포함 검색 |

wafer는 rotation/flip augmentation을 함부로 하면 안 됩니다. 3시 방향 defect와 9시 방향 defect는 공정적으로 다른 의미일 수 있기 때문입니다.

## 8. 실행 명령

에디터 실행:

```powershell
python scripts/run_pattern_asset_editor.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

hybrid synthetic data 생성:

```powershell
python scripts/compose_synthetic_from_assets.py `
  --base-sample-dir data/synthetic/fbm_grouping_scale_pilot/synth_000000 `
  --assets-root data/pattern_assets `
  --out-dir data/synthetic/asset_composed `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

전체 파이프라인 실행:

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --count 20 `
  --assets-per-wafer 3 `
  --procedural-families scratch,edge,shot_grid,random
```

PyTorch가 설치된 학습 환경에서 small U-Net 학습:

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --out outputs/pattern_asset_pipeline/asset_unet_segmentation.html `
  --metrics outputs/pattern_asset_pipeline/asset_unet_segmentation_metrics.json `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 20
```

생성되는 핵심 보고서:

```text
outputs/reports/pattern_asset_project_report.html
```

## 9. 사용자가 지금 피드백해야 하는 것

사용자가 지금 확인해야 하는 것은 모델 성능이 아닙니다. 아직은 라벨 공장을 만드는 단계입니다.

확인할 내용은 다음입니다.

| 확인 항목 | 질문 |
|---|---|
| human asset | local/scratch/ring 누끼가 실제 defect를 잘 담았는가? |
| ring 저장 | ring 하나가 쪼개지지 않고 하나의 asset으로 저장되는가? |
| scratch assist | Trace Scratch Line이 실제 scratch를 충분히 따라가는가? |
| edge procedural | edge sector가 실제 edge 불량처럼 보이는가? |
| shot_grid procedural | 반복 위치가 실제 shot/grid defect처럼 보이는가? |
| random procedural | random이 구조 없는 산발 fail baseline처럼 보이는가? |
| report coverage | 사람이 더 누끼 따야 할 family와 코드 조정 family가 구분되어 보이는가? |

## 10. 다음 구현 순서

1. `scratch`, `local`, `ring` human asset을 실제 wafer에서 더 모읍니다.
2. `scratch` procedural fallback은 cold-start용으로 쓰고, 실제 scratch asset이 들어오면 그쪽 기준으로 realism을 보정합니다.
3. `edge`, `shot_grid`, `random` procedural generator의 realism을 사용자가 본 결과로 보정합니다.
4. hybrid synthetic manifest를 train/val/test로 고정합니다.
5. family별 최소 positive sample 수를 보장합니다.
6. `scripts/train_unet_segmentation.py`로 small U-Net 모델을 학습합니다.
7. 모델 prediction을 `fbm_prediction_masks/v1`로 export합니다.
8. 에디터에서 prediction을 불러와 사람이 수정하는 active learning loop를 완성합니다.
9. encoder embedding을 저장하고 cosine/FAISS 기반 유사 wafer 검색을 붙입니다.

## 11. 논문/기술 근거

- [Semantic Segmentation-Based Wafer Map Mixed-Type Defect Pattern Recognition](https://ieeexplore.ieee.org/document/10122621/): family별 pixel mask를 예측하는 방향의 근거입니다.
- [Wafer Map Defect Pattern Classification and Image Retrieval Using CNN](https://ieeexplore.ieee.org/document/8263132/): CNN embedding으로 유사 wafer 검색을 하는 방향의 근거입니다.
- [Classification of Mixed-Type Defect Patterns in Wafer Bin Maps Using CNNs](https://ieeexplore.ieee.org/document/8368296/): mixed-type defect를 multi-label 관점으로 다루는 근거입니다.
- [An efficient deep learning framework for mixed-type wafer map defect pattern recognition](https://pubs.aip.org/aip/adv/article/14/4/045329/3283648/An-efficient-deep-learning-framework-for-mixed): 초기 모델을 작고 검증 가능한 구조로 시작하는 근거입니다.
