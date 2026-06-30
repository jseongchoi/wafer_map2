# 모델 이해 가이드

이 문서는 WaferMap에서 쓰는 segmentation 모델이 무엇을 입력받고, 무엇을 예측하고,
그 결과를 어떻게 해석해야 하는지 설명합니다.

한 줄로 말하면 현재 모델은 아래 일을 합니다.

```text
wafer map과 위치 정보를 입력받는다
-> 각 pixel이 local/scratch/ring/edge/shot_grid/random 중 어디에 속하는지 확률로 예측한다
-> 사람이 segmentation tool에서 그 예측을 수정한다
```

## 1. 모델의 역할

현재 U-Net은 최종 자동 판정기가 아닙니다.
목표는 사람이 처음부터 전부 mask를 그리지 않도록 “수정 가능한 초안”을 만들어주는 것입니다.

```text
나쁜 기대:
U-Net이 실제 wafer 불량을 완벽히 자동 판정한다.

좋은 기대:
U-Net이 대략적인 local/scratch/ring 후보 mask를 만들고,
사람이 틀린 부분을 빠르게 고친다.
```

이 correction 결과가 다시 pattern asset으로 쌓이면 다음 학습 데이터가 좋아집니다.

## 2. 왜 U-Net인가

U-Net은 image segmentation에서 자주 쓰는 encoder-decoder 구조입니다.
쉽게 말하면 아래처럼 동작합니다.

```text
Encoder:
  wafer 전체를 보며 큰 구조를 이해한다.
  예: ring인지, edge인지, shot 반복인지

Decoder:
  다시 pixel 위치로 돌아오며 mask를 만든다.
  예: 이 pixel은 scratch일 확률 0.8

Skip connection:
  encoder 초반의 위치 정보를 decoder에 다시 전달한다.
  예: 작은 local blob이나 얇은 scratch가 사라지지 않게 돕는다.
```

wafer map은 “어디에 있는가”가 매우 중요합니다.
그래서 이 프로젝트의 U-Net은 단순 이미지 한 장이 아니라 위치 channel을 함께 받습니다.

## 3. 입력 tensor

학습 코드에서 input은 아래 shape입니다.

```text
X.shape = [12, output_size, output_size]
```

현재 기본 `output_size`는 학습 script에서 96입니다.
각 channel은 아래 의미를 가집니다.

| Channel | 이름 | 쉬운 설명 |
|---:|---|---|
| 0 | `severity_mean` | downsample 영역의 평균 fail/severity |
| 1 | `severity_max` | 작은 결함을 놓치지 않기 위한 최대 severity |
| 2 | `fail_density` | fail pixel이 얼마나 빽빽한지 |
| 3 | `wafer_mask` | wafer 안쪽인지 |
| 4 | `valid_test_mask` | 실제 측정 가능한 영역인지 |
| 5 | `stby_mask` | STBY/missing-test context |
| 6 | `x_norm` | 좌우 위치 |
| 7 | `y_norm` | 상하 위치 |
| 8 | `radial_norm` | 중심에서 얼마나 먼지 |
| 9 | `angle_sin` | 각도 정보의 sin |
| 10 | `angle_cos` | 각도 정보의 cos |
| 11 | `edge_distance_norm` | edge에서 얼마나 안쪽인지 |

예를 들어 `edge` defect는 edge 근처에서 자주 나타나므로
`edge_distance_norm`이 도움이 됩니다. `ring`은 중심 거리인 `radial_norm`이
중요합니다. `shot_grid`는 제품별 shot layout metadata가 더 있으면 더 좋아집니다.

## 4. Target tensor

모델이 맞혀야 하는 정답은 아래 shape입니다.

```text
Y.shape = [6, output_size, output_size]
```

target channel 순서:

```text
0 local
1 scratch
2 ring
3 edge
4 shot_grid
5 random
```

각 값은 0 또는 1입니다.

```text
Y[local, y, x] = 1이면
그 pixel은 local defect target이다.
```

중요한 점은 `stby_pattern`이 target이 아니라는 것입니다.
STBY는 defect 정답이 아니라 context로 봅니다.

## 5. 왜 softmax가 아니라 sigmoid인가

일반 classification에서는 하나만 고르는 softmax를 자주 씁니다.
하지만 wafer defect는 한 pixel 또는 가까운 영역에 여러 family가 겹칠 수 있습니다.

예:

```text
edge 근처에 scratch가 지나감
ring 위에 local blob이 얹힘
```

그래서 이 모델은 family마다 독립적인 확률을 냅니다.

```text
P(local at pixel) = 0.75
P(scratch at pixel) = 0.10
P(edge at pixel) = 0.80
```

이 구조가 sigmoid multi-label segmentation입니다.

## 6. Loss는 무엇을 의미하나

학습 script는 PyTorch가 있으면 `BCEWithLogitsLoss`를 사용합니다.

```text
logits = model(X)
loss = BCEWithLogitsLoss(logits, Y)
probability = sigmoid(logits)
```

BCE는 각 family/pixel에 대해 “0인지 1인지”를 맞추는 손실입니다.
불량 pixel은 정상 pixel보다 훨씬 적으므로 `pos_weight`를 사용해 positive target이
너무 무시되지 않게 합니다.

## 7. Prediction은 어떻게 mask가 되나

모델 출력은 바로 mask가 아니라 확률입니다.

```text
scratch probability = 0.73
threshold = 0.5
scratch mask = 1
```

export 명령:

```powershell
python scripts/export_unet_predictions.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model outputs/models/asset_unet_segmentation.pt `
  --out outputs/predictions/fbm_prediction_masks.json `
  --threshold 0.5
```

threshold를 낮추면 더 많이 잡지만 false positive가 늘 수 있습니다.
threshold를 높이면 더 보수적으로 잡지만 작은 defect를 놓칠 수 있습니다.

## 8. 학습 명령

```powershell
python scripts/train_unet_segmentation.py `
  --manifest outputs/pattern_asset_pipeline/asset_segmentation_manifest.csv `
  --model-out outputs/models/asset_unet_segmentation.pt `
  --output-size 96 `
  --epochs 6 `
  --batch-size 4
```

학습 전에 반드시 readiness manifest가 있어야 합니다.

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

## 9. 모델 결과를 어떻게 봐야 하나

처음에는 mIoU 같은 숫자 하나보다 아래 질문이 더 중요합니다.

| 질문 | 이유 |
|---|---|
| prediction이 완전 blank보다 나은가? | correction seed 역할을 하는지 확인 |
| local blob을 대략 잡는가? | 작은 결함 보존 확인 |
| scratch가 끊겨도 방향은 맞는가? | 사람이 고칠 수 있는지 확인 |
| edge/ring을 서로 헷갈리는가? | 위치 channel과 taxonomy 점검 |
| shot_grid를 못 잡는가? | shot layout metadata 부족 가능성 |

## 10. 실제 wafer에서 왜 틀릴 수 있나

synthetic data로 시작한 모델은 실제 wafer에서 틀릴 수 있습니다.
이건 자연스러운 현상입니다.

주요 원인:

- 실제 wafer의 noise/intensity 분포가 합성과 다름
- 실제 shot layout이 synthetic rule과 다름
- ring/edge family 정의가 제품마다 다름
- 학습 asset 수가 부족함
- `mixed_unknown`이 많은데 family 정의를 억지로 단순화함

해결은 모델 구조를 무작정 키우는 것이 아니라 correction loop를 돌리는 것입니다.

```text
실제 wafer prediction
-> 사람이 수정
-> 수정된 mask를 asset으로 저장
-> 합성 데이터에 반영
-> 재학습
```

## 11. 모델 설명을 라벨 작업에 연결하기

라벨 작업자가 알아야 할 핵심:

```text
bbox는 모델 정답이 아니다.
mask가 모델 정답이다.
parametric rule도 최종적으로는 mask가 되어야 한다.
애매한 라벨은 모델을 똑똑하게 하지 않고 헷갈리게 한다.
```

예를 들어 `shot_grid`를 사람이 전부 칠하지 않아도 됩니다.
하지만 rule로 만든 최종 `pattern_masks[shot_grid]`는 반드시 있어야 합니다.

## 12. 현재 모델의 한계

현재 small U-Net은 시작점입니다.

한계:

- 실제 wafer로 충분히 검증된 production model이 아닙니다.
- family coverage가 부족하면 특정 class를 배우지 못합니다.
- `shot_grid`는 layout 정보가 부족하면 잘 잡기 어렵습니다.
- 아주 얇은 scratch는 resize 과정에서 약해질 수 있습니다.

그래도 이 모델은 중요한 역할을 합니다.
사람이 모든 mask를 처음부터 그리는 대신, 첫 prediction을 만들고 그것을 고쳐
다음 학습 데이터로 되돌리는 loop를 시작하게 해줍니다.

## 13. 관련 코드

| 역할 | 코드 |
|---|---|
| input/target tensor 생성 | `src/wafermap/training/segmentation.py` |
| U-Net 학습 | `scripts/train_unet_segmentation.py` |
| prediction export | `scripts/export_unet_predictions.py` |
| readiness manifest | `scripts/build_segmentation_readiness.py` |
| correction tool | `scripts/run_segmentation_tool.py` |
