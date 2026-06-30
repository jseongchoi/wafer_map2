# 반도체 AI 설계 검토

이 문서는 현재 WaferMap의 AI/데이터 사이언스 설계가 왜 이런 방향인지 설명합니다.
핵심은 “실제 wafer를 완벽하게 다 라벨링하지 못한다”는 현실을 전제로,
학습 가능한 segmentation dataset을 만드는 것입니다.

## 1. 현재 기술 판단

| 판단 | 결론 |
|---|---|
| 문제 형태 | wafer classification이 아니라 multi-label segmentation |
| label 단위 | family별 binary mask |
| 데이터 확보 | 실제 asset + parametric rule + 합성 |
| 모델 역할 | 최종 자동 판정기보다 correction seed |
| 위치 정보 | wafer 좌표, radius, angle, edge distance를 input에 포함 |
| 검증 기준 | mask 품질, readiness, correction 가능성 |

## 2. 왜 segmentation인가

wafer 한 장에는 여러 불량이 같이 나타납니다.

```text
local blob + scratch + edge band
```

classification으로 “scratch wafer”라고만 하면 local과 edge 정보가 사라집니다.
따라서 target은 아래처럼 family별 mask가 되어야 합니다.

```text
target[local, y, x]
target[scratch, y, x]
target[edge, y, x]
```

이는 multi-class가 아니라 multi-label입니다. 한 pixel이 여러 family에 겹칠 수도
있으므로 sigmoid 기반 target이 더 자연스럽습니다.

## 3. 왜 합성 데이터를 쓰는가

실제 wafer만으로는 세 가지 문제가 있습니다.

1. 불량 수가 family별로 불균형합니다.
2. 모든 pixel mask를 사람이 만들기 어렵습니다.
3. 애매한 불량은 정의 자체가 흔들립니다.

합성 데이터는 이 문제를 완화합니다.

```text
실제 wafer에서 local asset 저장
ring은 radius/width rule로 생성
shot_grid는 shot layout rule로 생성
base wafer 위에 조합
-> 정답 mask를 아는 sample 생성
```

## 4. Observed target만 학습

모델은 관측 가능한 영역에서만 학습해야 합니다.

```text
target = pattern_mask & wafer_mask & valid_test_mask
```

wafer 밖이나 측정되지 않은 영역을 target으로 넣으면 모델이 잘못된 위치 prior를
배울 수 있습니다.

## 5. 입력 channel 설계

U-Net 입력은 단순 image 한 장이 아니라 좌표 정보를 함께 포함합니다.

예시 channel:

```text
severity_mean
severity_max
fail_density
wafer_mask
valid_test_mask
stby_mask
x_norm
y_norm
radial_norm
angle_sin
angle_cos
edge_distance_norm
```

반도체 wafer defect는 위치 의존성이 큽니다. `edge`, `ring`, `shot_grid` 같은
패턴은 좌표 정보를 주지 않으면 모델이 더 어렵게 배웁니다.

## 6. Resize와 label policy

image를 resize할 때는 input과 mask를 같은 좌표 변환으로 처리해야 합니다.

권장:

- severity 같은 연속/등급 값은 적절한 interpolation 또는 보존 규칙 사용
- binary mask는 nearest-like 방식으로 class boundary를 보존
- target은 최종적으로 0/1 binary로 유지

위험한 예:

```text
image만 resize하고 mask는 원본 좌표를 그대로 둠
```

이 경우 preview는 그럴듯해도 학습 target이 어긋납니다.

## 7. Asset composition 판단

합성할 때는 너무 예쁘거나 너무 비현실적인 sample을 만들면 안 됩니다.

좋은 합성:

```text
실제 local asset 1개
scratch procedural 1개
edge sector 1개
기존 wafer fail density 유지
```

나쁜 합성:

```text
wafer 대부분을 defect로 채움
family가 6개 모두 과하게 겹침
valid_test_mask 밖에 target 생성
```

## 8. 모델 성능 해석

초기 U-Net은 production 판정기가 아닙니다.
초기 성공 기준은 아래와 같습니다.

- prediction이 blank보다 낫다.
- 사람이 수정할 때 시간을 줄인다.
- 특정 family에서 false positive/false negative 경향이 보인다.
- 수정 결과를 asset으로 다시 축적할 수 있다.

즉, “한 번 학습해서 끝”이 아니라 correction loop가 설계의 일부입니다.

## 9. 남은 전문가 질문

전문가에게 확인해야 할 질문:

- `shot_grid`의 실제 shot layout metadata를 어디서 얻을 수 있는가?
- `ring`과 `edge`가 겹치는 wafer를 어떤 기준으로 나눌 것인가?
- diffuse 불량을 별도 family로 둘지, `mixed_unknown`으로 둘지?
- `random`을 defect family로 유지할지 baseline context로만 둘지?
- 제품/공정별로 family 정의가 달라지는가?

## 10. 이 검토로 추가된 체크

- 문서에서 `bbox`와 `mask`의 역할을 분리합니다.
- U-Net target contract를 `training_data_contract.md`에 명확히 둡니다.
- `parametric_mask`를 정식 라벨 유형으로 설명합니다.
- readiness report에서 family coverage와 mask ratio를 봅니다.
- HTML 문서로 생성해 작업자가 읽기 쉽게 만듭니다.
