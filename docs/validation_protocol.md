# 검증 방법

## 1. 목적

실제 wafer 데이터와 label을 repo에 포함할 수 없기 때문에 validation은 세 축으로 진행한다.

- 합성 데이터 내부 검증: generator가 의도한 구조와 mask를 올바르게 저장하는지 확인
- 전문가 시각 검증: 사용자의 현업 경험을 기준으로 합성 wafer가 그럴듯한지 평가
- 방법론 검증: 실제 데이터에서도 계산 가능한 feature만으로 유사 wafer 검색, grouping, score 해석이 가능한지 확인

## 2. 합성 데이터 내부 검증

Generator가 만든 sample마다 다음을 자동 확인한다.

```text
wafer_mask 밖의 severity가 0인가?
stby_mask가 chip boundary에 맞는 rectangular 영역인가?
valid_test_mask = 0인 곳과 stby_mask = 1인 곳이 일치하는가?
pattern_masks가 multi-label overlap을 허용하는가?
grade 값이 0~7 범위인가?
net die count가 목표 net die 수에 가까운가?
metadata와 array shape가 일치하는가?
```

## 3. Realism Metric

실제 데이터를 repo로 가져올 수 없어도, 보안 환경에서 같은 feature extractor를 실행해 집계 통계만 비교할 수 있다.

PNG raw 입력 sanity:

```text
허용 gray value가 0, 31, 151, 175, 191, 207, 223, 255뿐인가?
chip 전체가 255인 영역만 stby로 분리되는가?
일부 pixel만 255인 영역은 Grade 7로 유지되는가?
제품별 chip_blocks와 grid가 PNG shape와 정확히 맞는가?
wafer_mask_strategy가 제품 layout과 맞는가?
```

비교 대상 metric:

```text
grade histogram
stby chip count distribution
stby area ratio
fail density
grade-weighted severity
radial density profile
angular density profile
edge density ratio
edge-chip outer-face gradient
local hotspot distribution
shot-relative contrast
nearest-neighbor distance distribution
tile-level severity distribution
```

Visual realism 확인 기준:

```text
wafer body가 preview에서 즉시 보여야 한다.
edge는 smooth circle보다 net die/chip layout을 반영한 stair-step boundary가 적합하다.
wafer 밖 영역과 in-wafer Grade 0은 mask로 구분되어야 한다.
scratch/ring/edge/local은 실제 FBM처럼 sparse/noisy field 안에 묻힌 신호에 가까워야 한다.
stby는 왜곡 요인이자 의미 있는 패턴 신호로 둘 다 검토되어야 한다.
```

## 4. 전문가 Review Scorecard

각 synthetic batch에 대해 사용자가 1~5점으로 평가한다.

```text
visual realism
grade realism
stby realism
scratch realism
ring realism
edge realism
local/random realism
shot-relative realism
overlap realism
overall usefulness
```

점수 기준:

```text
1 = 실제와 거의 다름
2 = 일부 요소만 유사
3 = prototype 개발용으로는 가능
4 = 실제와 상당히 유사
5 = 실제와 구분하기 어려운 수준
```

## 5. Real Label 없이 방법론 검증

현재 핵심 검증은 FBM 자체에서 정보를 뽑아 유사 wafer를 찾을 수 있는지다.

검증 항목:

```text
1. 실제 데이터용 feature만 사용
   - synthetic mask ratio와 label은 feature에서 제외
   - 실제 wafer에서도 계산 가능한 feature만 사용

2. Similarity search
   - nearest-neighbor의 synthetic validation label Jaccard가 random pair보다 높은지 확인
   - top-k를 3, 5, 10으로 바꿔도 lift가 유지되는지 확인
   - 전체 평균 lift와 class별 lift를 분리해서 본다.
   - 전체 lift가 높아도 scratch/local 같은 작은 결함 class가 약하면 “정밀 결함 분해 완료”로 해석하지 않는다.

3. Grouping stability
   - feature subset/noise 반복 실험
   - co-association heatmap
   - 권장 coarse K 확인

4. Feature family ablation
   - edge, shot, stby, radial, angular, morphology feature를 제거했을 때 검색 품질 변화 확인
   - 관련 defect 관점의 lift가 예상대로 낮아지는지 확인
   - feature 제거 후 lift가 오히려 올라가는 경우는 해당 feature가 잡음일 수 있으므로 보강 또는 제거 후보로 표시한다.

5. Visual nearest-neighbor review
   - query wafer와 top neighbor gallery 생성
   - 전문가가 같은 계열로 볼 수 있는지 확인
   - 특히 scratch/local/ring은 synthetic label metric보다 expert visual review를 더 중요하게 본다.
```

## 6. Segmentation 검증

Synthetic label을 사용할 수 있는 다음 단계에서는 segmentation baseline을 검증한다.

```text
per-class IoU
Dice/F1
overlap detection recall
small defect recall
stby-hidden origin recall
clock-position report consistency
```

## 7. 보안 경계

Repo에는 실제 wafer image, raw array, lot/process 민감 정보를 저장하지 않는다.

허용 가능한 정보:

```text
익명화된 집계 통계
수동으로 작성한 realism feedback
synthetic config template
feature schema
code
```
