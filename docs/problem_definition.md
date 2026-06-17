# Problem Definition

## 1. Background

EDS 테스트 과정에서 wafer의 각 chip/die 내부 cell block 단위 fail bit 정보를 Fail Bit Map으로 표현한다. 한 chip은 제품에 따라 cell block width/height가 다르며, 현재 synthetic 실험에서는 chip 하나를 기본 `100 x 50` cell block으로 가정한다.

Wafer 하나는 약 600개 net die를 목표로 한다. 전체 FBM은 chip 배열과 cell block 해상도에 따라 수천 x 수천 이상의 고해상도 1채널 array가 된다.

Fail bit count는 원래 연속 count 값이지만 현재 분석에서는 Grade 0~7로 양자화된 값을 사용한다.

- Grade 0: fail bit 없음
- Grade 1~7: fail bit count bucket, 숫자가 클수록 severity가 큼
- None wafer: wafer 외부 영역
- Stby Fail Chip: test 자체가 불가능해서 chip 내부 fail bit 분포를 모르는 영역

Stby는 Grade 7과 같은 의미가 아니다. 시각화에서는 흰색으로 보일 수 있지만 분석 tensor에서는 `stby_mask`와 `valid_test_mask`로 별도 관리해야 한다.

## 2. Core Objective

입력으로 고해상도 1채널 FBM을 받았을 때, wafer 내부에 중첩된 여러 불량 신호를 정량화하고 유사한 wafer를 찾는 것이 현재 목표다.

최종적으로 원하는 출력 예시는 다음과 같다.

```text
Wafer summary
- 12시 방향: ring/arc 계열 신호 강함
- 3시 edge: edge-local defect 가능성
- 5시 방향: local blob + stby-hidden origin 가능성
- shot coordinate: lower-left 반복 defect 가능성
- 유사 wafer: 과거 wafer A/B/C와 feature space에서 가까움
```

이 프로젝트는 단순 wafer-level classification이 아니다. 다음 문제로 정의한다.

```text
고해상도 1채널 Fail Bit Map에서 여러 공정성 불량 패턴이 중첩된 상태를
observable feature, 유사맵 검색, coarse grouping, 그리고 향후 segmentation으로 분해하는
multi-view spatial defect parsing 문제
```

## 3. Current Required Capabilities

- 실제 wafer 데이터를 사용하지 않고 유사 synthetic wafer map을 생성한다.
- synthetic map이 현업 관점에서 plausibility를 갖는지 expert review로 검증한다.
- FBM 자체에서 observable feature를 추출한다.
- 유사 wafer 검색과 coarse grouping을 수행한다.
- defect 관점별 score를 만든다.
- stby chip이 결함 신호를 가릴 수도 있고, 그 자체가 구조적 패턴일 수도 있음을 반영한다.
- 실제 wafer는 repo에 저장하지 않고, 보안 환경에서 feature extraction과 nearest-neighbor sanity check를 수행할 수 있게 준비한다.

ANOVA/통계 검정은 현재 1차 목표가 아니다. 공정 데이터, 설비 데이터, lot, recipe, chamber, test condition과 FBM feature를 조인할 수 있을 때 수행하는 후속 분석이다.

## 4. Current Pattern Scope

현재 synthetic taxonomy는 다음 계열을 포함한다.

- `scratch`: spin arc 또는 center-radial 계열 scratch
- `ring`: donut, partial ring, edge-ring 계열 불량
- `edge`: edge ring, localized edge, edge arc 계열 불량
- `local`: 단일 blob, 쌍 blob, 삼각형처럼 연결되는 triple blob
- `random`: 공간 구조가 약한 산발 fail
- `shot_grid`: photo shot/reticle 상대좌표에서 반복되는 lower-left, bottom-edge, left-edge 계열 불량
- `stby_pattern`: stby chip이 random 또는 defect origin-coupled 형태로 발생하는 패턴

초기 flow/water-streak 계열은 사용자 피드백에 따라 현재 scope에서 제외했다.

## 5. Modeling Principle

Stby Fail Chip은 severity가 아니라 missing-test signal이다.

Stby는 두 의미를 동시에 가진다.

- 관측 불가능 영역: chip 내부 fail bit 분포를 모른다.
- 패턴 신호: stby chip 배열 자체가 scratch, ring, edge, local origin과 관련될 수 있다.

따라서 분석 tensor는 다음처럼 분리한다.

```text
severity: Grade 0~7
wafer_mask: wafer 내부/외부
valid_test_mask: 실제 fail bit이 관측된 영역
stby_mask: chip-level missing-test 영역
```

## 6. Constraints

- 실제 wafer 데이터는 보안상 repo에 포함하지 않는다.
- 실제 wafer 정답 label은 없다고 가정한다.
- 실제 데이터 검증은 expert review와 보안 환경 내부 실행으로 수행한다.
- 제품마다 chip cell block size와 die layout이 다르다.
- 여러 defect는 중첩될 수 있다.
- 작은 위치에서 발생한 defect가 stby chip에 가려질 수 있다.
- stby chip은 왜곡 요인이면서 동시에 의미 있는 defect 신호일 수 있다.

## 7. Non-Goals For Current Milestone

- 실제 공정 root cause를 확정적으로 예측하지 않는다.
- 실제 wafer 데이터 없이 production 성능을 주장하지 않는다.
- 처음부터 GPU 기반 foundation model 학습을 목표로 하지 않는다.
- grayscale image 하나만 보고 Grade 7과 Stby를 같은 의미로 취급하지 않는다.
- ANOVA를 현재 핵심 목표로 두지 않는다.
