# Synthetic Data Plan

## 1. Goal

실제 wafer 데이터를 사용할 수 없으므로, 도메인 지식 기반 synthetic wafer map generator를 먼저 만든다.

이 generator의 1차 목표는 모델 학습이 아니라 다음이다.

- 사용자가 보기에 실제 Fail Bit Map과 충분히 유사한지 검증
- 주요 defect pattern을 조절 가능한 parameter로 생성
- Stby Fail Chip의 왜곡과 패턴성을 동시에 재현
- EDA, feature extraction, similarity search 개발용 benchmark 제공

## 2. Generation Pipeline

권장 생성 순서:

```text
1. Wafer geometry 생성
2. Net die/chip layout 생성
3. Net die/chip layout 기반의 stair-step wafer body mask 생성
4. Continuous latent severity field 생성
5. Pattern별 sparse/noisy latent intensity field 합성
6. Random/background fail noise와 edge lift 추가
7. Severity field를 grade 1~7로 quantization
8. Stby chip mask 생성
9. Stby 영역의 observed severity를 hidden/missing 처리
10. Visualization image 생성
11. Ground-truth masks와 metadata 저장
```

중요: Stby를 먼저 흰색으로 칠하는 방식이 아니라, 내부적으로는 `valid_test_mask=0`으로 숨긴 뒤 visualization 단계에서 흰색으로 렌더링한다.

검증용 preview에서는 wafer body가 분명히 보여야 한다. 실제 Fail Bit Map은 net die/chip layout의 영향으로 edge가 계단식으로 보이는 것이 자연스럽다. 사람이 합성 품질을 평가할 수 있도록 grade colormap, stair-step wafer boundary, Stby chip block을 명확히 렌더링한다.

## 3. Wafer Geometry

초기 기본값:

```yaml
target_net_die: 600
chip_blocks:
  width: 100
  height: 50
wafer_shape: circular
edge_exclusion: configurable
```

die layout은 원형 wafer boundary 안에 들어가는 rectangular chip grid로 생성한다. 제품별 chip size와 die count가 바뀔 수 있으므로 모든 geometry 값은 config로 둔다.

## 4. Grade Quantization

실제 grade는 fail bit count 구간을 양자화한 값이다. Synthetic에서는 continuous severity field를 만든 뒤 grade threshold로 변환한다.

```text
latent severity: float, 0.0~1.0+
grade: integer, 0~7
```

threshold는 config로 둔다.

```yaml
grade_thresholds:
  grade_1: 0.05
  grade_2: 0.12
  grade_3: 0.22
  grade_4: 0.35
  grade_5: 0.50
  grade_6: 0.70
  grade_7: 0.90
```

나중에 실제 제품의 grade 정책을 알면 이 threshold를 바꿔 calibration한다.

## 5. Stby Modeling

Stby Fail Chip은 chip 단위로 발생한다. Synthetic에서는 다음 mode를 지원한다.

```text
random_stby
scratch_like_stby
ring_like_stby
edge_like_stby
local_cluster_stby
mixed_stby
```

Stby는 두 layer로 저장한다.

```text
stby_mask: chip 전체가 테스트 불가인 영역
stby_pattern_mask: stby 발생 구조의 class mask
```

Stby가 underlying defect와 coupling될 수 있도록 한다.

예:

```text
scratch defect가 지나간 chip 중 일부가 Stby로 전환됨
ring defect 중심부 chip이 Stby로 전환됨
edge defect가 강한 chip이 Stby로 전환됨
```

이렇게 해야 "혜성이 충돌한 곳은 Stby로 인해 측정 불가지만 주변에 scratch/ring 흔적이 남는" 상황을 재현할 수 있다.

## 6. Initial Synthetic Dataset Splits

초기 실험 dataset:

```text
debug: 20 samples
pilot: 200 samples
mvp: 1,000 samples
stress: 5,000+ samples
```

각 split은 동일한 generator code와 다른 random seed/config로 생성한다.

## 7. Human Expert Validation

실제 wafer 데이터를 repo에 넣을 수 없으므로, synthetic sample은 사용자가 눈으로 검증한다.

검증 질문:

```text
1. 실제 Fail Bit Map처럼 보이는가?
2. 어떤 부분이 너무 인위적인가?
3. Stby chip 형태와 크기가 실제와 맞는가?
4. grade 분포가 너무 균일하거나 과장되지 않았는가?
5. 작은 overlap defect가 실제처럼 보이는가?
6. edge/scratch/ring/local/random 패턴이 구분 가능한가?
7. wafer body와 none-wafer 영역이 검증 preview에서 즉시 구분되는가?
8. defect가 두꺼운 수학 함수 band처럼 보이지 않고 sparse/noisy하게 보이는가?
```

합성 데이터는 사용자의 피드백을 받아 parameter를 조정하는 방식으로 개선한다.

## 8. MVP Commands

```powershell
python scripts/generate_synthetic.py --config configs/synth/debug.json --out data/synthetic/debug --count 3
python scripts/validate_synthetic.py --data data/synthetic/debug
python scripts/extract_features.py --data data/synthetic/debug --out outputs/reports/synthetic_features.csv
```

Debug config는 Stby chip 시각 검증을 위해 `stby_pattern` probability를 1.0으로 둔다.
