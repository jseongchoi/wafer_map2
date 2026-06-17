# Modeling Strategy

## 1. Positioning

현재 프로젝트의 성공 기준은 “딥러닝 모델을 학습했다”가 아니다. 먼저 synthetic FBM이 실제 Fail Bit Map의 구조를 충분히 닮아야 하고, 그 위에서 observable feature가 유사맵 검색과 defect score 해석에 쓸 수 있어야 한다.

현재 우선순위는 다음 순서다.

```text
synthetic generator
-> realism validation
-> interpretable observable feature extraction
-> nearest-neighbor similarity search
-> coarse grouping and feature-family ablation
-> synthetic-label segmentation baseline
-> secure real-unlabeled sanity check
-> self-supervised/domain adaptation
```

Process statistics table은 목표에서 사라진 것이 아니라 후순위다. 공정/설비/lot/recipe/chamber/test metadata와 조인할 수 있을 때 ANOVA 같은 통계 검정으로 확장한다.

## 2. Why Not AutoEncoder First

AutoEncoder는 정상 패턴을 복원하고 reconstruction error를 보는 데 유용할 수 있다. 하지만 현재 문제의 중심은 단순 anomaly detection이 아니다.

우리가 먼저 알고 싶은 것은 다음이다.

```text
어떤 defect pattern이 어디에, 얼마나 강하게 나타나는가?
stby chip이 어떤 defect origin을 가리고 있는가?
비슷한 defect 조합을 가진 wafer들이 서로 가까이 묶이는가?
```

AutoEncoder 단독 접근은 stby chip처럼 큰 rectangular missing region에 과도하게 반응할 수 있고, scratch/ring/edge/local/shot-relative 같은 의미적 패턴 분해에는 직접적이지 않다.

따라서 AutoEncoder는 1차 방법론이 아니라 보조 역할로 둔다.

```text
unlabeled representation pretraining
outlier 후보 탐색
synthetic-real domain gap 확인
rare pattern discovery
```

## 3. Baseline Layer

먼저 해석 가능한 rule/statistical baseline을 만든다.

이 baseline은 종착지가 아니다. 현재 feature logic은 사람이 이해할 수 있는 기준선을 만들기 위한 계측층이다. 최종 목표는 다음을 조합하는 hybrid 방향이다.

```text
semantic tensor contract
-> interpretable observable feature baseline
-> interest-conditioned retrieval
-> expert review feedback
-> morphology / segmentation model
-> self-supervised or metric-learning embedding
```

따라서 지금의 rule/statistical feature는 다음 역할을 한다.

- synthetic label로 검증 가능한 첫 기준선
- real unlabeled wafer에 바로 적용 가능한 안전한 feature
- 딥러닝 embedding 또는 segmentation 결과와 비교할 baseline
- expert review에서 사람이 판단할 기준 이름 제공
- downstream 모델이 무엇을 놓치는지 확인하는 diagnostic layer

현재 feature 계열:

```text
global: fail density, grade-weighted severity
stby: stby ratio
radial/angular: wafer 중심 기준 극좌표 profile
edge: edge density, edge-chip outer face, localized edge sector
ring/scratch morphology: radial peak, angular peak
local morphology: hotspot peak, top-k compactness, hotspot count
connected morphology: component count, blob compactness, triangle-like blob layout, scratch-like elongation
shot-relative: lower-left, bottom-edge, left-edge contrast
```

이 baseline은 다음 용도로 쓴다.

- 유사 wafer nearest-neighbor 검색
- coarse wafer grouping
- defect score ranking
- feature family ablation
- 향후 segmentation/model 결과의 비교 기준

## 4. Current CPU Validation

GPU 없이 노트북 CPU에서 먼저 검증한다.

현재 검증 축:

- label leakage 없이 observable feature만 사용했는가?
- nearest-neighbor가 random pair보다 더 비슷한 synthetic validation label을 갖는가?
- feature subset/noise를 줘도 그룹이 유지되는가?
- cluster 수와 top-k 검색 조건을 바꿔도 결과가 안정적인가?
- feature family를 제거하면 관련 defect 신호가 예상대로 약해지는가?
- nearest-neighbor gallery가 전문가 눈에도 납득 가능한가?

현재 scale pilot 결론:

- 155장 scale pilot에서 observable feature는 50개다.
- top-5 유사검색 lift는 약 1.36x이며, bootstrap 95% CI는 약 1.31x ~ 1.41x다.
- permutation p-value는 약 0.001로 random pair 대비 의미 있는 검색 신호가 유지된다.
- scale 기준 coarse cluster 수는 K=4로 재계산했고, shot/high-edge/local-rich cluster가 분리된다.
- shot, edge, ring, stby feature family는 retrieval에 실제로 기여한다.
- local은 관심 기준 검색에서 약 1.11x로 약하게 개선됐고, holdout에서는 더 명확하게 개선됐다.
- scratch는 scale에서도 약 1.10x 수준이라 wafer-level/component feature만으로는 부족하다.

관심 기준별 retrieval을 추가한 뒤의 결론:

- `edge_focus`와 `shot_focus`는 관심 feature subset을 분리하면 더 명확하게 평가된다.
- `local_focus`는 connected-component morphology를 넣으면 일부 개선된다.
- `scratch_focus`는 morphology를 넣어도 불안정하므로 단순 feature tuning보다 segmentation 또는 curve/arc detector로 넘기는 것이 맞다.
- 같은 wafer라도 관심 defect가 다르면 유사도의 기준도 달라져야 한다.

Defect feature target retrieval을 추가한 뒤의 결론:

- `class_name` 단위 검색은 scale/holdout 모두 안정적이다.
- `class_name + radial_zone`, `class_name + location_label`, `feature_key`처럼 관심 기준을 세분화해도 random baseline 대비 lift는 유지된다.
- 다만 세밀한 target은 support가 작아 P@K가 낮아지므로, 실무에서는 “정확한 label 자동 판정”보다 “후보 wafer ranking + expert review”로 써야 한다.
- edge/ring/shot은 feature-key 수준에서도 비교적 강하고, scratch/local/stby 위치 검색은 spatial model 보강 후보로 남는다.
- chip-level polar spatial feature를 추가한 결과, global similarity에 그대로 섞으면 전체 retrieval 품질이 떨어진다.
- 따라서 global grouping/confidence는 compact 50개 feature를 유지하고, polar spatial feature는 `class_location`/`feature_key` 같은 위치-aware retrieval target에서만 사용한다.
- 이 조건부 사용에서는 scale feature-key lift가 약 3.13x, holdout feature-key lift가 약 1.97x로 개선되어 위치-aware retrieval 신호가 존재함을 확인했다.

Resize benchmark를 추가한 뒤의 결론:

- naive grayscale resize는 stby/Grade7, none-wafer/Grade0 의미가 섞여 retrieval representation으로 부적합하다.
- semantic multichannel pooling도 flatten vector를 그대로 Euclidean retrieval에 쓰면 compact feature보다 훨씬 약하다.
- scale/holdout 모두 compact feature가 resize-only 표현보다 안정적이다.
- 따라서 resize는 global retrieval의 대체재가 아니라 segmentation, embedding, candidate proposal, patch crop 입력으로 사용한다.

Holdout stress 결론:

- 다른 seed/class prior/grade threshold 조건의 120장 holdout에서 top-5 lift는 약 1.40x로 유지됐다.
- bootstrap 95% CI는 약 1.32x ~ 1.48x이고 permutation p-value는 약 0.002다.
- edge/stby/ring은 holdout에서도 유지된다.
- local은 관심 기준 검색에서 약 1.33x로 개선되어 morphology 보강 효과가 확인됐다.
- shot은 scale에서는 강하지만 holdout에서는 약 1.23x로 낮아져 조건 변화에 민감하다.
- scratch는 holdout 관심 검색에서 약 0.92x로 무너져, 다음 단계에서 segmentation 또는 scratch-specific representation으로 넘긴다.

초기 36장 pilot에서는 top-5 lift 약 1.46x, K=4가 관측됐다. 따라서 K=4는 폐기된 값이 아니라 작은 batch 기준의 후보이고, 현재 운영 기준은 scale pilot의 K=3/4 안정 구간이다.

## 5. Segmentation Baseline

Synthetic label이 준비되어 있으므로 다음 단계에서는 multi-label segmentation baseline을 학습할 수 있다.

이 단계의 목적은 모든 것을 딥러닝으로 대체하는 것이 아니다. 현재 observable feature baseline이 약한 scratch/local/overlap 결함을 보완하고, wafer 안에서 관심 결함의 위치와 면적을 더 직접적으로 추정하기 위한 보조층이다.

원칙:

```text
output activation: sigmoid
loss: BCE/Focal/Dice 조합
target: class별 pattern mask
overlap: 허용
stby: 별도 mask이자 별도 pattern class로 처리
```

초기 후보:

```text
U-Net / U-Net++
DeepLab-style baseline
SegFormer-style lightweight transformer
ConvNeXt/Swin encoder + segmentation head
```

사용자 환경상 GPU가 제한적이므로 현재 local milestone에서는 학습보다 feature 검증과 report generation을 우선한다.

## 6. Secure Real-Unlabeled Direction

실제 wafer는 repo에 저장하지 않는다. 대신 보안 환경 안에서 다음만 실행할 수 있도록 준비한다.

```text
real FBM -> semantic tensor parsing
-> observable feature extraction
-> nearest-neighbor search against local feature store
-> coarse group and defect score report
```

실제 데이터 sanity check의 통과 기준:

- feature NaN/폭주 없음
- nearest-neighbor gallery가 전문가 육안으로 납득 가능
- synthetic mask-derived feature가 real inference path에 섞이지 않음
- stby와 Grade 7이 semantic channel에서 분리됨

## 7. Expected Outputs

현재 milestone의 핵심 output은 다음이다.

```text
1. wafer-level observable feature vector
2. nearest-neighbor similar wafer list
3. coarse group id
4. defect-family score summary
5. human-readable report and gallery
```

향후 process metadata가 붙으면 다음 질문으로 확장한다.

```text
특정 tool/chamber/recipe에서 shot_relative_score가 유의하게 높은가?
특정 공정 이후 edge_localized_score가 상승하는가?
특정 lot에서 stby_origin_hidden_score가 반복되는가?
```

## 8. References

- [DINOv2](https://github.com/facebookresearch/dinov2): self-supervised visual features.
- [AnomalyDINO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html): DINOv2 기반 few-shot industrial anomaly detection.
- [SAM2](https://ai.meta.com/research/sam2/): promptable segmentation and annotation/refinement candidate.
- [NVIDIA semiconductor defect workflow](https://developer.nvidia.com/blog/optimizing-semiconductor-defect-classification-with-generative-ai-and-vision-foundation-models/): semiconductor defect classification에서 VFM/VLM, SSL, domain adaptation 활용 흐름.
