# Project Alignment Review

기준일: 2026-06-17

## 목적 재확인

이 프로젝트의 목적은 1채널 고해상도 Wafer Fail Bit Map에서 실제 wafer에도 적용 가능한 표현을 만드는 것이다.

목표 표현은 단일 classification label이 아니라 다음 downstream task에 쓰이는 feature/retrieval 기반 표현이다.

```text
semantic FBM tensor
-> compact observable wafer feature
-> similar wafer retrieval
-> interest-conditioned defect retrieval
-> defect score / feature table
-> expert review feedback
-> real-unlabeled secure workflow
```

ANOVA나 공정 metadata 해석은 현재 목표가 아니다. 공정/설비/lot/recipe/chamber metadata가 붙은 뒤 feature table과 조인해 수행할 후속 분석이다.

## 현재 진행 상태

현재 프로젝트는 synthetic-only proof 단계에서 real-unlabeled workflow와 expert review loop를 닫는 단계로 넘어왔다.

확인된 근거:

- Synthetic generator는 Grade 0~7, none-wafer, valid-test, stby, edge/local/shot/ring/scratch 계열을 생성한다.
- Global retrieval은 compact observable feature 50개 기준을 유지한다.
- Scale 155장 top-k retrieval lift는 약 1.36x, holdout 120장은 약 1.40x로 random baseline 대비 신호가 있다.
- Interest-conditioned retrieval과 structured defect feature retrieval은 class, class_location, feature_key 기준에서 신호가 있다.
- Resize-only representation은 global retrieval 대체재로 부적합하다고 판단했다.
- Patch proposal은 edge/local/stby review 후보 생성 보조층으로 유지한다.
- Curve proposal은 ring/center arc review 후보 생성 보조층으로 유지한다.
- Scratch는 rule/proposal 과투자를 멈추고 segmentation 또는 scratch-specific line representation으로 분리한다.
- Real-unlabeled workflow MVP는 semantic `.npz` manifest, feature CSV, sanity JSON, nearest-neighbor CSV, expert review template까지 연결되어 있다.
- Global nearest-neighbor 경로에서는 `polar_*`, `stby_polar_*` 위치 feature를 제외한다.

## 본질 정렬 판정

판정: 본질대로 진행 중이다.

이유:

- 실제 inference feature는 observable-only 계약을 유지한다.
- Synthetic oracle인 `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 검증용으로 분리되어 있다.
- Global retrieval과 위치-aware retrieval의 feature 계약이 분리되어 있다.
- Proposal 계열을 주 경로가 아니라 review 후보 축소용 보조층으로 제한했다.
- 현재 다음 작업이 새 모델 도입이 아니라 real-unlabeled contract와 expert review loop에 맞춰져 있다.

가장 큰 리스크는 synthetic 성능을 real 성능으로 오해하는 것이다. 지금 수치는 방법론 가능성의 근거이지, 실제 wafer 성능 인증이 아니다. 따라서 다음 gate는 실제 보안 환경에서 semantic `.npz`를 export해 sanity와 top-k review를 받는 것이다.

## 지금 우선해야 할 작업

1. Real-unlabeled manifest/schema를 운영 계약으로 고정한다.
   - `real_unlabeled_manifest/v1`
   - `observable_fbm_features/v1`
   - standard key 예시와 `array_keys` 매핑 예시 유지
   - raw image/array repo 저장 금지

2. Expert review loop를 실제 업무 판단으로 닫는다.
   - nearest-neighbor CSV에서 reviewer template 생성
   - `reviewer_decision`, `query_defect_family`, `neighbor_defect_family`, `retrieval_failure_mode`, `next_action` 수집
   - summary의 `next_action_queue`를 feature/model backlog로 연결

3. Drift/sanity report를 real wafer 첫 반입 gate로 사용한다.
   - NaN/inf, invalid severity, stby/Grade7 혼동, chip index 오류 차단
   - reference 대비 feature drift는 성능 metric이 아니라 parser/분포 sanity로 해석

4. Scratch는 별도 track으로 유지한다.
   - 현재 compact wafer feature와 proposal만으로는 holdout 신호가 약하다.
   - line enhancement, skeleton/continuity, lightweight segmentation 후보로 분리한다.

## 외부 논문/유사 연구에서 얻는 기준

외부 연구 흐름은 현재 방향을 뒷받침하지만, 곧장 대형 모델로 넘어가라는 의미는 아니다.

- Iterative Cluster Harvesting for Wafer Map Defect Patterns (2024)
  - wafer map clustering에서 defect의 위치, 밀도, 회전, 모양 변화가 어렵다는 점을 명시한다.
  - feature extraction, dimensionality reduction, clustering을 반복해 manual labeling을 돕는 방향을 제시한다.
  - 현재 프로젝트의 `feature -> retrieval/grouping -> expert review` 방향과 잘 맞는다.
  - https://arxiv.org/abs/2404.15436

- Graph-theoretic spatial filtering for mixed-type wafer bin maps (2020/2021)
  - mixed-type wafer map 문제를 systematic pattern filtering과 spatial clustering으로 나눠 본다.
  - raw defect chip 간 인접성을 이용하는 관점은 scratch/local 같은 공간 연결성 보강 후보와 맞닿아 있다.
  - 현재 프로젝트에서는 대체 방법론이 아니라 morphology/connected-component 보강의 참고선으로 둔다.
  - https://arxiv.org/abs/2006.13824

- WaferSegClassNet (Computers in Industry, 2022)
  - mixed-type wafer defect에 대해 classification과 segmentation을 함께 수행하는 lightweight encoder-decoder 방향을 제시한다.
  - scratch/local/overlap처럼 wafer-level feature가 약한 계열을 segmentation track으로 분리한 현재 판단과 맞다.
  - 다만 real-unlabeled secure workflow와 expert review gate가 먼저이며, segmentation은 그 다음 보강 layer다.
  - https://arxiv.org/abs/2207.00960

- Semi-supervised / autoencoder / one-class wafer map 연구
  - VAE, teacher-student, adversarial autoencoder, DSVDD 같은 방법은 label scarcity와 anomaly detection에 유용하다.
  - 하지만 stby, none-wafer, Grade0, local/scratch/ring 의미 분리를 먼저 안정화하지 않으면 reconstruction error가 업무 defect 의미와 어긋날 수 있다.
  - https://arxiv.org/abs/2311.12840
  - https://arxiv.org/abs/2107.08823

따라서 논문 기준으로도 현재 우선순위는 합리적이다.

```text
지금: interpretable observable feature + retrieval + review
다음: real-unlabeled sanity + expert feedback
그 후: scratch/local 보강용 segmentation 또는 self-supervised embedding
```

## 다음 작업 게이트

다음 변경은 작고 검증 가능해야 한다.

- real-unlabeled smoke test는 계속 유지한다.
- 실패 경로에서도 output path 제한과 sanity JSON 생성이 유지되어야 한다.
- 실제 `.npz` 입력 1회 검증 전에는 loader를 과하게 framework화하지 않는다.
- reviewer가 채운 CSV가 생기면 summary metric과 `next_action_queue`로 바로 backlog를 만든다.
