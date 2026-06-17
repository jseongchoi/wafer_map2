# Current Milestone Checkpoint

기준일: 2026-06-17

## 원래 목표 재확인

이 프로젝트의 본질은 1채널 고해상도 Fail Bit Map에서 실제 wafer에도 계산 가능한 정보를 추출하는 것이다.

현재 최우선 목표는 다음이다.

```text
FBM observable feature extraction
-> 유사 wafer 검색 / 그룹핑
-> 관심 불량별 검색
-> wafer 내부 defect feature 수치화
-> real-unlabeled workflow와 expert review로 synthetic-to-real gap 확인
```

ANOVA는 지금 단계의 목표가 아니다. 공정/설비/lot/recipe metadata가 붙은 뒤에 FBM feature table을 조인해 수행할 후속 분석이다.

## 현재 이정표 위치

현재 위치는 Phase 2 후반에서 Phase 3 초입이다.

완료된 흐름:

```text
문제 정의
-> synthetic FBM 생성
-> 현실성 피드백 반영
-> observable feature 추출
-> 유사 wafer retrieval / grouping
-> scale + holdout 검증
-> 관심 불량별 retrieval
-> structured defect feature target retrieval
-> resize benchmark
-> patch proposal readiness
-> curve proposal readiness
```

아직 완료되지 않은 흐름:

```text
real wafer secure-path adapter
-> real-unlabeled sanity check
-> expert review 반복
-> scratch/shot/ring 보강
-> GPU segmentation 또는 metric-learning 검증
```

## 지금까지 잘 된 것

- Synthetic generator가 Grade 0~7, none-wafer, valid-test, stby fail chip을 분리해서 표현한다.
- 웨이퍼 edge 계단형 geometry, center-to-edge fail gradient, edge-chip outer-face gradient가 반영되어 있다.
- spin/radial scratch, ring, edge, local blob, shot-relative, stby-origin-coupled defect를 생성한다.
- Global 유사맵 검색은 compact observable feature 50개 기준으로 scale/holdout 모두 random baseline 대비 유의한 lift가 있다.
- 위치-aware defect feature retrieval은 polar spatial feature를 조건부로 사용할 때 신호가 있다.
- Synthetic oracle label과 mask는 검증/채점 전용으로 분리되어 있고, 실제 inference feature에는 넣지 않는다.
- Resize-only retrieval은 명확히 탈락시켰고, resize는 model input/proposal/patch crop 용도로 재정의했다.
- Patch proposal 실험으로 edge/local/stby 계열은 저해상도 score map만으로도 원본 후보 영역을 꽤 잘 잡는다는 것을 확인했다.
- Curve proposal 실험으로 ring/center arc 계열은 polar representation이 유효하다는 것을 확인했다.
- Scratch는 patch proposal과 curve proposal 모두에서 약해 segmentation 또는 scratch-specific line representation으로 넘겨야 한다는 경계가 더 명확해졌다.

## 핵심 수치

Global retrieval confidence:

| Dataset | Samples | Observable features | Lift | 95% CI | Permutation p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scale | 155 | 50 | 1.364x | 1.313x ~ 1.412x | 0.000999 |
| Holdout | 120 | 50 | 1.404x | 1.319x ~ 1.482x | 0.000999 |

Structured defect target retrieval:

| Dataset | Target kind | Mean lift | Precision@K | Hit@K |
| --- | --- | ---: | ---: | ---: |
| Scale | class | 1.414x | 0.824 | 0.983 |
| Scale | class_radial | 1.490x | 0.588 | 0.898 |
| Scale | class_location | 2.495x | 0.299 | 0.599 |
| Scale | feature_key | 3.129x | 0.340 | 0.649 |
| Holdout | class | 1.472x | 0.743 | 0.966 |
| Holdout | class_radial | 1.591x | 0.507 | 0.888 |
| Holdout | class_location | 1.856x | 0.178 | 0.500 |
| Holdout | feature_key | 1.965x | 0.191 | 0.485 |

Resize benchmark:

| Dataset | Compact feature | Best resize-only |
| --- | ---: | ---: |
| Scale | 1.364x | semantic_pool_32: 0.553x |
| Holdout | 1.404x | naive_gray_32: 0.486x |

Patch proposal readiness:

| Dataset | Class | Proposal recall | Random recall | Recall lift | Hit@30% |
| --- | --- | ---: | ---: | ---: | ---: |
| Scale | edge | 0.867 | 0.104 | 8.38x | 0.988 |
| Scale | local | 0.902 | 0.146 | 6.16x | 0.912 |
| Scale | stby_pattern | 0.874 | 0.139 | 6.30x | 1.000 |
| Scale | scratch | 0.144 | 0.158 | 0.91x | 0.217 |
| Scale | ring | 0.151 | 0.132 | 1.15x | 0.150 |
| Scale | shot_grid | 0.197 | 0.137 | 1.44x | 0.120 |
| Holdout | edge | 0.913 | 0.099 | 9.26x | 1.000 |
| Holdout | local | 0.882 | 0.143 | 6.18x | 0.892 |
| Holdout | stby_pattern | 0.735 | 0.128 | 5.74x | 1.000 |
| Holdout | scratch | 0.174 | 0.157 | 1.11x | 0.256 |
| Holdout | ring | 0.189 | 0.149 | 1.27x | 0.268 |
| Holdout | shot_grid | 0.152 | 0.149 | 1.02x | 0.103 |

Curve proposal readiness:

| Dataset | Class | Proposal recall | Random recall | Recall lift | Hit@30% | Hit@50% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scale | ring | 0.315 | 0.222 | 1.42x | 0.517 | 0.233 |
| Scale | scratch | 0.010 | 0.081 | 0.13x | 0.000 | 0.000 |
| Holdout | ring | 0.434 | 0.287 | 1.51x | 0.696 | 0.464 |
| Holdout | scratch | 0.038 | 0.174 | 0.22x | 0.051 | 0.026 |

해석:

- Edge/local/stby는 현재 proposal 방식으로도 실무 review 후보를 줄이는 데 쓸 수 있다.
- Scratch는 아직 해결되지 않았다. 특히 thin arc/원형 scratch는 hotspot window 방식으로는 불안정하다.
- Shot은 wafer-level retrieval에서는 신호가 있지만, 단일 hotspot proposal과는 잘 맞지 않는다. shot은 반복 위치/레티클 상대좌표 기반 feature로 유지하는 편이 맞다.
- Ring은 넓고 얇은 구조라 window recall 지표만으로 평가하면 과소평가된다. 별도 polar curve proposal에서는 scale/holdout 모두 유효한 신호가 확인됐다.

## 크로스체크 결과

확인한 계약:

- Global retrieval 입력 feature는 50개다.
- `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 global inference feature에 들어가지 않는다.
- CSV에는 검증용 `label_*`, 위치-aware용 `polar_*`, 결과 해석용 `cluster_id`, `pca_0`, `pca_1`가 함께 저장될 수 있다.
- 따라서 저장 CSV 컬럼 수와 실제 retrieval 입력 feature 수는 다르다.
- `actual_net_die`는 metadata이며 retrieval 입력 feature에서 제외된다.
- PowerShell 기본 `Get-Content`로 한글이 깨져 보일 수 있으나 파일은 UTF-8로 정상 저장되어 있다. 확인 시 `Get-Content -Encoding UTF8`을 쓴다.

코드 정리:

- Patch proposal에서 resize benchmark와 공통으로 쓰는 grid/integral pooling 유틸을 `src/wafermap/features/spatial_pool.py`로 분리했다.
- 아직 `standardize`, `nearest_neighbors`, report HTML helper 같은 script-level 중복은 남아 있다.
- 다만 검증 script가 계속 변하는 중이라 대규모 공통 abstraction은 아직 미루는 것이 낫다.

## 놓치면 안 되는 리스크

- Real wafer 성능은 아직 검증되지 않았다.
- Synthetic generator와 feature extractor가 같은 도메인 가정을 공유하므로, 일부 성능은 synthetic 과적합일 수 있다.
- Scratch는 현재 feature/retrieval/proposal 어디에서도 안정적으로 해결되지 않았다.
- Shot defect는 hotspot이 아니라 reticle-relative 반복성으로 봐야 한다.
- Defect score는 calibrated probability가 아니라 feature proxy다.
- Stby fail chip은 의미 있는 패턴이면서 동시에 내부 fail bit 관측을 가리는 censoring이다.
- Synthetic oracle mask는 실제 운영에서 model-predicted mask 또는 expert-reviewed target으로 대체되어야 한다.

## 다음 액션

1. Real-unlabeled workflow를 실제 보안 경로 입력 계약에 맞춘다.
   - repo에는 raw wafer를 저장하지 않는다.
   - output은 feature CSV, sanity JSON, review report만 남긴다.

2. Expert review template을 현재 retrieval/proposal 결과와 연결한다.
   - 유사맵이 같은 계열인지
   - proposal box가 실제 주목 영역을 줄여주는지
   - scratch/ring/shot 실패 케이스가 어떤 모양인지 기록한다.

3. Scratch-specific representation을 별도 이정표로 분리한다.
   - arc/ring/scratch curve detector
   - polar Hough/Radon-like projection
   - GPU 가능 시 small segmentation model

4. Patch proposal은 edge/local/stby triage용으로 유지한다.
   - 속도 개선이 필요하면 chip-level aggregation 또는 cached semantic map을 추가한다.
   - shot/ring/scratch를 proposal recall 하나로 평가하지 않는다.

5. Curve proposal은 ring/center arc triage용으로 유지한다.
   - scratch에는 현재 방식이 충분하지 않으므로 성공으로 해석하지 않는다.
   - scratch 실패 케이스는 line enhancement/segmentation 실험으로 넘긴다.

6. Full test와 report 재생성 계약을 유지한다.
   - 핵심 코드 변경 후 `python -m pytest -q`
   - output HTML은 regenerate 가능한 산출물로 유지한다.

## 현재 판정

본질대로 진행되고 있다. 단, 종착지는 룰 기반 proposal이나 synthetic label 점수 그 자체가 아니다.

현재 이정표의 의미는 다음이다.

```text
FBM에서 observable feature를 뽑아 downstream task가 가능하다는 최소 근거를 만들었다.
어떤 결함군은 현재 방식으로 충분히 유망하고,
어떤 결함군은 segmentation/curve representation으로 넘겨야 한다는 경계도 드러났다.
```

따라서 다음 단계는 더 많은 모델을 무작정 붙이는 것이 아니라, real-unlabeled 입력 경로와 expert review loop를 통해 synthetic-to-real gap을 줄이는 것이다.
