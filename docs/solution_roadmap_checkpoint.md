# Solution Roadmap Checkpoint

기준일: 2026-06-17

## 판단 목적

최근 patch/curve proposal 실험이 늘어나면서, 프로젝트가 원래 목표에서 벗어나고 있는지 점검했다.

결론부터 말하면:

```text
본질은 유지되고 있다.
다만 proposal 실험은 여기서 게이트를 닫고,
이제 real-unlabeled workflow와 expert review loop로 중심을 옮겨야 한다.
```

## 최초 제안 솔루션 중 현재 수행한 것

초기 솔루션은 다음 계층 구조였다.

```text
1. Synthetic FBM 생성과 현실성 검증
2. Observable feature extraction
3. 유사 wafer retrieval / grouping
4. 관심 defect별 retrieval / defect score
5. Mask가 있을 때 structured defect feature extraction
6. Real-unlabeled secure workflow
7. Expert review feedback loop
8. Segmentation / self-supervised embedding / domain adaptation
9. Process metadata statistics
```

현재 완료 또는 검증된 항목:

- 1번 synthetic generator: 완료, 계속 보정 가능
- 2번 observable feature: 완료, compact global feature 50개 기준 확립
- 3번 유사맵 retrieval/grouping: scale/holdout에서 유효성 확인
- 4번 관심 defect retrieval: class/class_location/feature_key 기준 유효성 확인
- 5번 structured defect feature extraction: synthetic oracle mask 기준 후처리 검증 완료
- 6번 real-unlabeled workflow: MVP 구현 완료, smoke test 통과

현재 진행해야 하는 항목:

- 7번 expert review feedback loop
- 6번 real-unlabeled workflow의 실제 입력 계약 검증

아직 당기면 안 되는 항목:

- 8번 GPU segmentation/self-supervised embedding
- 9번 process metadata statistics/ANOVA

## 과투자 위험 판단

Patch proposal과 curve proposal은 쓸모가 있었다.

얻은 결론:

- edge/local/stby는 patch proposal로 후보 영역을 줄일 수 있다.
- ring/center arc는 polar curve proposal이 맞다.
- scratch는 rule/CV proposal로 해결되지 않는다.
- shot은 hotspot proposal보다 reticle-relative feature로 봐야 한다.

하지만 proposal은 최종 솔루션의 중심이 아니다.

Proposal의 역할:

```text
review 후보를 줄이는 보조 도구
segmentation/model이 뭘 놓치는지 보는 diagnostic layer
feature extraction 전략을 검증하는 작은 실험
```

Proposal에 더 투자하면 위험한 부분:

- scratch를 rule로 계속 억지로 잡으려는 것
- proposal recall을 최종 성능처럼 해석하는 것
- synthetic oracle recall 튜닝에만 집중하는 것
- real-unlabeled 입력과 expert review를 미루는 것

따라서 proposal 실험은 현재 기준으로 게이트를 닫는다.

## 지금의 핵심 판정

현재 솔루션의 주 경로:

```text
FBM semantic tensor
-> compact observable feature
-> global similar wafer retrieval
-> interest-conditioned retrieval
-> defect score / structured feature table
-> expert review
-> real-unlabeled 적용
```

현재 보조 경로:

```text
patch proposal:
  edge/local/stby 후보 영역 review

curve proposal:
  ring/center arc 후보 영역 review

segmentation:
  scratch/local/stby overlap 보강 후보
```

## 이번 점검에서 발견해 수정한 것

Real-unlabeled nearest-neighbor 경로에서 global 검색에 `polar_*`, `stby_polar_*` feature가 섞일 수 있는 누락이 있었다.

정책:

```text
global retrieval:
  compact observable feature만 사용

location-aware retrieval:
  polar feature를 조건부 사용
```

수정:

- `scripts/extract_real_unlabeled_features.py`의 global observable feature selection에서 `polar_*`, `stby_polar_*`를 제외했다.
- `tests/test_real_unlabeled_workflow.py`에 회귀 테스트를 추가했다.
- synthetic smoke manifest로 real-unlabeled workflow를 재실행해 정상 출력을 확인했다.
- reference synthetic feature와 query feature의 aggregate drift summary를 sanity JSON/HTML에 추가했다.

## 다음 로드맵

### Step A. Real-Unlabeled Contract 확정

목표:

실제 wafer raw 데이터를 repo에 넣지 않고도 semantic tensor manifest를 받아 feature extraction이 가능해야 한다.

다음 작업:

- 실제 보안 환경에서 만들 수 있는 `.npz` key 후보 확정
- Grade 0~7, none-wafer, stby, valid-test mask mapping 확정
- 제품별 chip size/grid metadata 입력 규칙 확정
- feature sanity JSON을 실제 검토 가능한 수준으로 강화

완료 기준:

- 사용자가 실제 데이터 없이도 manifest 예시를 보고 입력 계약을 검토할 수 있다.
- synthetic smoke가 계속 통과한다.
- output에는 raw path/민감정보가 남지 않는다.

### Step B. Expert Review Loop 연결

목표:

자동 metric이 아니라 사용자의 현업 판단으로 retrieval/proposal 품질을 검증한다.

다음 작업:

- nearest-neighbor 결과를 review template과 연결
- query wafer별 관심 기준을 기록
- same family / partial / mismatch / missed major defect를 수집
- 실패 유형을 scratch, shot, ring, local, stby-hidden origin으로 분류

완료 기준:

- 사용자가 HTML/CSV를 보고 빠르게 판정할 수 있다.
- 리뷰 결과가 다음 feature/model 보강으로 직접 이어진다.

### Step C. Scratch는 별도 Track으로 격리

목표:

scratch를 현재 proposal/rule feature에 계속 과투자하지 않는다.

다음 후보:

- line enhancement
- skeleton/curve continuity
- small U-Net / lightweight segmentation
- GPU 환경에서 synthetic-label training

완료 기준:

- scratch recall을 global retrieval lift와 분리해서 평가한다.
- GPU가 필요한 단계는 사용자가 실행 가능한 환경에서만 본격 진행한다.

### Step D. Process Metadata는 나중

목표:

FBM feature table이 안정화된 뒤 공정 metadata와 조인한다.

지금 하지 않는 것:

- ANOVA를 지금 핵심 목표로 당기지 않는다.
- tool/chamber/recipe 효과를 synthetic 데이터만으로 해석하지 않는다.

## 당장 다음 작업

다음으로는 `real-unlabeled` workflow를 “실제 사용자가 검토 가능한 입력 계약 + expert review 연결” 수준으로 올리는 것이 맞다.

구체적으로:

```text
manifest/schema 문서 강화
-> synthetic smoke 유지
-> expert review template 연결
```

이 작업이 끝나면 실제 데이터가 없어도 사용자가 보안 환경에서 어떤 형태로 export해야 하는지 판단할 수 있다.
