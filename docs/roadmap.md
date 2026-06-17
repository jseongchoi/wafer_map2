# Roadmap

## Phase 0: Documentation And Agreement

목표:

- 문제 정의 확정
- data schema 확정
- synthetic generator 요구사항 확정
- validation protocol 확정

완료 조건:

```text
docs/problem_definition.md
docs/data_schema.md
docs/pattern_taxonomy.md
docs/synthetic_data_plan.md
docs/validation_protocol.md
docs/modeling_strategy.md
```

## Phase 1: Synthetic Generator MVP

목표:

- 약 600 net die wafer geometry 생성
- 제품별 chip block size config 지원
- Grade 0~7 quantization 지원
- scratch, ring, edge, local, random, shot_grid, stby_pattern 생성
- preview image와 synthetic validation mask 저장

완료 조건:

```text
review batch 생성
preview grid 생성
internal validation 통과
expert review feedback 반영
```

## Phase 2: FBM Information Extraction And Grouping

목표:

- wafer-level observable feature table 생성
- radial/angular/edge/shot/stby/local morphology feature 계산
- 유사 wafer nearest-neighbor 검색
- K=3/4 기준 coarse grouping
- feature family ablation
- 전문가 육안 확인용 neighbor gallery 생성

완료 조건:

```text
synthetic pilot batch feature table 생성
FBM grouping report 생성
grouping stability report 생성
parameter sweep report 생성
feature ablation report 생성
nearest-neighbor gallery 생성
```

현재 상태:

```text
155장 scale pilot top-5 유사검색 lift 약 1.36x
120장 holdout stress top-5 유사검색 lift 약 1.40x
observable feature 50개
edge/shot/ring/stby feature family가 retrieval에 기여함
local은 morphology 보강 후 일부 개선됨
scratch는 segmentation 또는 scratch-specific representation 보강 대상
```

초기 36장 pilot에서는 top-5 lift 약 1.46x와 K=4가 관측됐다. 현재 roadmap의 운영 기준은 155장 scale pilot이다.

추가 판정:

```text
patch proposal:
  edge/local/stby review 후보를 줄이는 보조 도구로 유지

curve proposal:
  ring/center arc review 후보를 줄이는 보조 도구로 유지

scratch:
  rule/proposal 과투자를 멈추고 segmentation 또는 scratch-specific line representation으로 분리
```

따라서 Phase 2의 proposal 계열 실험은 현재 게이트를 닫고, Phase 4 real-unlabeled와 expert review 연결을 우선한다.

## Phase 3: Synthetic-Label Segmentation Baseline

목표:

- multi-label segmentation dataset 구성
- U-Net 계열 baseline 학습
- defect heatmap과 report 생성
- 작은 local defect와 중첩 defect recall 측정

완료 조건:

```text
per-class synthetic IoU/Dice 산출
small overlap defect recall 측정
시계 방향 defect summary 생성
```

## Phase 4: Secure Real-Unlabeled Adaptation

목표:

- 실제 wafer를 repo 밖 보안 환경에서만 처리
- real FBM feature extraction 실행
- synthetic/real aggregate feature sanity check
- nearest-neighbor gallery에 대한 expert review 수집

완료 조건:

```text
real data를 저장하지 않고 feature만 추출 가능
feature NaN/폭주 없음
전문가가 유사맵 검색 결과를 검토 가능
synthetic generator parameter calibration feedback 반영
```

현재 상태:

```text
scripts/extract_real_unlabeled_features.py MVP 구현 완료
synthetic smoke manifest 실행 통과
global nearest-neighbor feature 계약에서 polar 위치 feature 제외하도록 수정
```

다음 보강:

- 실제 보안 환경에서 export 가능한 semantic `.npz` 입력 계약 확정
- reference synthetic feature와 query feature의 aggregate drift/sanity summary 추가
- nearest-neighbor 결과를 expert review template과 연결

## Phase 5: Process Metadata Statistics

목표:

- 공정/설비/lot/recipe/chamber/test metadata와 FBM feature 조인
- defect score별 통계 검정
- tool/chamber/recipe factor effect 확인

완료 조건:

```text
process metadata가 붙은 feature table 생성
ANOVA 또는 적절한 통계 검정 수행
공정 원인 후보를 expert review로 검토
```

## Phase 6: Advanced Modeling

후보:

- DINOv2-style self-supervised encoder
- patch-level anomaly detection
- synthetic-to-real domain adaptation
- weakly-supervised report generation
- interactive annotation/refinement workflow

주의:

Advanced model은 synthetic realism, baseline feature usefulness, secure real sanity check가 확인된 뒤 도입한다.
