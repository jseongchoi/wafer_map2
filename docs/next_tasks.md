# Next Tasks

기준일: 2026-06-17

내일 이어서 진행할 때의 목표는 새 모델을 붙이는 것이 아니라, 지금까지 검증한 synthetic scale pilot을 실제 업무 흐름으로 연결하는 것이다.

## 현재 위치

현재 프로젝트는 Phase 2 후반이다.

```text
Synthetic FBM 생성
-> observable feature 추출
-> 유사 wafer 검색
-> coarse grouping
-> stability / sweep / ablation
-> retrieval confidence 검증
-> defect feature target retrieval 검증
```

현재 판정:

- synthetic scale pilot 기준 유사 wafer 검색은 유망하다.
- observable feature는 morphology 보강 후 50개다.
- top-5 retrieval lift는 scale 155장에서 약 1.36x, holdout 120장에서 약 1.40x다.
- bootstrap 95% CI는 scale 기준 약 1.31x ~ 1.41x, holdout 기준 약 1.32x ~ 1.48x다.
- permutation p-value는 약 0.001이다.
- shot, edge, stby 계열은 triage 후보로 유효하다.
- local은 connected-component morphology로 일부 개선됐다.
- scratch는 여전히 불안정하므로 segmentation 또는 scratch-specific representation으로 넘긴다.
- structured defect feature target 기준으로도 검색 신호는 확인됐다.
- class target은 안정적이고, feature_key target은 random 대비 lift는 있으나 P@K가 낮아 expert review와 spatial 보강이 필요하다.
- chip-level polar spatial feature는 global similarity에는 잡음이지만, 위치-aware `class_location`/`feature_key` retrieval에는 도움이 된다.
- 현재 정책은 global grouping은 compact 50개 feature, 위치-aware retrieval은 polar feature를 조건부 사용이다.
- resize benchmark 결과, naive/semantic resize-only representation은 global retrieval에서 compact feature보다 크게 약하다.
- resize는 retrieval 대체재가 아니라 model input, candidate proposal, high-res patch crop 용도로 유지한다.
- patch/curve proposal은 보조 진단층으로 게이트를 닫고, edge/local/stby 및 ring/center arc review 후보 생성 용도로만 유지한다.
- scratch는 proposal/rule 튜닝에 더 과투자하지 않고 segmentation 또는 scratch-specific line representation으로 분리한다.
- real-unlabeled global nearest-neighbor 경로에서 `polar_*`, `stby_polar_*` 위치 feature가 섞이지 않도록 수정했다.
- real wafer 성능은 아직 검증되지 않았다.

## 다음 1순위: Real-Unlabeled Workflow

상태: MVP 구현 완료. `scripts/extract_real_unlabeled_features.py`와 `configs/eval/real_unlabeled_synthetic_smoke.json`로 synthetic smoke test가 가능하다. 다음 보강은 실제 보안 경로에서 semantic `.npz` manifest를 받아보는 것이다. 이번 점검 이후에는 proposal 실험보다 이 항목을 우선한다.

목표:

실제 wafer 데이터를 repo에 저장하지 않고도, 보안 환경 안에서 feature 추출과 유사맵 검색을 실행할 수 있는 workflow를 만든다.

할 일:

1. Real FBM 입력 계약 정의
   - 실제 array format 후보 정리
   - Grade 0~7, none-wafer, stby, valid-test mask 매핑 규칙 정의
   - 제품별 chip block width/height, die grid metadata 입력 방식 정의

2. Real sample loader 초안 작성
   - raw data를 repo에 저장하지 않는 CLI 구조 설계
   - local secure path를 입력으로 받되 output은 feature CSV/report만 생성
   - synthetic oracle field가 섞이지 않도록 observable-only 계약 유지

3. Real feature sanity check 작성
   - NaN/inf 없음
   - feature range 폭주 없음
   - stby와 Grade 7 분리 확인
   - none-wafer와 in-wafer Grade 0 분리 확인

4. Real nearest-neighbor review report 설계
   - query wafer별 top-k synthetic/real neighbor list
   - defect score summary
   - 전문가 검토용 gallery 또는 HTML table
   - 실제 이미지는 repo에 저장하지 않는 옵션 유지

5. Aggregate drift/sanity summary 추가
   - query feature와 reference feature의 평균/표준편차 차이 확인
   - stby/edge/fail-density 주요 feature가 synthetic reference 범위 밖으로 튀는지 표시
   - raw path나 민감정보 없이 JSON/HTML에 요약
   - 상태: 구현 및 synthetic smoke 확인 완료

완료 기준:

- real data 없이도 CLI와 schema가 명확히 정의되어 있다.
- synthetic sample을 real-like input으로 넣어 workflow가 end-to-end로 돈다.
- output에는 observable feature, nearest-neighbor 결과, sanity status만 남는다.

## 내일 2순위: Expert Review Protocol

상태: MVP 구현 완료. `scripts/make_expert_review_template.py`로 nearest-neighbor CSV를 reviewer 입력용 CSV/HTML로 바꾸고, `scripts/summarize_expert_review.py`로 사람이 채운 리뷰 결과를 집계할 수 있다.

목표:

유사맵 검색 결과를 사람이 평가할 수 있는 최소 형식을 만든다.

할 일:

1. Review label schema 정의
   - same family
   - partial match
   - mismatch
   - missed major defect
   - clock position match

2. Review JSON/CSV template 작성
   - sample_id
   - neighbor_id
   - reviewer_decision
   - dominant_defect
   - comment

3. Aggregate metric 정의
   - top-k same-family rate
   - partial-match rate
   - missed-major-defect rate
   - class별 review pass rate

완료 기준:

- 사용자가 real/synthetic gallery를 보고 빠르게 평가할 수 있다.
- 평가 결과가 다음 feature tuning에 바로 연결된다.

현재 산출물:

- `docs/expert_review_protocol.md`
- `outputs/reports/expert_review_template.csv`
- `outputs/reports/expert_review_template.html`
- `outputs/reports/expert_review_summary.html`
- `outputs/reports/expert_review_summary_metrics.json`
- `outputs/reports/fbm_interest_retrieval_scale_report.html`

다음 보강:

- 관심 defect 기준별 retrieval 결과를 expert review template과 연결한다.
- 보안 환경 내부 gallery id와 review template을 연결한다.
- 다중 리뷰어가 생기면 reviewer agreement와 adjudication queue를 추가한다.
- 리뷰 완료 후 defect family별 실패 사례를 scratch/local morphology 또는 segmentation 보강으로 연결한다.

## 내일 3순위: Robustness Holdout 유지 관리

상태: 구현 및 120장 실행 완료. `configs/synth/presets/fbm_grouping_holdout_stress.json`로 holdout stress를 생성했고, 전체 retrieval lift는 약 1.40x로 1.10x 기준을 넘었다.

목표:

지금의 155장 scale pilot이 generator parameter에 과적합된 결과인지 확인할 준비를 한다.

할 일:

1. Holdout synthetic config 작성
   - 다른 seed
   - 다른 class prior
   - 다른 grade threshold
   - shot intensity 약화
   - edge-light / center-like negative case 포함

2. Stress test 항목 정의
   - shot anchor/layout 변경
   - stby random vs origin-coupled 비율 변경
   - local droplet intensity 약화
   - ring vs spin scratch 혼동 case 강화

3. Holdout metric
   - retrieval lift
   - bootstrap CI
   - permutation p-value
   - class별 precision@k lift

완료 기준:

- holdout batch에서도 top-k lift가 1.10x 이상인지 확인할 수 있다.
- shot/local/stby prior 변화에 대한 취약점을 확인할 수 있다.

현재 판정:

- edge/stby/ring은 holdout에서도 유지된다.
- local은 morphology 보강 후 holdout 관심 검색에서 개선됐다.
- shot은 약한 prior/조건 변화에서 취약하다.
- scratch는 계속 약하므로 다음 단계에서 segmentation 또는 curve/arc-specific detector로 넘긴다.

## 내일 4순위: Scratch/Local 보강 방향 결정

목표:

scratch/local을 wafer-level retrieval feature로 계속 보강할지, segmentation baseline으로 넘길지 결정한다.

현재 판단:

- local은 morphology 보강 후 holdout 관심 검색에서 약 1.33x까지 개선됐지만, scale에서는 약 1.11x라 아직 보수적으로 본다.
- scratch는 scale에서 약 1.10x, holdout에서 약 0.92x로 불안정하다.
- 따라서 local은 CPU morphology를 유지하면서 expert review로 확인하고, scratch는 segmentation 또는 scratch-specific representation으로 넘긴다.

Segmentation readiness 중간점검:

- scale 155장과 holdout 120장에 대해 segmentation manifest/report를 생성했다.
- scratch는 면적이 작고 ring/local/stby와 자주 중첩된다.
- scale 기준 scratch + stby co-occurrence는 약 0.83, scratch + local co-occurrence는 약 0.70이다.
- holdout 기준도 scratch + stby 약 0.68, scratch + local 약 0.55로 유지된다.
- 따라서 scratch는 feature를 계속 덧대는 것보다 multi-label segmentation 또는 scratch-specific representation으로 넘긴다.

Segmentation smoke training:

- `scripts/train_segmentation_smoke.py`로 NumPy-only 1x1 sigmoid baseline을 실행했다.
- scale과 holdout 모두 weighted BCE loss가 감소해 manifest, tensor, target, loss 배관은 정상이다.
- 다만 1x1 baseline은 공간 context가 없어 scratch/local을 잡지 못한다.
- 다음 게이트는 PyTorch/GPU 환경의 small U-Net 또는 lightweight SegFormer smoke training이다.

Defect feature extraction summary:

- synthetic oracle mask를 사용해 structured defect feature extraction 후처리를 검증했다.
- mask가 주어지면 `class_name`, `feature_key`, `pixel_ratio`, `clock/radial location`, `stby_overlap_ratio` 같은 downstream feature row가 생성된다.
- 따라서 유사맵 검색, 관심 불량별 retrieval, defect score, 이후 공정 metadata 조인에 쓸 feature table 방향은 타당하다.
- feature-key retrieval은 scale에서 약 3.13x, holdout에서 약 1.97x lift를 보여 위치-aware 검색 신호가 있다.
- 남은 핵심 난제는 mask prediction이며, 특히 scratch/local/stby overlap을 spatial model이 얼마나 복원하는지가 다음 성패를 가른다.

후보:

1. Lightweight morphology 보강
   - connected component count
   - thin arc score
   - local blob compactness
   - multi-blob triangle score

2. Synthetic-label segmentation baseline
   - 작은 U-Net
   - low-resolution tile input
   - class별 sigmoid mask
   - scratch/local/stby-hidden origin recall 중심

완료 기준:

- local은 GPU 없이 morphology baseline으로 얼마나 갈 수 있는지 expert review에서 확인한다.
- scratch는 GPU 또는 foundation feature 기반 실험 후보로 올린다.

## 하지 말아야 할 것

- real wafer raw image/array를 repo에 저장하지 않는다.
- ANOVA를 지금 핵심 목표로 당기지 않는다.
- AutoEncoder를 1차 해결책으로 되돌리지 않는다.
- synthetic label `*_mask_ratio`를 inference feature에 섞지 않는다.
- 전체 retrieval lift만 보고 scratch/local까지 잘 된다고 말하지 않는다.
- K=3/K=4 cluster를 공정 원인 label처럼 해석하지 않는다.

## 내일 시작 명령

현재 검증 상태 확인:

```powershell
python -m pytest -q
python scripts/evaluate_retrieval_confidence.py --features outputs/reports/fbm_grouping_scale_features.csv --out outputs/reports/fbm_retrieval_confidence_scale_report.html --metrics outputs/reports/fbm_retrieval_confidence_scale_metrics.json
```

주요 문서 확인:

```text
docs/ai_ds_validation_audit.md
docs/scale_pilot_review.md
docs/modeling_strategy.md
docs/validation_protocol.md
```

## 내일 첫 작업 제안

첫 작업은 `scripts/extract_real_unlabeled_features.py` 초안을 만드는 것이다.

이 스크립트는 실제 데이터 loader가 확정되기 전이라도 다음 구조로 시작할 수 있다.

```text
input folder or manifest
-> semantic FBM tensor parser
-> observable feature extraction
-> sanity check
-> feature CSV
-> optional nearest-neighbor report
```

이렇게 시작하면 보안 데이터가 없어도 synthetic sample을 adapter로 넣어 end-to-end 형태를 먼저 검증할 수 있다.
