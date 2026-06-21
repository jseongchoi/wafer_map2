# 로드맵

## 현재 위치

현재 프로젝트는 합성 데이터 검증을 지나 **라벨 없는 실제 wafer 적용 준비 단계**에 있다.

```text
완료/검증됨:
합성 FBM generator
-> 실제 데이터용 feature 추출
-> 전체 유사 wafer 검색
-> 관심 불량별 검색
-> holdout 검증
-> 라벨 없는 실제 wafer 처리 절차
-> 전문가 리뷰 양식

현재 주 경로:
보안 환경의 제품별 raw PNG 폴더
-> 제품별 geometry 추론 / manifest 생성
-> 기본 검사 / feature 리포트
-> 전문가 리뷰

대체 경로:
보안 환경의 실제 wafer semantic .npz
-> 기본 검사 / drift 리포트
-> nearest-neighbor 검색
-> 전문가 리뷰
-> feature 보강 또는 AI 모델 후보 정리
```

## 목표 단계 이정표

현재 구현과 검증은 아래 8단계로 관리한다. `scripts/audit_project_readiness.py`는 이 순서대로 구현물, 테스트 근거, 실제 데이터 전 준비 산출물을 점검한다.

| 단계 | 목표 | 현재 판정 기준 |
| --- | --- | --- |
| 0. 문제와 스키마 기준 | wafer map 의미, gray value, 보안 경계, manifest 기준을 고정한다. | 문서와 `src/wafermap/real/manifest.py`가 존재하고 manifest 테스트가 통과한다. |
| 1. 합성 데이터 | 실제 전까지 반복 가능한 raw grayscale PNG와 oracle 검증 데이터를 만든다. | generator/validator/render 테스트가 통과한다. |
| 2. raw PNG 읽기 | 제품별 폴더의 정해진 gray PNG를 읽고 geometry/stby를 검증한다. | `png_grayscale_raw`, 제품별 batch, STBY 모호성 테스트가 통과한다. |
| 3. 관측 가능한 feature | 실제 wafer에서도 계산 가능한 feature와 nearest-neighbor 산출물을 만든다. | oracle 정보 누출 없이 feature CSV, sanity JSON, neighbor CSV가 생성된다. |
| 4. 전문가 리뷰 흐름 | neighbor와 feature를 리뷰 양식, 실패 유형, 다음 작업으로 연결한다. | 전문가 리뷰 양식과 요약 테스트가 통과한다. |
| 5. CPU AI 기준선 | CPU에서 끝까지 재현 가능한 embedding/classification 기준선을 만든다. | NumPy CPU encoder train/score 테스트와 준비 기준이 통과한다. |
| 6. 실제 데이터 전 준비 | 합성 데이터로 전체 흐름과 raw PNG batch 점검을 한 번에 검증한다. | `run_pre_real_readiness.py` 요약에 synthetic PNG batch 산출물이 포함되고 PASS한다. |
| 7. 실제 실행 전 최종 점검 | 실제 raw PNG가 오기 전에 batch 명령, private manifest, PNG 읽기, report, CPU scoring 경로를 synthetic PNG로 검증한다. | `run_pre_real_readiness.py` 요약에 synthetic PNG batch 산출물이 포함되고 PASS한다. |
| 8. 실제 데이터 batch | 사용자가 보안 폴더에 넣은 실제 raw PNG를 분석하고 공유 가능한 파생 리포트를 만든다. | `outputs/private` manifest, 오류 없는 sanity, 운영 `batch_metadata.json`, feature/report/neighbor/review CSV가 생성된다. 실제 데이터 전에는 PENDING이 정상이다. |

## Phase 0. 문제 정의와 입력 기준

상태: 완료

목표:

- 문제 정의
- FBM 분석용 배열 형식 정의
- 합성 generator 요구사항 정리
- 검증 방법 정리

핵심 문서:

- [프로젝트 개요](project_overview.md)
- [데이터 형식](data_schema.md)
- [검증 방법](validation_protocol.md)
- [불량 패턴 정리](pattern_taxonomy.md)

## Phase 1. 합성 FBM Generator

상태: 완료, 필요 시 보정

완료된 것:

- 약 600 net die급 wafer geometry
- Grade 0~7
- wafer 밖 영역, 실제 test 영역, stby 분리
- edge, local, shot-relative, stby-origin, ring, scratch 계열 생성
- 합성 데이터 oracle mask는 검증용으로만 사용

주의:

- 합성 데이터 성능은 실제 wafer 성능 인증이 아니다.
- Generator와 feature extractor가 같은 가정을 공유할 수 있으므로 실제 wafer 리뷰가 필요하다.

## Phase 2. Feature와 유사 Wafer 검색

상태: 완료, 기준선으로 유지

완료된 것:

- compact feature 50개
- scale 155장 top-k retrieval lift 약 1.36x
- holdout 120장 top-k retrieval lift 약 1.40x
- class/class_location/feature_key 기준 관심 불량 검색 신호 확인
- 단순 resize representation은 전체 유사 wafer 검색 대체재로 부적합하다고 판단

사용 원칙:

- 전체 유사 wafer 검색은 compact feature 50개 기준을 유지한다.
- `polar_*`, `stby_polar_*`는 위치가 중요한 검색에서만 조건부로 쓴다.
- `label_*`, `*_mask_ratio`, `pattern_masks`, `pattern_intensity`는 실제 inference feature가 아니다.

## Phase 3. Proposal과 Segmentation 준비

상태: 보조 도구로 정리 완료

현재 판단:

- Patch proposal: edge/local/stby 리뷰 후보를 줄이는 용도
- Curve proposal: ring/center arc 리뷰 후보를 줄이는 용도
- Scratch: rule/proposal 과투자를 멈추고 segmentation 또는 scratch 전용 line feature로 분리

주의:

- Proposal recall은 최종 성능 지표가 아니다.
- 지금은 proposal 튜닝보다 라벨 없는 실제 wafer 처리 절차와 전문가 리뷰가 우선이다.

## Phase 4. 라벨 없는 실제 Wafer 적용

상태: 현재 주 작업

목표:

- 실제 wafer raw data를 repo에 저장하지 않고 feature를 추출한다.
- 보안 환경의 제품별 raw PNG 폴더 또는 `.npz`와 manifest를 입력으로 받는다.
- feature CSV, sanity JSON, drift report, nearest-neighbor CSV, 전문가 리뷰 양식을 생성한다.

완료된 것:

- `scripts/analyze_png_raw_folders.py`
- `scripts/extract_real_unlabeled_features.py`
- `real_unlabeled_manifest/v1`
- `observable_fbm_features/v1`
- `png_grayscale_raw` 입력
- 실제 gray value 기준: `0, 31, 151, 175, 191, 207, 223, 255`
- full-255 chip 단위 stby 판정
- 제품 폴더별 chip geometry 추론
- `.npz` array validation
- reference 대비 feature drift summary
- nearest-neighbor CSV와 전문가 리뷰 양식 연결

다음 확인 단계:

1. 실제 보안 환경에서 제품별 raw PNG 폴더 1개를 batch script로 실행
2. sanity JSON에서 gray value, stby chip, chip geometry, wafer mask 추론이 맞는지 확인
3. stby가 없어 geometry 추론이 실패하는 제품은 `--geometry-json`으로 보완
4. 필요하면 `.npz_semantic_arrays` 경로도 1건 smoke로 확인
5. top-k nearest-neighbor 결과를 전문가가 최소 20~50 pair 평가
6. `next_action_queue`를 보고 feature 보강 또는 AI 모델 후보 결정

## Phase 5. Scratch/Local 보강

상태: 대기

시작 조건:

- 전문가 리뷰에서 scratch/local 계열 실패가 반복적으로 확인된다.
- 현재 feature 보강만으로 부족하다는 근거가 생긴다.

후보:

- connected-component morphology
- line enhancement / skeleton continuity
- lightweight multi-label segmentation
- self-supervised embedding 또는 metric learning

현재 착수한 AI 배관:

- `scripts/run_pre_real_readiness.py`로 실제 데이터 전 합성 데이터부터 실제 형식 처리까지의 전체 흐름을 한 번에 검증한다.
- `scripts/build_segmentation_readiness.py`로 synthetic segmentation manifest를 만든다.
- `scripts/train_embedding_smoke.py`로 synthetic tensor를 임베딩하고 label-aware nearest-neighbor Jaccard를 확인한다.
- `scripts/train_cpu_encoder_model.py`로 NumPy shared encoder multi-label 모델을 학습한다.
- `scripts/score_unlabeled_cpu_encoder.py`로 실제 PNG manifest에 class probability, embedding, synthetic neighbor를 붙인다.
- 이 단계는 CPU에서 끝까지 재현 가능한 기준선이다. 실사용 deep model 성능 주장은 아니며, 이후 PyTorch shared encoder / contrastive learning / segmentation 모델로 교체할 수 있게 입출력 기준을 먼저 고정한다.

실제 raw PNG가 들어오기 전 마지막 확인 기준:

1. `run_pre_real_readiness.py`가 PASS한다.
2. `outputs/pre_real_readiness/reports/pre_real_readiness_summary.json`에 모든 핵심 output path가 남는다.
3. `scripts/audit_project_readiness.py`에서 0~7단계는 PASS하고, 실제 PNG batch 산출물만 남은 8단계는 PENDING으로 표시된다.
4. summary의 `real_png_batch_command`에서 `<SECURE_RAW_PNG_ROOT>`만 실제 보안 폴더로 바꿔 실행할 수 있다.
