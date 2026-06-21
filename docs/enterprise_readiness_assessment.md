# 기업 도입 준비도 평가

이 문서는 현재 WaferMap 솔루션을 기업 판매 또는 파일럿 제안 관점에서 단계별로 판정한 결과다. 결론부터 말하면, 현재 상태는 **바로 판매 가능한 상태가 아니라 제한된 파일럿 직전**이다. 실제 raw PNG batch와 전문가 리뷰 결과가 들어오기 전까지 8단계는 의도적으로 `PENDING`이어야 한다.

## 현재 판정

감사 명령:

```powershell
python scripts\audit_project_readiness.py
```

현재 감사 결과:

| 구분 | 결과 |
| --- | --- |
| 전체 상태 | PENDING |
| PASS | 6 |
| CHECK | 2 |
| PENDING | 1 |

해석:

- 0~5단계는 구현, 테스트, 문서 기준으로 `PASS`다.
- 6~7단계는 `CHECK`다. 작은 합성 데이터 실행과 CPU 기준선은 방법 확인이지 성능 보증이 아니다.
- 8단계는 `PENDING`이다. 실제 raw PNG batch, 오류 없는 기본 검사, 리뷰 양식, 민감정보 없는 공유 리포트가 아직 없다.

## 단계별 기업 관점 판정

| 단계 | 현재 상태 | 기업 관점 판정 | 남은 조건 |
| --- | --- | --- | --- |
| 0. 문제와 스키마 기준 | PASS | 파일럿 가능 | schema 변경 시 문서와 테스트 동시 갱신 |
| 1. 합성 데이터 | PASS | 파일럿 가능 | 합성 데이터 결과를 실제 성능으로 주장하지 않기 |
| 2. raw PNG 읽기 | PASS | 파일럿 가능 | 제품별 geometry/mask를 실제 데이터로 검증 |
| 3. 관측 가능한 feature | PASS | 파일럿 가능 | 실제 wafer 분포와 기본 검사 결과 확인 |
| 4. 전문가 리뷰 흐름 | PASS | 파일럿 가능 | 20~50개 이상 실제 리뷰 쌍 수집 |
| 5. CPU AI 기준선 | PASS | 시제품에서 파일럿 사이 | 리뷰 우선순위 참고값으로만 사용, 실제 label 전에는 성능 주장 금지 |
| 6. 실제 데이터 전 준비 | CHECK | 파일럿 전 점검 | 충분한 합성 데이터 수와 CPU 기준선 통과 필요 |
| 7. 실제 실행 전 최종 점검 | CHECK | 파일럿 전 점검 | 최신 사전 실행 결과, 민감정보 노출 0, 실데이터 명령 검증 |
| 8. 실제 데이터 batch | PENDING | 대기 | 실제 raw PNG batch 산출물과 전문가 리뷰 필요 |

## 이번 검증에서 강화한 항목

- 실제 sample id는 `product_<10hex>_w<10hex>` 형식으로 강제한다.
- 제품 폴더명을 sample id로 쓰는 옵션은 비활성화했다.
- 실제 path가 들어가는 manifest는 `outputs/private`로 분리한다.
- 실제 PNG 운영 실행용 `--production-run`을 추가해 명시 geometry, 양수 `actual_net_die`, private manifest, `outputs/reports` 출력, reference feature를 강제한다.
- CPU scoring 출력은 기본적으로 `outputs/` 밖으로 나갈 수 없게 제한한다.
- 공유 산출물에서 `png_path`, `arrays_npz`, `metadata_json`, 절대 경로, workspace override를 탐지한다.
- 실제 데이터 전 준비 요약이 작은 sample 수 또는 CPU 기준선 `CHECK`를 숨기지 못하게 했다.
- 실제 데이터 전 준비 요약이 실행 스크립트보다 오래되면 자동 점검이 `CHECK`로 떨어진다.
- 실제 데이터 전 준비 요약에 config sha256, 실행 스크립트 sha256, git commit/dirty 정보를 기록하고, 자동 점검이 hash mismatch를 `CHECK`로 떨어뜨린다.
- 실제 데이터 전 산출물 점검에 필수 산출물 sha256을 기록하고, 자동 점검이 산출물 hash mismatch를 `CHECK`로 떨어뜨린다.
- 문서에 mojibake 코드포인트가 들어오면 테스트가 실패한다.

## 판매 가능 수준으로 올리기 위한 판정 기준

기업 제안 전에 최소한 아래 조건을 만족해야 한다.

| 항목 | 기준 |
| --- | --- |
| 실제 PNG batch | `features.csv`, `sanity.json`, `batch_metadata.json`, `report.html`, `neighbors.csv`, `review_template.csv` 생성 |
| 기본 검사 | error 0 |
| 공유 보안 | raw PNG, 실제 path, private manifest, 제품/lot/tool 식별자 노출 0 |
| geometry/mask | 제품 담당자가 승인한 chip geometry 또는 mask 기준 사용 |
| batch metadata | `production_run=true`, `geometry_contract=explicit`, `manifest_location=outputs/private`, 제품 수와 양수 `actual_net_die` 제품 수 일치 |
| output row count | `features.csv`, `sanity.json`, `neighbors.csv`, `review_template.csv`가 `png_sample_count`와 모순되지 않음 |
| 전문가 리뷰 | 최소 20개, 권장 50개 이상 query wafer 리뷰 |
| 검색 품질 | accepted_match_rate 70% 이상, query_topk_accept_rate 80% 이상 |
| 실패 방지 | missed_major_defect_rate 5% 이하 |
| CPU AI | 실제 리뷰 label 전에는 `synthetic_label_hint`로만 표시 |
| 재현성 | 전체 테스트 PASS, 최신 사전 준비 요약, provenance hash 기록, 자동 점검의 CHECK/PENDING 이유 명시 |

## 남은 최우선/후속 작업

최우선:

- 실제 raw PNG batch를 보안 환경에서 실행하고 8단계 산출물을 생성해야 한다.
- 제품별 geometry/mask 추론이 실제 layout과 맞는지 검증해야 한다. 운영 단계에서는 `--production-run`과 승인된 `--geometry-json`을 쓰고, wafer mask 전략은 기본 검사와 리뷰에서 별도 승인해야 한다.

후속:

- 전체 테스트가 느리다. 빠른 기본 검사와 느린 통합 검사를 분리해야 한다.
- CLI가 아직 독립 스크립트 중심이다. 장기적으로는 `src/wafermap/cli`와 설치형 실행 명령으로 옮기는 것이 좋다.
- 자동 점검은 다음 단계에서 CI run id와 실행 환경 정보를 더 기록해야 한다.

## 현재 결론

지금 솔루션은 “실제 데이터만 넣으면 바로 판매” 수준은 아니다. 하지만 raw PNG 읽기, feature 추출, 전문가 리뷰, CPU 기준선, 실제 데이터 전 준비, 보안 출력 경계까지 전체 흐름은 갖췄다. 다음 이정표는 실제 raw PNG batch를 한 번 돌리고, 기본 검사 오류, 경로 노출, 리뷰 품질을 기준으로 8단계를 `PASS` 또는 `CHECK`로 판정하는 것이다.
