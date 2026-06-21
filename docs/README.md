# 문서 길잡이

이 폴더는 처음 보는 사람이 프로젝트를 빠르게 이해하고 실행할 수 있도록 핵심 문서만 유지한다.

## 추천 읽기 순서

0. [로드맵 메인 HTML](index.html)
   - 현재 위치, 단계별 진행 상황, 다음 할 일을 한 화면에서 보는 요약 페이지.

1. [프로젝트 개요](project_overview.md)
   - 목표, 현재 상태, 지금까지 확인한 근거, 하지 말아야 할 일.

2. [용어와 변수 설명](glossary.md)
   - `severity`, `wafer_mask`, `manifest`, `top-k`, `retrieval_failure_mode`처럼 처음 보는 단어의 뜻.

3. [실험과 판단 기록](experiment_history.md)
   - 지금까지 시도한 feature, retrieval, resize, proposal, segmentation smoke test와 각 판단 근거.

4. [로드맵](roadmap.md)
   - 완료된 단계, 현재 단계, 다음 확인 단계.

5. [데이터 형식](data_schema.md)
   - FBM 배열, 합성 sample, 라벨 없는 실제 wafer manifest 작성 기준.

6. [라벨 없는 실제 wafer 처리 절차](real_unlabeled_workflow.md)
   - 보안 환경의 제품별 raw PNG 폴더 또는 `.npz` 입력에서 feature, sanity 결과, nearest-neighbor 결과, 리뷰 양식을 만드는 절차.

7. [실제 raw PNG 운영 안내서](real_png_operator_runbook.md)
   - raw PNG 폴더 준비, geometry JSON, 운영 실행, 자동 점검, 리뷰, 결과 공유 양식까지 한 번에 따라가는 절차.

8. [전문가 리뷰 절차](expert_review_protocol.md)
   - top-k 검색 결과를 사람이 평가하고, 다음 feature/AI 보강 작업으로 연결하는 방법.

9. [Real Wafer 리뷰 체크리스트](real_wafer_review_checklist.md)
   - 실제 wafer 리뷰 전에 준비할 것, 실행 결과, 리뷰어가 채워야 할 항목, AI 모델 구현 상태.

10. [기업 도입 준비도 평가](enterprise_readiness_assessment.md)
   - 기업 파일럿/판매 관점에서 단계별 현재 상태, 판정 기준, 남은 최우선/후속 작업을 정리한 문서.

## 목적별 읽기 경로

- 전체 맥락만 빠르게 파악: `project_overview.md` -> `roadmap.md`
- 용어가 낯설 때: `glossary.md` -> `data_schema.md`
- 실제 PNG를 넣고 결과만 공유할 때: `real_png_operator_runbook.md` -> `real_wafer_review_checklist.md`
- 내일 실제 wafer로 검증할 일 확인: `real_wafer_review_checklist.md` -> `real_unlabeled_workflow.md`
- 실험 과정과 판단 근거 확인: `experiment_history.md` -> `modeling_strategy.md` -> `validation_protocol.md`
- 구현/연동 담당자 관점: `data_schema.md` -> `real_unlabeled_workflow.md` -> `expert_review_protocol.md`

## 설계와 검증 참고

- [실험과 판단 기록](experiment_history.md): 실험한 기법과 유지/중단 판단의 흐름.
- [용어와 변수 설명](glossary.md): 입력 배열, manifest, 검색 지표, 리뷰 컬럼 정의.
- [모델링 전략](modeling_strategy.md): feature 기반 기준선, segmentation/self-supervised 모델의 역할, 참고 논문.
- [검증 방법](validation_protocol.md): 합성 데이터 검증, 실제 데이터 sanity check, retrieval lift, 전문가 리뷰 기준.
- [불량 패턴 정리](pattern_taxonomy.md): scratch, ring, edge, local, random, shot, stby pattern 정의.

## 문서 정리 원칙

- `README.md`는 프로젝트 입구만 담당한다.
- 이 파일은 문서 안내만 담당한다.
- 실험별 HTML/JSON report는 `outputs/`에 재생성 가능한 산출물로 둔다.
- 과거 checkpoint나 중복 진행 보고서는 핵심 문서에 흡수하고 별도 문서로 유지하지 않는다.
