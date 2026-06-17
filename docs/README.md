# Documentation Guide

이 폴더는 처음 보는 사람이 프로젝트를 빠르게 이해하고 실행할 수 있도록 핵심 문서만 유지한다.

## 추천 읽기 순서

1. [프로젝트 개요](project_overview.md)
   - 목적, 현재 상태, 목표 대비 진행/연구 근거, 하지 말아야 할 일.

2. [실험/기법 진행 기록](experiment_history.md)
   - 지금까지 실험한 feature, retrieval, resize, proposal, segmentation smoke, real-unlabeled workflow와 각 판단 근거.

3. [로드맵](roadmap.md)
   - 단계별 진행 상태와 다음 작업.

4. [데이터 스키마](data_schema.md)
   - semantic tensor, synthetic sample, real-unlabeled manifest 계약.

5. [Real-Unlabeled Workflow](real_unlabeled_workflow.md)
   - 보안 환경의 `.npz` 입력에서 feature/sanity/nearest-neighbor/review template까지 생성하는 운영 절차.

6. [Expert Review Protocol](expert_review_protocol.md)
   - top-k retrieval 결과를 사람이 평가하고 feature/model backlog로 연결하는 방법.

7. [Real Wafer Review Checklist](real_wafer_review_checklist.md)
   - 실제 wafer 리뷰 전 준비물, 실행 결과, 리뷰어가 채워야 할 항목, AI 모델 구현 상태.

## 읽는 목적별 경로

- 전체 맥락만 빠르게 파악: `project_overview.md` -> `roadmap.md`
- 내가 내일 할 일 확인: `real_wafer_review_checklist.md` -> `real_unlabeled_workflow.md`
- 실험 과정과 판단 근거 확인: `experiment_history.md` -> `modeling_strategy.md` -> `validation_protocol.md`
- 구현/연동 담당자 관점: `data_schema.md` -> `real_unlabeled_workflow.md` -> `expert_review_protocol.md`

## 설계와 검증 참고

- [Experiment History](experiment_history.md): 실험한 기법과 폐기/유지 판단의 상세 흐름.
- [Modeling Strategy](modeling_strategy.md): observable baseline, segmentation/self-supervised 모델의 위치, 참고 논문.
- [Validation Protocol](validation_protocol.md): synthetic/real sanity, retrieval lift, expert review gate.
- [Pattern Taxonomy](pattern_taxonomy.md): scratch, ring, edge, local, random, shot, stby pattern 정의.

## 문서 정리 원칙

- README는 입구만 담당한다.
- 이 파일은 문서 index만 담당한다.
- 실험별 HTML/JSON report는 `outputs/`에 재생성 산출물로 둔다.
- 과거 checkpoint나 중복 진행 보고서는 핵심 문서에 흡수하고 별도 문서로 유지하지 않는다.
