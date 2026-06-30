# 문서 목차

WaferMap 문서는 아래 한 가지 제품 흐름을 기준으로 정리합니다.

```text
FBM wafer map
-> 대표 불량 패턴 추출
-> 여러 불량이 섞인 합성맵 생성
-> multi-label segmentation 학습/검증
-> 실제 wafer 예측과 수정
-> 수정 결과를 다시 asset으로 축적
```

## 먼저 볼 문서

0. [문서 표지판](index.html): 어떤 문서를 먼저 볼지, 현재 구현은 어디까지인지, 다음에 어떤 명령을 실행할지 한 화면에서 봅니다.

1. [핵심 방향](core_direction.md): 이 프로젝트가 흔들리면 안 되는 범위와 의사결정 기준입니다.

2. [전체 실행 흐름](end_to_end_workflow.md): wafer 입력부터 asset, 합성맵, U-Net 학습, 예측 export, correction까지의 기준 명령 흐름입니다.

3. [작업자 매뉴얼](operator_manual.md): 실제로 파이프라인을 돌릴 때 보는 짧은 실행 가이드입니다.

4. [반도체 AI 설계 검토](semiconductor_ai_review.md): resize, target clipping, 위치 정보를 쓰는 합성, readiness check를 왜 이렇게 하는지 설명합니다.

## 핵심 참고 문서

- [설계 구조](architecture.md): 패키지 경계, command 소유권, 새 코드를 넣을 위치를 정리합니다.
- [프로젝트 개요](project_overview.md): 현재 제품 형태, 호환성 파일, 다음 구현 순서를 정리합니다.
- [세그멘테이션 도구 흐름](segmentation_tool_workflow.md): tool 조작, mask 편집, asset 저장, prediction prefill 흐름입니다.
- [FBM 데이터 흐름](fbm_data_flow_guide.md): 파일 위치, manifest 출력, asset 경로, model artifact 경로입니다.
- [합성 데이터 파이프라인](fbm_pattern_asset_pipeline.md): pattern asset 형식, procedural fallback, readiness, training 흐름입니다.
- [불량 family 정의](pattern_taxonomy.md): `local`, `scratch`, `ring`, `edge`, `shot_grid`, `random`의 의미입니다.
- [라벨 데이터 가이드](label_data_guidelines.md): wafer-level tag, rough region, clean mask, reusable pattern asset을 어떻게 저장할지 예시로 설명합니다.
- [학습 데이터 규격](training_data_contract.md): U-Net 학습에 필요한 `arrays.npz`, `metadata.json`, manifest, input tensor, target tensor 규격입니다.
- [모델 이해 가이드](modeling_strategy.md): U-Net이 무엇을 입력받고, 어떤 mask를 예측하고, prediction을 어떻게 해석하는지 설명합니다.
- [데이터 스키마](data_schema.md): arrays, masks, metadata, manifest schema입니다.
- [로드맵](roadmap.md): 단계별 구현 상태입니다.
- [용어 사전](glossary.md): `severity`, `manifest`, `pattern asset`, `retrieval_failure_mode` 같은 공통 용어입니다.
- [실행 명령 지도](../scripts/README.md): 안정적으로 쓰는 command-line entrypoint 목록입니다.

## 실제 데이터와 검토

- [실제 PNG 실행 가이드](real_png_operator_runbook.md)
- [실제 unlabeled wafer 흐름](real_unlabeled_workflow.md)
- [전문가 검토 프로토콜](expert_review_protocol.md)
- [실제 wafer 검토 체크리스트](real_wafer_review_checklist.md)

## 보조/이력 문서

- [실험 이력](experiment_history.md)
- [모델링 전략](modeling_strategy.md)
- [검증 프로토콜](validation_protocol.md)
- [문서 품질 검증표](documentation_quality_audit.md)
- [엔터프라이즈 준비도 평가](enterprise_readiness_assessment.md)
- [호환성 pattern asset editor](legacy_pattern_asset_editor.md)

## 문서 관리 규칙

- 메인 command 순서는 [전체 실행 흐름](end_to_end_workflow.md)을 기준으로 유지합니다.
- tool 조작 상세는 [세그멘테이션 도구 흐름](segmentation_tool_workflow.md)에 둡니다.
- artifact와 폴더 규격은 [FBM 데이터 흐름](fbm_data_flow_guide.md)에 둡니다.
- 합성과 학습 규격은 [합성 데이터 파이프라인](fbm_pattern_asset_pipeline.md)과 [학습 데이터 규격](training_data_contract.md)에 둡니다.
- 새 script를 추가하면 [실행 명령 지도](../scripts/README.md)에 등록합니다.
- 브라우저에서 읽는 예쁜 HTML 문서는 `python scripts/build_static_docs.py`로 다시 생성합니다.
