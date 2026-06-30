# 실행 명령 지도

이 폴더는 사용자가 직접 실행하는 안정적인 command entrypoint를 모아 둡니다.
핵심 구현은 가능하면 `src/wafermap/`에 두고, `scripts/` 파일은 argument를 받고
라이브러리 함수를 호출하는 얇은 wrapper로 유지합니다.

## 1. 현재 주력 Segmentation Pipeline

먼저 볼 명령입니다. 현재 제품 방향은 이 흐름을 기준으로 합니다.

| Script | 목적 | 언제 쓰나 |
|---|---|---|
| `analyze_png_raw_folders.py` | raw PNG 폴더를 manifest와 report로 변환 | 실제 wafer image가 있을 때 |
| `run_segmentation_tool.py` | local browser segmentation tool 실행, pattern asset 저장 | 사람이 mask를 만들 때 |
| `build_pattern_asset_report.py` | 저장된 pattern asset 검수 report 생성 | asset 품질 확인 |
| `compose_synthetic_from_assets.py` | asset/procedural defect를 base wafer에 합성 | 학습 sample 생성 |
| `build_segmentation_readiness.py` | segmentation manifest와 readiness metric 생성 | 학습 전 점검 |
| `run_pattern_asset_pipeline.py` | 합성, readiness, smoke, report를 한 번에 실행 | 전체 경로 점검 |
| `train_unet_segmentation.py` | coordinate-aware small U-Net 학습 | 학습 시작 |
| `export_unet_predictions.py` | U-Net prediction을 `fbm_prediction_masks/v1`로 export | tool correction seed 생성 |

예시:

```powershell
python scripts/run_pattern_asset_pipeline.py `
  --assets-root data/pattern_assets `
  --composed-dir data/synthetic/asset_composed `
  --work-dir outputs/pattern_asset_pipeline `
  --report-out outputs/reports/pattern_asset_project_report.html
```

## 2. 데이터와 합성 생성

| Script | 목적 |
|---|---|
| `generate_synthetic.py` | procedural synthetic wafer sample 생성 |
| `validate_synthetic.py` | synthetic sample schema 검증 |
| `build_segmentation_readiness.py` | 학습 manifest, readiness metric, gallery 생성 |
| `train_segmentation_smoke.py` | segmentation tensor/loss 연결 smoke test |
| `train_embedding_smoke.py` | embedding retrieval wiring smoke test |

## 3. 실제 wafer 진단

| Script | 목적 |
|---|---|
| `extract_real_unlabeled_features.py` | real/unlabeled manifest에서 feature 추출 |
| `extract_features.py` | synthetic sample feature 추출 |
| `interpret_fbm.py` | feature/sanity/neighbor 결과 해석 report 생성 |
| `summarize_defect_locations.py` | defect 위치 분포 요약 |
| `make_expert_review_template.py` | 전문가 리뷰용 CSV template 생성 |
| `summarize_expert_review.py` | 전문가 리뷰 CSV 집계 |

## 4. 모델링과 Retrieval 진단

이 영역은 현재 segmentation dataset pipeline보다 우선순위가 낮습니다.
하지만 과거 실험 재현과 후보 추천에 필요할 수 있어 유지합니다.

| Script | 목적 |
|---|---|
| `train_cpu_encoder_model.py` | 가벼운 CPU encoder baseline 학습 |
| `score_unlabeled_cpu_encoder.py` | unlabeled wafer batch scoring |
| `evaluate_defect_feature_retrieval.py` | defect feature retrieval 동작 평가 |
| `evaluate_interest_retrieval.py` | interest/retrieval target 평가 |
| `evaluate_retrieval_confidence.py` | retrieval confidence 평가 |
| `evaluate_resize_benchmark.py` | resize-only representation benchmark |

## 5. 연구/이력용 평가

아래 명령은 실험 추적을 위해 남깁니다. 메인 workflow로 보지 않습니다.

| Script | 목적 |
|---|---|
| `analyze_fbm_grouping.py` | FBM synthetic grouping 분석 |
| `evaluate_curve_proposals.py` | curve/ring proposal 평가 |
| `evaluate_feature_ablation.py` | feature ablation 평가 |
| `evaluate_grouping_stability.py` | grouping stability 평가 |
| `evaluate_methodology.py` | methodology probe 평가 |
| `evaluate_patch_proposals.py` | patch proposal 평가 |
| `sweep_grouping_parameters.py` | grouping parameter sweep |

## 6. 호환성

| Script | 목적 |
|---|---|
| `run_pattern_asset_editor.py` | 기존 이름을 유지하는 호환 wrapper. 실제 방향은 segmentation tool 중심 |

## 7. Report와 상태 공유

| Script | 목적 |
|---|---|
| `make_report.py` | synthetic experiment report 생성 |
| `make_progress_briefing.py` | 진행 상황 briefing artifact 생성 |
| `make_leader_status_report.py` | 리더/상태 report 생성 |
| `run_pre_real_readiness.py` | pre-real readiness pipeline 실행 |

## 8. 새 script 추가 규칙

- core logic은 `src/wafermap/`에 둡니다.
- script는 argument parsing과 출력 경로 안내에 집중합니다.
- 작업자가 직접 쓰는 명령이면 `README.md`, `docs/operator_manual.md`, 관련 workflow 문서에 추가합니다.
- 실험용이면 이 문서의 연구/이력 섹션에 표시합니다.
- 기존 script 파일명을 없애야 한다면 wrapper를 남겨 호환성을 지킵니다.
