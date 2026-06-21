# WaferMap Defect Intelligence

1채널 고해상도 Wafer Fail Bit Map(FBM)에서 실제 wafer에도 쓸 수 있는 feature와 검색 절차를 만드는 프로젝트입니다.

이 프로젝트의 목표는 불량 이름 하나를 맞히는 분류기가 아닙니다. 목표는 아래 흐름을 실제 업무에서 쓸 수 있게 만드는 것입니다.

```text
FBM 배열 정리
-> 실제 wafer에서도 계산 가능한 feature 추출
-> 비슷한 wafer 검색
-> 관심 불량별 검색
-> defect score / feature table 생성
-> 전문가 리뷰
-> 라벨 없는 실제 wafer 검증
```

## 현재 상태

- 합성 FBM generator는 Grade 0~7, wafer 밖 영역, 실제 test 영역, stby, edge/local/shot/ring/scratch 계열을 생성합니다.
- 전체 유사 wafer 검색은 실제 데이터에서도 계산 가능한 compact feature 50개를 기준으로 합니다.
- Scale 155장 top-k retrieval lift는 약 1.36x, holdout 120장은 약 1.40x입니다.
- 위치가 중요한 검색에서만 `polar_*`, `stby_polar_*` feature를 조건부로 씁니다.
- 단순 resize representation은 전체 유사 wafer 검색의 대체재로 쓰지 않습니다.
- Patch/curve proposal은 최종 판정 도구가 아니라 리뷰 후보를 줄이는 보조 도구입니다.
- Scratch는 rule/proposal을 더 세게 튜닝하지 않고 segmentation 또는 scratch 전용 line feature 쪽으로 분리합니다.
- 라벨 없는 실제 wafer 처리 절차는 제품별 raw PNG 폴더 또는 semantic `.npz` manifest에서 feature CSV, sanity JSON, nearest-neighbor CSV, 전문가 리뷰 양식까지 연결되어 있습니다.

## 먼저 읽을 문서

- [로드맵 메인 HTML](docs/index.html)
- [문서 길잡이](docs/README.md)
- [용어와 변수 설명](docs/glossary.md)
- [프로젝트 개요](docs/project_overview.md)
- [실험과 판단 기록](docs/experiment_history.md)
- [로드맵](docs/roadmap.md)
- [실제 raw PNG 운영 안내서](docs/real_png_operator_runbook.md)
- [라벨 없는 실제 wafer 처리 절차](docs/real_unlabeled_workflow.md)
- [전문가 리뷰 절차](docs/expert_review_protocol.md)
- [Real Wafer 리뷰 체크리스트](docs/real_wafer_review_checklist.md)

## 빠른 실행

```powershell
python -m pytest -q --basetemp .pytest_tmp
python scripts/generate_synthetic.py --config configs/synth/debug.json --out data/synthetic/debug --count 3
python scripts/validate_synthetic.py --data data/synthetic/debug
python scripts/extract_features.py --data data/synthetic/debug --out outputs/reports/synthetic_features.csv
```

합성 데이터 segmentation/embedding 사전 점검:

```powershell
python scripts/build_segmentation_readiness.py `
  --data data/synthetic/debug `
  --manifest outputs/reports/fbm_segmentation_manifest.csv

python scripts/train_embedding_smoke.py `
  --manifest outputs/reports/fbm_segmentation_manifest.csv `
  --out outputs/reports/fbm_embedding_smoke_report.html `
  --metrics outputs/reports/fbm_embedding_smoke_metrics.json `
  --embeddings-out outputs/reports/fbm_embedding_smoke_embeddings.csv

python scripts/train_cpu_encoder_model.py `
  --manifest outputs/reports/fbm_segmentation_manifest.csv `
  --model-out outputs/models/fbm_cpu_encoder_model.npz `
  --out outputs/reports/fbm_cpu_encoder_report.html `
  --metrics outputs/reports/fbm_cpu_encoder_metrics.json `
  --predictions-out outputs/reports/fbm_cpu_encoder_val_predictions.csv
```

실제 데이터 전 전체 준비 실행:

```powershell
python scripts/run_pre_real_readiness.py `
  --config configs/synth/debug.json `
  --out-root outputs/pre_real_readiness `
  --count 20
```

이 명령은 합성 raw grayscale 생성, schema validation, reference feature store, segmentation 준비 점검, embedding 점검, CPU encoder 학습, 라벨 없는 scoring 점검, 합성 raw PNG batch 점검을 한 번에 실행한다.

프로젝트 단계별 자동 점검:

```powershell
python scripts/audit_project_readiness.py
```

이 명령은 목표 단계별 구현물, 테스트 근거, 실제 데이터 전 준비 결과, 실제 PNG batch 산출물 유무를 점검해 `outputs/reports/project_readiness_audit.json`과 `.html`을 만든다. 실제 raw PNG를 아직 돌리기 전에는 실제 wafer 산출물이 필요한 8단계만 `PENDING`으로 남는 것이 정상이다.

라벨 없는 실제 wafer 처리 smoke test:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --features-out outputs/reports/real_unlabeled_features.csv `
  --sanity-out outputs/reports/real_unlabeled_sanity.json `
  --report-out outputs/reports/real_unlabeled_report.html `
  --neighbors-out outputs/reports/real_unlabeled_neighbors.csv `
  --review-template-out outputs/reports/real_unlabeled_expert_review_template.csv
```

제품별 raw PNG 폴더 일괄 처리:

```powershell
python scripts/analyze_png_raw_folders.py `
  --raw-root D:/secure_fbm/raw_png `
  --production-run `
  --geometry-json D:/secure_fbm/product_geometry.json `
  --out-dir outputs/reports/real_png_batch `
  --reference-features outputs/pre_real_readiness/reports/synthetic_reference_features.csv `
  --cpu-model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz
```

CPU encoder scoring can also be run separately for an existing manifest:

```powershell
python scripts/score_unlabeled_cpu_encoder.py `
  --model outputs/pre_real_readiness/models/fbm_cpu_encoder_model.npz `
  --manifest outputs/private/real_png_batch_manifest.json `
  --predictions-out outputs/reports/real_png_batch/cpu_encoder_predictions.csv `
  --neighbors-out outputs/reports/real_png_batch/cpu_encoder_neighbors.csv `
  --sanity-out outputs/reports/real_png_batch/cpu_encoder_sanity.json `
  --report-out outputs/reports/real_png_batch/cpu_encoder_report.html
```

입력 폴더는 제품별 하위 폴더를 둔다.

```text
D:/secure_fbm/raw_png/
  product_a/
    wafer_001.png
    wafer_002.png
  product_b/
    wafer_001.png
```

## 보안 원칙

- 실제 wafer raw image/array는 repo에 저장하지 않습니다.
- 실제 wafer 입력은 보안 환경의 제품별 raw PNG 폴더 또는 `.npz` manifest로만 참조합니다.
- 실제 path가 들어간 batch manifest 원본은 기본적으로 `outputs/private/`에 생성되며 공유하지 않습니다.
- 결과를 공유할 때는 파일 단위로만 공유하고, `manifest`, 실제 path, lot/tool/recipe/chamber/wafer id가 포함되지 않았는지 확인합니다.
- Repo에는 code, config, schema, synthetic preset, 익명화된 feature/report만 남깁니다.
- 합성 데이터의 oracle label/mask는 검증용이며, 실제 inference feature에 섞지 않습니다.
