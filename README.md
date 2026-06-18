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
- 라벨 없는 실제 wafer 처리 절차는 feature CSV, sanity JSON, nearest-neighbor CSV, 전문가 리뷰 template까지 연결되어 있습니다.

## 먼저 읽을 문서

- [로드맵 메인 HTML](docs/index.html)
- [문서 길잡이](docs/README.md)
- [프로젝트 개요](docs/project_overview.md)
- [실험과 판단 기록](docs/experiment_history.md)
- [로드맵](docs/roadmap.md)
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

라벨 없는 실제 wafer 처리 smoke test:

```powershell
python scripts/extract_real_unlabeled_features.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --reference-features outputs/reports/fbm_grouping_scale_features.csv `
  --features-out outputs/reports/real_unlabeled_features.csv `
  --sanity-out outputs/reports/real_unlabeled_sanity.json `
  --report-out outputs/reports/real_unlabeled_report.html `
  --neighbors-out outputs/reports/real_unlabeled_neighbors.csv `
  --review-template-out outputs/reports/real_unlabeled_expert_review_template.csv
```

## 보안 원칙

- 실제 wafer raw image/array는 repo에 저장하지 않습니다.
- 실제 wafer 입력은 보안 환경의 `.npz`와 manifest로만 참조합니다.
- Repo에는 code, config, schema, synthetic preset, 익명화된 feature/report만 남깁니다.
- 합성 데이터의 oracle label/mask는 검증용이며, 실제 inference feature에 섞지 않습니다.
