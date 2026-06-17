# WaferMap Defect Intelligence

1채널 고해상도 Wafer Fail Bit Map(FBM)에서 실제 wafer에도 적용 가능한 feature 표현을 만들기 위한 프로젝트입니다.

핵심 목표는 단일 defect classification이 아니라 다음 흐름을 안정적으로 만드는 것입니다.

```text
semantic FBM tensor
-> compact observable feature
-> similar wafer retrieval
-> interest-conditioned defect retrieval
-> defect score / feature table
-> expert review
-> real-unlabeled secure workflow
```

ANOVA와 공정/설비/lot/recipe/chamber metadata 분석은 현재 목표가 아닙니다. FBM feature table이 안정화되고 process metadata와 조인된 뒤의 후속 분석입니다.

## 현재 상태

- Synthetic generator는 Grade 0~7, none-wafer, valid-test, stby, edge/local/shot/ring/scratch 계열을 생성합니다.
- Global retrieval은 compact observable feature 50개 기준을 유지합니다.
- Scale 155장 top-k retrieval lift는 약 1.36x, holdout 120장은 약 1.40x입니다.
- 위치-aware retrieval은 `class_location`, `feature_key` 같은 경우에만 `polar_*`, `stby_polar_*` feature를 조건부 사용합니다.
- Resize-only representation은 global retrieval 대체재로 쓰지 않습니다.
- Patch/curve proposal은 review 후보 축소용 보조층입니다.
- Scratch는 rule/proposal 과투자 대신 segmentation 또는 scratch-specific line representation track으로 분리합니다.
- Real-unlabeled workflow MVP는 feature CSV, sanity JSON, nearest-neighbor CSV, expert review template까지 연결되어 있습니다.

## 먼저 읽을 문서

- [문서 길잡이](docs/README.md)
- [프로젝트 개요](docs/project_overview.md)
- [로드맵](docs/roadmap.md)
- [Real-Unlabeled Workflow](docs/real_unlabeled_workflow.md)
- [Expert Review Protocol](docs/expert_review_protocol.md)
- [Real Wafer Review Checklist](docs/real_wafer_review_checklist.md)

## 빠른 실행

```powershell
python -m pytest -q --basetemp .pytest_tmp
python scripts/generate_synthetic.py --config configs/synth/debug.json --out data/synthetic/debug --count 3
python scripts/validate_synthetic.py --data data/synthetic/debug
python scripts/extract_features.py --data data/synthetic/debug --out outputs/reports/synthetic_features.csv
```

Real-unlabeled smoke:

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
- Real input은 보안 환경의 `.npz` manifest로 참조합니다.
- Repo에는 code, config, schema, synthetic preset, 익명화된 feature/report만 남깁니다.
- Synthetic oracle label/mask는 검증용이며 real inference feature에 섞지 않습니다.
