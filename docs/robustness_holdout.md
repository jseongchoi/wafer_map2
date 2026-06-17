# Robustness Holdout

## 목적

현재 scale pilot의 결과가 특정 generator seed, class prior, grade threshold에만 맞은 것인지 확인한다.

이 단계는 최종 모델 학습이 아니라 다음 질문에 답하기 위한 gate다.

```text
현재 observable feature baseline이 조건이 바뀐 synthetic batch에서도 유지되는가?
관심 defect 기준별 retrieval 강약이 재현되는가?
scratch/local 약점이 holdout에서도 반복되는가?
```

## Holdout 설정

Config:

```text
configs/synth/presets/fbm_grouping_holdout_stress.json
```

변경점:

- seed 변경
- sample count 120
- scratch/ring 비중 증가
- edge/shot 비중 감소
- stby chip 수 증가
- grade threshold 상향으로 subtle/weak signal 증가

## 실행 명령

```powershell
python scripts/generate_synthetic.py `
  --config configs/synth/presets/fbm_grouping_holdout_stress.json `
  --out data/synthetic/fbm_grouping_holdout_stress `
  --no-preview

python scripts/validate_synthetic.py `
  --data data/synthetic/fbm_grouping_holdout_stress

python scripts/analyze_fbm_grouping.py `
  --data data/synthetic/fbm_grouping_holdout_stress `
  --out outputs/reports/fbm_grouping_holdout_report.html `
  --metrics outputs/reports/fbm_grouping_holdout_metrics.json `
  --features outputs/reports/fbm_grouping_holdout_features.csv `
  --neighbors outputs/reports/fbm_grouping_holdout_neighbors.csv `
  --figure outputs/figures/fbm_grouping_holdout_pca.png `
  --clusters 3

python scripts/evaluate_retrieval_confidence.py `
  --features outputs/reports/fbm_grouping_holdout_features.csv `
  --out outputs/reports/fbm_retrieval_confidence_holdout_report.html `
  --metrics outputs/reports/fbm_retrieval_confidence_holdout_metrics.json

python scripts/evaluate_interest_retrieval.py `
  --features outputs/reports/fbm_grouping_holdout_features.csv `
  --data data/synthetic/fbm_grouping_holdout_stress `
  --out outputs/reports/fbm_interest_retrieval_holdout_report.html `
  --metrics outputs/reports/fbm_interest_retrieval_holdout_metrics.json `
  --neighbors-out outputs/reports/fbm_interest_retrieval_holdout_neighbors.csv `
  --gallery outputs/figures/fbm_interest_neighbor_gallery_holdout.png
```

## 통과 기준

- 전체 top-k retrieval lift가 1.10x 이상
- interest-based retrieval에서 shot/edge/ring 중 일부 기준이 유지
- scratch/local이 계속 약하면 segmentation 또는 morphology 보강으로 넘긴다
- 성능이 무너지면 feature distance 또는 synthetic generator 편향을 다시 점검한다

## 2026-06-17 결과

120장 holdout 전체를 morphology feature 50개 기준으로 재실행했다.

산출물:

- `outputs/reports/fbm_grouping_holdout_report.html`
- `outputs/reports/fbm_retrieval_confidence_holdout_report.html`
- `outputs/reports/fbm_interest_retrieval_holdout_report.html`
- `outputs/figures/fbm_interest_neighbor_gallery_holdout.png`

전체 retrieval:

- sample: 120
- observable feature: 50
- top-5 label-Jaccard lift: 약 1.40x
- bootstrap 95% CI: 약 1.32x ~ 1.48x
- permutation p-value: 약 0.002

관심 기준별 retrieval:

- `edge_focus`: 약 2.64x
- `stby_focus`: 약 1.48x
- `ring_focus`: 약 1.52x
- `shot_focus`: 약 1.23x
- `local_focus`: 약 1.33x
- `scratch_focus`: 약 0.92x

판정:

- baseline은 holdout에서도 무너지지 않았다.
- edge와 stby는 안정적인 triage 후보로 유지한다.
- ring은 synthetic holdout에서 신호가 유지된다.
- local은 connected-component morphology 보강 후 개선되어, 물방울/쌍방울/삼각 blob류에는 CPU feature baseline이 일부 유효하다.
- shot은 generator 조건 변화에 민감하므로 real review 전 calibration이 필요하다.
- scratch는 morphology 보강 후에도 불안정하므로 다음 단계에서 segmentation 또는 curve/arc-specific detector로 넘기는 것이 맞다.
