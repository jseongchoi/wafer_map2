# Segmentation Readiness

## 목적

이 문서는 Phase 3로 넘어가기 전에 synthetic segmentation target이 학습 가능한 형태인지 확인한 중간점검이다.

현재 결론은 다음과 같다.

```text
local: morphology baseline으로 일부 유효하므로 expert review로 확인
scratch: wafer-level feature가 불안정하므로 segmentation 또는 scratch-specific representation 필요
```

## 산출물

Scale pilot:

- `outputs/reports/fbm_segmentation_readiness_report.html`
- `outputs/reports/fbm_segmentation_readiness_metrics.json`
- `outputs/reports/fbm_segmentation_manifest.csv`
- `outputs/figures/fbm_segmentation_readiness_gallery.png`

Holdout stress:

- `outputs/reports/fbm_segmentation_readiness_holdout_report.html`
- `outputs/reports/fbm_segmentation_readiness_holdout_metrics.json`
- `outputs/reports/fbm_segmentation_manifest_holdout.csv`
- `outputs/figures/fbm_segmentation_readiness_holdout_gallery.png`

Smoke training:

- `outputs/reports/fbm_segmentation_smoke_report.html`
- `outputs/reports/fbm_segmentation_smoke_metrics.json`
- `outputs/figures/fbm_segmentation_smoke_loss.png`
- `outputs/reports/fbm_segmentation_smoke_holdout_report.html`
- `outputs/reports/fbm_segmentation_smoke_holdout_metrics.json`
- `outputs/figures/fbm_segmentation_smoke_holdout_loss.png`

## 입력/출력 계약

초기 segmentation input channel:

```text
severity
wafer_mask
valid_test_mask
stby_mask
```

초기 target channel:

```text
scratch
ring
edge
local
random
shot_grid
stby_pattern
```

Target은 class별 sigmoid multi-label mask로 둔다. 결함은 중첩될 수 있으므로 softmax single-class segmentation으로 만들지 않는다.

## 주요 관찰

Scale 155장 기준:

- scratch positive sample: 60장
- scratch pixel mean ratio: 약 0.0009
- scratch + ring co-occurrence: 약 0.48
- scratch + local co-occurrence: 약 0.70
- scratch + stby co-occurrence: 약 0.83
- scratch pixel 중 stby로 가려지는 비율 p95: 약 0.109

Holdout 120장 기준:

- scratch positive sample: 78장
- scratch pixel mean ratio: 약 0.0017
- scratch + ring co-occurrence: 약 0.41
- scratch + local co-occurrence: 약 0.55
- scratch + stby co-occurrence: 약 0.68
- scratch pixel 중 stby로 가려지는 비율 p95: 약 0.123

## 해석

Scratch는 픽셀 면적이 작고 ring/local/stby와 자주 같이 나타난다. 따라서 wafer-level feature vector에서 scratch만 분리하는 것은 구조적으로 어렵다.

특히 stby가 scratch의 시작점이나 충돌점을 가리면 severity만으로는 원인을 직접 볼 수 없다. 이 경우 input에는 `stby_mask`를 넣되, target은 scratch와 stby를 별도 channel로 둬야 한다.

## 다음 단계

1. NumPy-only 1x1 sigmoid smoke training으로 manifest, tensor, target, weighted BCE 연결을 확인했다.
2. 이 smoke baseline은 loss가 감소하지만 scratch/local 공간 구조는 잡지 못한다.
3. 다음 모델은 작은 U-Net 또는 lightweight SegFormer 계열로 간다.
4. metric은 전체 mIoU보다 scratch recall, stby-hidden scratch recall, local small-blob recall을 우선한다.
