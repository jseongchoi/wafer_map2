# Curve Proposal Strategy

기준일: 2026-06-17

## 왜 추가했나

Patch proposal report의 cyan 네모는 defect 자체가 아니라 원본 해상도에서 열어볼 후보 window다.

이 방식은 edge/local/stby처럼 국소 hotspot 또는 특정 edge 영역으로 나타나는 결함에는 잘 맞는다. 하지만 scratch, partial ring, center arc처럼 wafer 전체 좌표계에서 곡선으로 이어지는 결함은 네모 window만으로 해석하면 안 된다.

그래서 별도 polar-CV baseline을 추가했다.

```text
Observed FBM severity/stby
-> pixel-level polar accumulator
-> annulus / partial_ring / spin_arc / radial_scratch 후보 생성
-> synthetic oracle mask로 recall 채점
```

Synthetic oracle은 평가에만 사용하고, proposal score는 관측 가능한 `severity`, fail density, high-grade, stby, wafer mask로만 계산한다.

## 결과

| Dataset | Class | Proposal recall | Random recall | Recall lift | Hit@30% | Hit@50% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scale | ring | 0.315 | 0.222 | 1.42x | 0.517 | 0.233 |
| Scale | scratch | 0.010 | 0.081 | 0.13x | 0.000 | 0.000 |
| Holdout | ring | 0.434 | 0.287 | 1.51x | 0.696 | 0.464 |
| Holdout | scratch | 0.038 | 0.174 | 0.22x | 0.051 | 0.026 |

## 해석

Ring/partial-ring 계열:

- polar representation이 맞다.
- 네모 patch proposal보다 annulus/partial-ring 후보가 더 의미 있는 해석 단위다.
- center arc 또는 wafer 중앙부 호 모양 결함은 이 계열로 계속 관리한다.

Scratch 계열:

- 단순 patch proposal도 약했고, polar curve proposal도 평균적으로 약하다.
- 일부 큰 arc scratch는 gallery에서 잡히지만, 전체 평균은 random보다 낮다.
- 따라서 scratch는 현재 CPU-only rule/CV baseline의 성공 영역이 아니다.
- 다음 단계에서는 line enhancement, skeleton/curve extraction, 또는 segmentation 모델로 넘겨야 한다.

중요한 판단:

```text
edge/local/stby:
  patch proposal로 후보 영역 축소 가능

ring/center arc:
  polar curve proposal로 후보 곡선 영역 축소 가능

scratch:
  별도 scratch-specific representation 또는 segmentation 필요
```

## 현재 산출물

- `scripts/evaluate_curve_proposals.py`
- `outputs/reports/fbm_curve_proposal_scale_report.html`
- `outputs/reports/fbm_curve_proposal_scale_metrics.json`
- `outputs/figures/fbm_curve_proposal_scale_gallery.png`
- `outputs/reports/fbm_curve_proposal_holdout_report.html`
- `outputs/reports/fbm_curve_proposal_holdout_metrics.json`
- `outputs/figures/fbm_curve_proposal_holdout_gallery.png`

## 다음 보강

1. Scratch-specific line enhancement
   - thin trace 강조
   - local background subtraction
   - radial/arc skeleton continuity score

2. Segmentation baseline
   - small U-Net 또는 lightweight SegFormer
   - scratch/local/stby overlap을 multi-label target으로 학습
   - GPU 환경에서만 본격 검증

3. Expert review
   - ring/center arc proposal이 실제 현업 defect review에 맞는지 확인
   - scratch 실패 케이스를 모아 어떤 모양이 놓치는지 분류
