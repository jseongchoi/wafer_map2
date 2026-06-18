# 불량 패턴 정리

이 문서는 완전한 공정 taxonomy가 아니다. 합성 데이터 생성, feature 검증, 전문가 리뷰에서 같은 단어를 쓰기 위한 최소 defect family 정의다.

## Defect Family

| Family | 의미 | 현재 처리 |
| --- | --- | --- |
| `edge` | wafer edge 근처 fail density 상승 또는 localized edge sector | compact feature와 patch proposal에서 비교적 안정적 |
| `shot_grid` | reticle/shot-relative 반복 위치 defect | wafer polar 좌표가 아니라 shot-relative feature로 관리 |
| `stby_pattern` | stby fail chip의 chip-level missing-test pattern | `stby_mask`, `valid_test_mask`로 Grade 7과 분리 |
| `stby_hidden_origin` | stby가 실제 defect origin을 가리는 경우 | 전문가 리뷰와 향후 stby-origin coupling feature 후보 |
| `ring` | wafer 중심 기준 annulus 또는 partial ring/arc | curve proposal과 radial profile로 관리 |
| `scratch` | 길고 좁은 선형/곡선형 defect | 현재 feature/proposal로 약함. 별도 line/segmentation track |
| `local` | 국소 hotspot 또는 compact blob cluster | connected-component morphology와 전문가 리뷰로 보강 |
| `random` | 구조가 약한 산발성 fail | background/noise baseline |
| `mixed` | 여러 family가 동시에 보이는 경우 | multi-label mask와 전문가 리뷰로 관리 |

## Synthetic Mode

`scratch`:

- `spin_arc`: 회전/스핀 공정에서 나온 arc-like scratch
- `radial`: 중심에서 edge 방향으로 퍼지는 scratch
- straight chord-like scratch는 기본 target이 아니다.

`ring`:

- full ring
- partial ring
- center arc
- radius/width mismatch가 중요한 failure mode다.

`shot_grid`:

- shot row/column modulo 기준 반복
- lower-left, bottom-edge, left-edge contrast로 관찰
- wafer polar coordinate만으로 설명하지 않는다.

`stby_pattern`:

- random
- scratch_like
- ring_like
- edge_like
- local_cluster

## Overlap Rule

Defect family는 서로 중첩될 수 있다.

Synthetic validation mask는 multi-label이다.

```text
pattern_masks[class, y, x] = 0 or 1
```

한 pixel/cell block에 여러 class가 동시에 1일 수 있다.

Segmentation baseline을 학습할 때는 softmax가 아니라 class별 sigmoid output을 기본으로 한다.

## Review Rule

전체 유사 wafer 검색은 hard class 하나를 맞히는 문제가 아니다.

전문가 리뷰에서는 다음을 따로 본다.

- family가 같은가?
- 위치/clock이 비슷한가?
- query의 주요 defect를 놓쳤는가?
- mismatch라면 어떤 feature/model 보강 작업으로 연결할 것인가?
