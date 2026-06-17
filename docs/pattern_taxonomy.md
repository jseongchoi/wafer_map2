# Pattern Taxonomy

## 1. Taxonomy Version

Initial taxonomy version: `v0.1`

초기 목적은 완벽한 공정 taxonomy가 아니라, 합성 데이터와 EDA baseline을 만들기 위한 최소 패턴 집합을 정의하는 것이다.

## 2. Pattern Classes

### scratch

Revision note:

```text
Primary synthetic modes: spin_arc | radial
spin_arc: circular/arc scratch from spin or rotation-driven process steps
radial: center-to-edge scratch spreading from wafer center
straight chord-like scratch: not the default target pattern
```

길고 좁은 선형/곡선형 defect.

Synthetic parameters:

```text
start/end point
curvature
width
length
angle
intensity profile
fragmentation
stby coupling probability
```

Expected features:

```text
high elongation
high Hough/Radon response
localized angular direction
possible chip-wise missing if coupled with stby
```

### ring

물방울, 흐름성, 번짐성 defect.

Synthetic parameters:

```text
source point
ring radius
length
spread width
tail decay
anisotropy
multi-blob trail
```

Expected features:

```text
elongated blob
smooth intensity gradient
directional tail
less line-sharp than scratch
```

### edge

Revision note:

```text
Wafer edge can naturally have higher baseline fail bit density than wafer center.
Synthetic edge modeling should include both baseline edge lift and localized edge defects.
```

Wafer edge 근처의 ring, arc, local edge defect.

Synthetic parameters:

```text
radial band
angular span
edge distance
arc thickness
severity decay inward
```

Expected features:

```text
high edge density
large radius concentration
partial angular sector localization
```

### local

국소 hotspot 또는 cluster.

Synthetic parameters:

```text
center
radius
blob count
cluster compactness
peak severity
```

Expected features:

```text
connected component concentration
moderate/high local density
low global coverage
```

### random

구조가 약한 산발성 fail.

Synthetic parameters:

```text
global probability
spatial noise type
grade distribution
optional weak radial bias
```

Expected features:

```text
small connected components
low elongation
weak spatial autocorrelation
```

### shot_grid

Photo shot-relative repeated defect.

Synthetic parameters:

```text
shot_rows
shot_cols
anchor region inside shot field
affected slots
slot dropout
shot-relative intensity template
```

Expected features:

```text
fail-prone region repeats by row/column modulo shot layout
can appear as lower-left or edge-biased region inside each 3x3 shot
not centered on wafer polar coordinates
can overlap with edge, local, scratch, and stby
```

### stby_pattern

Stby Fail Chip의 chip-level missing pattern.

Synthetic parameters:

```text
mode: random | scratch_like | ring_like | edge_like | local_cluster
chip_selection_probability
pattern coupling with observed defect
mask dilation at chip granularity
```

Expected features:

```text
rectangular chip blocks
valid_test_mask = 0
can hide underlying grade/severity
can be meaningful pattern itself
```

## 3. Overlap Rule

Defect pattern은 서로 중첩될 수 있다. 따라서 정답 mask는 class별 binary/multi-label mask로 저장한다.

```text
pattern_masks[class, y, x] = 1 or 0
```

한 pixel/cell block에 여러 class가 동시에 1일 수 있다.

모델 학습 시 segmentation head는 softmax가 아니라 sigmoid multi-label output을 기본으로 한다.
