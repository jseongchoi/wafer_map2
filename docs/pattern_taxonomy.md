# Pattern Taxonomy

이 문서는 WaferMap에서 쓰는 defect family의 의미를 맞추기 위한 최소 taxonomy입니다. 공정 전체 taxonomy가 아니라 dataset generation과 model target 정의를 위한 실무 기준입니다.

## Model Families

| Family | Meaning | Source policy |
|---|---|---|
| `local` | compact hotspot, blob, cluster | human asset primary |
| `scratch` | long linear or curved scratch | human asset primary, procedural fallback |
| `ring` | full ring, partial ring, annulus, arc | human asset primary |
| `edge` | edge band, edge sector, edge-localized density rise | procedural primary, optional human asset |
| `shot_grid` | repeated shot-relative defect | procedural primary, optional human asset |
| `random` | sparse random fail points | procedural only baseline |

These families map to `pattern_masks[class, y, x]` and are trained as multi-label sigmoid targets.

## Segmentation Tool Labels

The local tool exposes the same family names as the model target channels.

| Tool family | Asset family | Notes |
|---|---|---|
| `local` | `local` | compact blob or local cluster |
| `scratch` | `scratch` | scratch-like line/arc |
| `ring` | `ring` | ring, partial ring, annulus |
| `edge` | `edge` | abnormal edge band/sector |
| `shot_grid` | `shot_grid` | repeated shot-relative defect |
| `random` | `random` | sparse unstructured fail pattern |

## STBY / Missing Test

STBY is represented by semantic arrays, not by a primary defect family by default.

- Describes chip-level missing-test regions through `stby_mask` and `valid_test_mask`.
- Not currently a primary segmentation target.
- If a missing-test mosaic is later treated as a trainable defect, add a dedicated model class rather than hiding it inside `local`.

## Ambiguity Rules

`edge` and `ring` can cover large wafer regions. For these, annotation rules should prefer the visible defect band/arc rather than the full wafer area.

`scratch` and `ring` can overlap. If a long arc is concentric with wafer center, prefer `ring`. If it cuts across radius or follows process scratch geometry, prefer `scratch`.

`shot_grid` requires repeated shot-relative structure. A single hotspot is `local`.

Keep ambiguous global patterns as separate review examples before adding new families.

## Target Contract

```text
pattern_masks[class, y, x] = 0 or 1
```

One pixel can belong to multiple classes. The segmentation baseline therefore uses class-wise sigmoid targets, not softmax.

## Review Questions

- Is the selected family observable from the wafer map?
- Is the mask too broad or too narrow?
- Should this be a procedural generator instead of a human asset?
- Does the model need a new target family, or can this map to an existing family?
