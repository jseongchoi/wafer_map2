# Pattern Taxonomy

이 문서는 WaferMap에서 쓰는 defect family와 CVAT label의 의미를 맞추기 위한 최소 taxonomy입니다. 공정 전체 taxonomy가 아니라 dataset generation과 model target 정의를 위한 실무 기준입니다.

## Model Family

| Family | Meaning | Current handling |
|---|---|---|
| `local` | compact hotspot, blob, cluster | CVAT/human asset primary |
| `scratch` | long linear or curved scratch | CVAT/human asset primary, procedural fallback |
| `ring` | full ring, partial ring, annulus, arc | CVAT/human asset primary |
| `edge` | edge band, edge sector, edge-localized density rise | procedural primary, optional CVAT asset |
| `shot_grid` | repeated shot-relative defect | procedural primary, optional CVAT asset |
| `random` | sparse unstructured fail baseline | procedural only |

These families map to `pattern_masks[class, y, x]` and are trained as multi-label sigmoid targets.

## CVAT Labels

CVAT labels are configured in [../configs/cvat/wafer_defect_labels.json](../configs/cvat/wafer_defect_labels.json). A CVAT label can map to an existing model family.

| CVAT label | Asset family | Notes |
|---|---|---|
| `local` | `local` | normal local/blob defect |
| `scratch` | `scratch` | scratch defect |
| `ring` | `ring` | ring or partial ring |
| `edge` | `edge` | edge band/sector |
| `shot_grid` | `shot_grid` | shot-relative repeated pattern |
| `random` | `random` | sparse fail baseline |
| `stby_blob` | `local` | STBY/missing-test mosaic blob, grade override 7 |

`stby_blob` aliases include `stby_fail` and `missing_test_blob`.

## STBY Terms

`stby_blob`:

- CVAT annotation label.
- Imported as `local` pattern asset for the current composer.
- Uses `grade_override: 7` so missing-test areas remain visible when composed.

`stby_pattern`:

- Synthetic generator concept.
- Describes chip-level missing-test regions through `stby_mask` and `valid_test_mask`.
- Not currently a primary segmentation target.

`stby_hidden_origin`:

- Review concept for cases where STBY/missing-test area may hide the real physical defect origin.
- Keep as metadata/review signal until there is enough evidence to model it directly.

## Global Patterns

`edge` and `ring` can cover large wafer regions. For these, annotation rules should prefer the visible defect band/arc rather than the full wafer area.

Recommended rules:

- For edge band defects, label the abnormal band/sector only.
- For ring defects, label the annulus or arc thickness, not the full disk inside the ring.
- If the pattern is too global to polygon accurately, start with a coarse polygon and rely on synthetic/procedural generation for variation.
- Keep ambiguous global patterns as separate CVAT task examples for review before adding new labels.

## Overlap Rule

Defect families may overlap.

```text
pattern_masks[class, y, x] = 0 or 1
```

One pixel can belong to multiple classes. The segmentation baseline therefore uses class-wise sigmoid targets, not softmax.

## Review Questions

During review, ask:

- Is the family correct?
- Is the mask too broad or too narrow?
- Did a global pattern get labeled as a local blob?
- Should this example become a procedural rule instead of a human asset?
- Does the label schema need a new CVAT label, or can it map to an existing asset family?
