# Compatibility Pattern Asset Editor

`scripts/run_pattern_asset_editor.py` is the original filename for the local browser segmentation tool engine. Keep it because tests and existing user commands may still call it.

For new operator-facing work, use:

```powershell
python scripts/run_segmentation_tool.py `
  --manifest configs/eval/real_unlabeled_synthetic_smoke.json `
  --sample-id real_like_synth_000000 `
  --assets-root data/pattern_assets
```

The compatibility filename and the new primary command save the same asset structure:

```text
data/pattern_assets/<family>/<asset_id>/
  grade.png
  mask.png
  preview.png
  metadata.json
```

Therefore assets saved through either command can be consumed by `compose_synthetic_from_assets.py` and `run_pattern_asset_pipeline.py`.

## Useful Existing Behaviors

- multi-family mask editing
- lasso selection and smart fit
- low-resolution interaction for large wafers
- prediction mask loading
- model proposal preview/apply
- grouped or split component saving

## Refactor Direction

The browser handler still lives in a large script. Move reusable pieces into `src/wafermap/assets/` only when a change needs sharing or test isolation. Keep `run_pattern_asset_editor.py` as a compatibility wrapper or engine filename even after internals move.
