# Scripts Command Map

This folder contains stable command-line entrypoints. It is intentionally flat for backward compatibility with existing tests, docs, and user commands.

If you add a new script, register it here and keep reusable logic in `src/wafermap/`.

## Primary In-Repo Segmentation Pipeline

Run these first for the current product direction.

| Script | Purpose |
|---|---|
| `analyze_png_raw_folders.py` | Build a real/unlabeled manifest and reports from raw PNG folders. |
| `run_segmentation_tool.py` | Open the local browser segmentation tool and save reusable pattern assets. |
| `build_pattern_asset_report.py` | Generate a review report for saved pattern assets. |
| `compose_synthetic_from_assets.py` | Compose pattern assets and procedural defects onto base wafers. |
| `run_pattern_asset_pipeline.py` | Run composition, readiness, smoke validation, and report generation end to end. |
| `train_unet_segmentation.py` | Train or dependency-check the coordinate-aware small U-Net entrypoint. |

## Data And Synthetic Generation

| Script | Purpose |
|---|---|
| `generate_synthetic.py` | Generate procedural synthetic wafer samples. |
| `validate_synthetic.py` | Validate generated synthetic samples. |
| `build_segmentation_readiness.py` | Build segmentation manifests, readiness metrics, and galleries. |
| `train_segmentation_smoke.py` | Smoke-test segmentation tensor/loss wiring. |
| `train_embedding_smoke.py` | Smoke-test simple embedding retrieval wiring. |

## Real Wafer Diagnostics

| Script | Purpose |
|---|---|
| `extract_real_unlabeled_features.py` | Load real/unlabeled manifests and extract real-like sample features. |
| `extract_features.py` | Extract features from synthetic samples. |
| `interpret_fbm.py` | Produce interpretation outputs from feature/sanity/neighbor artifacts. |
| `summarize_defect_locations.py` | Summarize defect location distributions. |
| `summarize_expert_review.py` | Aggregate expert review CSV feedback. |
| `make_expert_review_template.py` | Create review templates. |

## Modeling And Retrieval Diagnostics

These are secondary to the segmentation dataset pipeline.

| Script | Purpose |
|---|---|
| `train_cpu_encoder_model.py` | Train lightweight CPU encoder baseline. |
| `score_unlabeled_cpu_encoder.py` | Score unlabeled wafer batches with CPU encoder artifacts. |
| `evaluate_defect_feature_retrieval.py` | Evaluate defect feature retrieval behavior. |
| `evaluate_interest_retrieval.py` | Evaluate interest/retrieval targets. |
| `evaluate_retrieval_confidence.py` | Evaluate confidence estimates for retrieval. |
| `evaluate_resize_benchmark.py` | Benchmark resize-only representation behavior. |

## Research / Historical Evaluation

Keep these for experiment traceability, but do not treat them as the main workflow.

| Script | Purpose |
|---|---|
| `analyze_fbm_grouping.py` | Analyze grouping behavior for FBM synthetic runs. |
| `evaluate_curve_proposals.py` | Evaluate curve/ring proposal ideas. |
| `evaluate_feature_ablation.py` | Evaluate feature ablations. |
| `evaluate_grouping_stability.py` | Evaluate grouping stability. |
| `evaluate_methodology.py` | Evaluate methodology probes. |
| `evaluate_patch_proposals.py` | Evaluate patch proposal behavior. |
| `sweep_grouping_parameters.py` | Sweep grouping parameters. |

## Compatibility

| Script | Purpose |
|---|---|
| `run_pattern_asset_editor.py` | Backward-compatible filename for the local segmentation tool engine. |

## Reports And Project Status

| Script | Purpose |
|---|---|
| `make_report.py` | Generate synthetic experiment reports. |
| `make_progress_briefing.py` | Generate progress briefing artifacts. |
| `make_leader_status_report.py` | Generate leader/status reports. |
| `run_pre_real_readiness.py` | Run pre-real readiness pipeline. |

## Rules For New Scripts

- Put core logic in `src/wafermap/`.
- Keep scripts thin: parse arguments, call library functions, print output paths.
- If the script is operator-facing, add it to `README.md`, `docs/operator_manual.md`, or the relevant workflow doc.
- If the script is experimental, mark it as research/historical here.
- Do not break existing script filenames without keeping wrappers.
