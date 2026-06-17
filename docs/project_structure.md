# Project Structure

권장 폴더 구조 초안은 다음과 같다. 이 문서는 최종 구현 목록이 아니라 프로젝트가 커질 때 유지하고 싶은 방향을 설명한다. 현재 실제 구현은 synthetic generator, feature extraction, grouping/stability/sweep/ablation report 중심으로 진행 중이다.

```text
WaferMap/
  README.md
  .gitignore
  pyproject.toml
  configs/
    synth/
    model/
    train/
    eval/
  data/
    raw/
    interim/
    processed/
    synthetic/
    labels/
  docs/
    problem_definition.md
    data_schema.md
    pattern_taxonomy.md
    synthetic_data_plan.md
    validation_protocol.md
    modeling_strategy.md
    methodology_validation.md
    milestone_review.md
    scale_pilot_review.md
    revised_physical_assumptions.md
    review_presets.md
    roadmap.md
    project_structure.md
  notebooks/
    00_data_exploration.ipynb
    01_synthetic_generation.ipynb
    02_baseline_features.ipynb
    03_similarity_search.ipynb
    04_model_training.ipynb
  src/
    wafermap/
      data/
        schema.py
        io.py
        preprocess.py
        tiling.py
      synth/
        wafer_geometry.py
        pattern_generators.py
        stby.py
        quantization.py
        export.py
      features/
        radial_angular.py
        connected_components.py
        line_scores.py
        wafer_vector.py
      models/
        unet.py
        segformer.py
        heads.py
      training/
        losses.py
        train_segmentation.py
      evaluation/
        metrics.py
        realism_metrics.py
      reporting/
        clock_position.py
        defect_summary.py
      viz/
        render.py
        overlays.py
  scripts/
    generate_synthetic.py
    validate_synthetic.py
    extract_features.py
    analyze_fbm_grouping.py
    evaluate_grouping_stability.py
    sweep_grouping_parameters.py
    evaluate_feature_ablation.py
    evaluate_methodology.py
    make_report.py
    make_progress_briefing.py
  tests/
    test_geometry.py
    test_synthetic_generator.py
    test_features.py
  outputs/
    figures/
    predictions/
    reports/
```

## Data Policy

`data/raw`는 실제 데이터를 넣는 위치가 아니라 local-only placeholder다. 실제 보안 데이터는 git에 commit하지 않는다.

권장 `.gitignore` 정책:

```text
data/raw/**
data/interim/**
data/processed/**
data/synthetic/**
outputs/**
!**/.gitkeep
```

합성 데이터도 크기가 커질 수 있으므로 기본적으로 git에는 code/config/docs만 관리한다.
