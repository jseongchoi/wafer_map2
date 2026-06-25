import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wafermap.data import PATTERN_CLASSES
from wafermap.synth import SyntheticConfig, generate_sample


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pattern_asset_extraction_and_max_composition(tmp_path):
    editor = _load_script("run_pattern_asset_editor.py")
    composer = _load_script("compose_synthetic_from_assets.py")
    report = _load_script("build_pattern_asset_report.py")
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=23), 0)
    sample.severity[:] = 0
    sample.pattern_masks[:] = 0
    y0, x0 = np.argwhere(sample.valid_test_mask > 0)[10]
    sample.severity[y0 : y0 + 3, x0 : x0 + 3] = 7
    mask = np.zeros(sample.shape, dtype=bool)
    mask[y0 : y0 + 3, x0 : x0 + 3] = True
    assets_root = tmp_path / "pattern_assets"

    saved = editor.save_pattern_assets(
        sample=sample,
        masks_by_family={"local": mask},
        assets_root=assets_root,
        margin_ratio=0.20,
    )

    assert len(saved) == 1
    asset_dir = Path(saved[0]["path"])
    assert asset_dir.parent.name == "local"
    assert (asset_dir / "grade.png").exists()
    assert (asset_dir / "mask.png").exists()
    assert (asset_dir / "preview.png").exists()
    assert (asset_dir / "metadata.json").exists()
    with Image.open(asset_dir / "mask.png") as image:
        assert np.asarray(image).sum() > 0
    scanned = editor.scan_pattern_assets(assets_root)
    assert scanned[0]["asset_id"] == asset_dir.name
    assert scanned[0]["valid"] is True

    report_path = tmp_path / "asset_report.html"
    report.main(["--assets-root", str(assets_root), "--out", str(report_path)])
    report_text = report_path.read_text(encoding="utf-8")
    assert "FBM Pattern Asset Library Report" in report_text
    assert asset_dir.name in report_text

    base = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=29), 0)
    base.severity[:] = 0
    base.pattern_masks[:] = 0
    assets = composer.load_assets(assets_root)
    composed = composer.compose_sample(base, assets, composer.random.Random(5), 1, "composed_unit")

    local_idx = PATTERN_CLASSES.index("local")
    assert composed.severity.max() == 7
    assert composed.pattern_masks[local_idx].sum() > 0
    assert composed.metadata["composition_rule"] == "max"
    assert composed.metadata["multi_label"] is True
    assert composed.metadata["placement_mode"] == "source_jitter"
    assert composed.metadata["placed_assets"][0]["placement_mode"] == "source_jitter"


def test_procedural_families_generate_labeled_masks_without_assets():
    composer = _load_script("compose_synthetic_from_assets.py")
    base = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=30), 0)
    base.severity[:] = 0
    base.pattern_masks[:] = 0

    random_only = composer.compose_sample(
        base,
        [],
        composer.random.Random(11),
        0,
        "procedural_random",
        procedural_families=("random",),
    )
    random_idx = PATTERN_CLASSES.index("random")
    assert random_only.pattern_masks[random_idx].sum() > 0
    assert random_only.metadata["procedural_patterns"][0]["family"] == "random"

    old_scratch_probability = composer.PROCEDURAL_PROBABILITIES["scratch"]
    composer.PROCEDURAL_PROBABILITIES["scratch"] = 1.0
    try:
        scratch_only = composer.compose_sample(
            base,
            [],
            composer.random.Random(13),
            0,
            "procedural_scratch",
            procedural_families=("scratch",),
        )
    finally:
        composer.PROCEDURAL_PROBABILITIES["scratch"] = old_scratch_probability
    scratch_idx = PATTERN_CLASSES.index("scratch")
    assert scratch_only.pattern_masks[scratch_idx].sum() > 0
    assert scratch_only.metadata["procedural_patterns"][0]["family"] == "scratch"

    old_probability = composer.PROCEDURAL_PROBABILITIES["edge"]
    composer.PROCEDURAL_PROBABILITIES["edge"] = 1.0
    try:
        edge_only = composer.compose_sample(
            base,
            [],
            composer.random.Random(12),
            0,
            "procedural_edge",
            procedural_families=("edge",),
        )
    finally:
        composer.PROCEDURAL_PROBABILITIES["edge"] = old_probability
    edge_idx = PATTERN_CLASSES.index("edge")
    assert edge_only.pattern_masks[edge_idx].sum() > 0
    assert edge_only.metadata["procedural_patterns"][0]["source"] == "procedural"


def test_pattern_asset_save_mode_keeps_disconnected_family_together_by_default(tmp_path):
    editor = _load_script("run_pattern_asset_editor.py")
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=31), 0)
    sample.severity[:] = 0
    coords = np.argwhere(sample.valid_test_mask > 0)
    y1, x1 = coords[5]
    y2, x2 = coords[-6]
    sample.severity[y1, x1] = 6
    sample.severity[y2, x2] = 7
    disconnected = np.zeros(sample.shape, dtype=bool)
    disconnected[y1, x1] = True
    disconnected[y2, x2] = True

    grouped = editor.save_pattern_assets(
        sample=sample,
        masks_by_family={"ring": disconnected},
        assets_root=tmp_path / "grouped",
        margin_ratio=0.20,
    )
    split = editor.save_pattern_assets(
        sample=sample,
        masks_by_family={"ring": disconnected},
        assets_root=tmp_path / "split",
        margin_ratio=0.20,
        split_components=True,
    )

    assert len(grouped) == 1
    assert len(split) == 2
    assert Path(grouped[0]["path"]).parent.name == "ring"
    assert {Path(item["path"]).parent.name for item in split} == {"ring"}


def test_prediction_mask_json_can_prefill_editor_masks(tmp_path):
    editor = _load_script("run_pattern_asset_editor.py")
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(
        json.dumps(
            {
                "schema_version": "fbm_prediction_masks/v1",
                "samples": [
                    {
                        "sample_id": "wafer_a",
                        "masks": {
                            "local": [[0, 2], [5, 1]],
                            "scratch": [],
                            "ring": [],
                            "edge": [],
                            "shot_grid": [],
                            "random": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    masks = editor.load_prediction_masks(prediction_path, "wafer_a", (2, 3))

    assert masks["local"] == [[0, 2], [5, 1]]
    assert masks["ring"] == []


def test_model_proposal_json_loads_as_editor_proposals(tmp_path):
    editor = _load_script("run_pattern_asset_editor.py")
    proposal_path = tmp_path / "proposals.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": "fbm_model_proposals/v1",
                "source": "umap_segment_v0",
                "samples": [
                    {
                        "sample_id": "wafer_a",
                        "shape": [2, 3],
                        "proposals": [
                            {
                                "family": "edge",
                                "rle": [[1, 2]],
                                "confidence": 0.91,
                                "description": "external edge segment",
                                "parameters": {"embedding_cluster": 4},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proposals = editor.load_model_proposals(proposal_path, "wafer_a", (2, 3), (4, 6))

    assert len(proposals) == 1
    assert proposals[0]["family"] == "edge"
    assert proposals[0]["confidence"] == 0.91
    assert proposals[0]["pixel_count"] == 8
    assert proposals[0]["parameters"]["source"] == "umap_segment_v0"
    assert proposals[0]["parameters"]["embedding_cluster"] == 4
    mask = editor.rle_to_mask(proposals[0]["rle"], (4, 6))
    assert int(mask.sum()) == 8


def test_pattern_asset_editor_downsamples_for_editing_and_upsamples_masks():
    editor = _load_script("run_pattern_asset_editor.py")

    assert editor.editor_shape((2000, 1000), 768) == (768, 384)
    assert editor.editor_shape((100, 80), 768) == (100, 80)
    assert editor.editor_shape((100, 80), 0) == (100, 80)

    edit_mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    source_mask = editor.resize_mask_nearest(edit_mask, (4, 4))

    assert source_mask.shape == (4, 4)
    assert source_mask[:2, :2].all()
    assert source_mask[2:, 2:].all()
    assert not source_mask[:2, 2:].any()
    assert not source_mask[2:, :2].any()


def test_pattern_asset_editor_auto_proposes_global_geometry():
    editor = _load_script("run_pattern_asset_editor.py")
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=80, chip_width=6, chip_height=6, seed=51), 0)
    sample.severity[:] = 0
    sample.pattern_masks[:] = 0
    radius, _theta = editor.polar_geometry(sample.wafer_mask > 0)
    valid = sample.valid_test_mask > 0
    sample.severity[valid & (radius >= 0.90)] = 6
    sample.severity[valid & (np.abs(radius - 0.52) <= 0.015)] = 7

    proposals = editor.analyze_pattern_proposals(sample, sample.shape, min_grade=5)
    families = {proposal["family"] for proposal in proposals}

    assert {"edge", "ring"}.issubset(families)
    for proposal in proposals:
        mask = editor.rle_to_mask(proposal["rle"], sample.shape)
        assert mask.sum() == proposal["pixel_count"]
        assert proposal["confidence"] > 0


def test_pattern_asset_editor_exposes_client_side_color_schemes():
    editor = _load_script("run_pattern_asset_editor.py")

    assert 'id="colorScheme"' in editor.EDITOR_HTML
    assert "const COLOR_SCHEMES" in editor.EDITOR_HTML
    assert "function renderBase" in editor.EDITOR_HTML
    assert "sample.stby_mask_b64" in editor.EDITOR_HTML
    assert 'id="loadModelProposals"' in editor.EDITOR_HTML
    assert "function loadModelProposals" in editor.EDITOR_HTML
    assert "<summary>Proposals</summary>" in editor.EDITOR_HTML
    assert "<summary>Output</summary>" in editor.EDITOR_HTML


def test_pattern_asset_pipeline_writes_stby_excluded_manifest_and_project_report(tmp_path):
    editor = _load_script("run_pattern_asset_editor.py")
    pipeline = _load_script("run_pattern_asset_pipeline.py")
    sample = generate_sample(SyntheticConfig(count=1, target_net_die=40, chip_width=6, chip_height=6, seed=41), 0)
    sample.severity[:] = 0
    sample.pattern_masks[:] = 0
    y0, x0 = np.argwhere(sample.valid_test_mask > 0)[12]
    sample.severity[y0 : y0 + 3, x0 : x0 + 3] = 7
    mask = np.zeros(sample.shape, dtype=bool)
    mask[y0 : y0 + 3, x0 : x0 + 3] = True
    assets_root = tmp_path / "pattern_assets"
    editor.save_pattern_assets(
        sample=sample,
        masks_by_family={"local": mask},
        assets_root=assets_root,
        margin_ratio=0.20,
    )

    base_dir = tmp_path / "base"
    pipeline.load_script("compose_synthetic_from_assets.py").write_sample(sample, base_dir)
    report = tmp_path / "project_report.html"
    pipeline.main(
        [
            "--base-sample-dir",
            str(base_dir),
            "--assets-root",
            str(assets_root),
            "--composed-dir",
            str(tmp_path / "composed"),
            "--work-dir",
            str(tmp_path / "work"),
            "--report-out",
            str(report),
            "--count",
            "4",
            "--assets-per-wafer",
            "1",
            "--output-size",
            "12",
            "--embedding-dim",
            "3",
            "--top-k",
            "2",
        ]
    )

    metrics = json.loads((tmp_path / "work" / "asset_segmentation_readiness_metrics.json").read_text(encoding="utf-8"))
    assert "stby_pattern" not in metrics["target_channels"]
    assert "local" in metrics["target_channels"]
    assert report.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "WaferMap Hybrid Synthetic Data Report" in report_text
    assert "procedural" in report_text
