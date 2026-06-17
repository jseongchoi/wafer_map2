import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _feature_row() -> dict[str, str]:
    return {
        "sample_id": "s1",
        "actual_net_die": "4",
        "cluster_id": "0",
        "pca_0": "0.0",
        "pca_1": "0.0",
        "total_fail_density": "0.1",
        "stby_ratio": "0.0",
        "label_edge": "1",
        "edge_mask_ratio": "0.2",
        "polar_r00_a00_severity": "99.0",
        "stby_polar_r00_a00_ratio": "99.0",
    }


def test_global_ablation_selector_uses_compact_features_only():
    module = _load_script("evaluate_feature_ablation")

    names = module.observable_feature_names(_feature_row())

    assert names == ["total_fail_density", "stby_ratio"]


def test_methodology_selector_uses_compact_features_only():
    module = _load_script("evaluate_methodology")

    names = module.observable_feature_names([_feature_row()])

    assert names == ["total_fail_density", "stby_ratio"]
