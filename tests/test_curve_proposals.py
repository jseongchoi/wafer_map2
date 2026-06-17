import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_curve_proposals.py"
    spec = importlib.util.spec_from_file_location("evaluate_curve_proposals", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_circular_bin_mask_wraps_around_zero():
    module = _load_script()
    indices = np.arange(8, dtype=np.int32)

    mask = module.circular_bin_mask(indices, 6, 10, 8)

    assert mask.tolist() == [True, True, False, False, False, False, True, True]


def test_build_candidates_finds_high_scoring_partial_ring():
    module = _load_script()
    polar_score = np.zeros((8, 12), dtype=np.float32)
    polar_score[3, 4:8] = 10.0

    candidates = module.build_candidates("ring", polar_score, top_k=1)

    assert candidates[0].kind == "partial_ring"
    assert candidates[0].r0 <= 3 < candidates[0].r1
