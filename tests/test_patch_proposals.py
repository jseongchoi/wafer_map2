import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_patch_proposals.py"
    spec = importlib.util.spec_from_file_location("evaluate_patch_proposals", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_top_windows_prefers_high_score_region():
    module = _load_script()
    score = np.zeros((6, 6), dtype=np.float32)
    score[3:5, 3:5] = 10.0

    windows = module.select_top_windows(score, window_cells=2, top_k=1)

    assert len(windows) == 1
    assert windows[0].y0 == 3
    assert windows[0].x0 == 3


def test_proposal_recall_counts_target_pixels_inside_boxes_once():
    module = _load_script()
    target = np.zeros((8, 8), dtype=bool)
    target[2:6, 2:6] = True
    boxes = [
        (0, 4, 0, 4, 1.0),
        (3, 8, 3, 8, 0.5),
    ]

    recall = module.proposal_recall(target, boxes)

    expected = np.zeros_like(target)
    expected[0:4, 0:4] = True
    expected[3:8, 3:8] = True
    assert recall == np.logical_and(target, expected).sum() / target.sum()
