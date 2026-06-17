import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_review_template_strips_reference_labels_and_adds_blank_fields():
    module = _load_script("make_expert_review_template")
    rows, warnings = module.build_template_rows(
        [
            {
                "query_sample_id": "q1",
                "rank": "1",
                "neighbor_sample_id": "n1",
                "distance": "1.23456789",
                "label_edge": "1",
            }
        ]
    )

    assert rows[0]["review_case_id"] == "q1__rank1__n1"
    assert rows[0]["distance"] == "1.23457"
    assert rows[0]["reviewer_decision"] == ""
    assert rows[0]["query_defect_family"] == ""
    assert rows[0]["neighbor_defect_family"] == ""
    assert rows[0]["retrieval_failure_mode"] == ""
    assert rows[0]["next_action"] == ""
    assert "label_edge" not in rows[0]
    assert warnings


def test_review_summary_computes_query_topk_acceptance():
    module = _load_script("summarize_expert_review")
    rows = [
        {
            "query_sample_id": "q1",
            "rank": "1",
            "neighbor_sample_id": "n1",
            "reviewer_decision": "mismatch",
            "query_defect_family": "local",
            "neighbor_defect_family": "edge",
            "dominant_defect": "edge",
            "clock_position_match": "no",
            "missed_major_defect": "yes",
            "retrieval_failure_mode": "missed_local",
            "next_action": "tune_observable_feature",
            "safe_comment": "",
        },
        {
            "query_sample_id": "q1",
            "rank": "2",
            "neighbor_sample_id": "n2",
            "reviewer_decision": "same_family",
            "query_defect_family": "edge",
            "neighbor_defect_family": "edge",
            "dominant_defect": "edge",
            "clock_position_match": "yes",
            "missed_major_defect": "no",
            "retrieval_failure_mode": "none",
            "next_action": "keep_baseline",
            "safe_comment": "",
        },
        {
            "query_sample_id": "q2",
            "rank": "1",
            "neighbor_sample_id": "n3",
            "reviewer_decision": "partial_match",
            "query_defect_family": "local",
            "neighbor_defect_family": "local",
            "dominant_defect": "local",
            "clock_position_match": "partial",
            "missed_major_defect": "no",
            "retrieval_failure_mode": "wrong_clock_position",
            "next_action": "add_location_aware_feature",
            "safe_comment": "",
        },
    ]

    metrics = module.summarize_rows(rows, top_k=2)

    assert metrics["valid_review_rows"] == 3
    assert metrics["same_family_rate"] == 1 / 3
    assert metrics["accepted_match_rate"] == 2 / 3
    assert metrics["query_topk_same_family_rate"] == 0.5
    assert metrics["query_topk_accept_rate"] == 1.0
    assert metrics["query_missed_major_defect_rate"] == 0.5
    assert metrics["retrieval_failure_mode_counts"]["missed_local"] == 1
    assert metrics["query_defect_family_counts"]["local"] == 2
    assert metrics["neighbor_defect_family_counts"]["edge"] == 2
    assert metrics["next_action_counts"]["add_location_aware_feature"] == 1
    assert metrics["next_action_queue"][0]["next_action"] in {
        "add_location_aware_feature",
        "tune_observable_feature",
    }


def test_review_summary_flags_sensitive_comments():
    module = _load_script("summarize_expert_review")
    rows = [
        {
            "query_sample_id": "q1",
            "rank": "1",
            "neighbor_sample_id": "n1",
            "reviewer_decision": "same_family",
            "query_defect_family": "shot_grid",
            "neighbor_defect_family": "shot_grid",
            "dominant_defect": "shot_grid",
            "clock_position_match": "yes",
            "missed_major_defect": "no",
            "retrieval_failure_mode": "none",
            "next_action": "keep_baseline",
            "safe_comment": "looks like lot ABC123 from C:/secure/raw.npy",
        }
    ]

    metrics = module.summarize_rows(rows, top_k=1)

    issues = {item["issue"] for item in metrics["sensitive_comment_flags"]}
    assert "lot_identifier" in issues
    assert "windows_path" in issues
