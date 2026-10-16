from scripts.evaluate_vlm_review_dispositions import summarize_support


def test_unresolved_reviews_do_not_become_true_negatives_or_perfect_precision():
    rows = [{"expected_collision": True}, {"expected_collision": False}]
    result = summarize_support(rows, [False, False])
    assert result["supported_positive_precision"] is None
    assert result["support_coverage_of_positive_cases"] == 0
    assert result["unresolved_cases"] == 2
    assert "accuracy" not in result
    accepted = summarize_support(rows, [True, True])
    assert accepted["supported_positive_precision"] == 0.5
    assert accepted["supported_false_cases"] == 1
