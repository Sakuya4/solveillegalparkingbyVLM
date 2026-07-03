from illegal_parking.metrics import ConfusionCounts, evaluate_binary_events


def test_evaluate_binary_events_counts_confusion_matrix():
    result = evaluate_binary_events(
        y_true=[True, True, False, False],
        y_pred=[True, False, True, False],
    )

    assert result.counts == ConfusionCounts(tp=1, fp=1, fn=1, tn=1)


def test_evaluate_binary_events_computes_core_rates():
    result = evaluate_binary_events(
        y_true=[True, True, True, False, False],
        y_pred=[True, True, False, True, False],
    )

    assert result.precision == 2 / 3
    assert result.recall == 2 / 3
    assert result.false_positive_rate == 1 / 2


def test_evaluate_binary_events_handles_zero_denominators():
    result = evaluate_binary_events(y_true=[False, False], y_pred=[False, False])

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.false_positive_rate == 0.0
