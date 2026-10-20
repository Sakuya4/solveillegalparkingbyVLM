import pytest

pytest.importorskip("PySide6")
from illegal_parking.desktop_context_training import comparison_rows


def model(key, selected=True, exceeded=False):
    return {"model": key, "selected_threshold": .8 if selected else None,
            "test": {"summary": {"event_recall": .3, "off_target_episodes_per_clip": .4, "trigger_delay_sec": {"median": -.2}},
                     "pre_event": {"macro_positive_fraction": .1, "clips_with_support": 70}} if selected else None,
            "test_episode_cap_exceeded": exceeded, "test_prefix_cap_exceeded": False}


def test_native_comparison_retains_unselected_policies_and_budget_exceedance():
    rows, _, _ = comparison_rows({"models": [model("anchored_candidate_iid", selected=False), model("continuous_global_geographic", exceeded=True)], "contrasts": []}, None)
    assert rows[0][-1] == "Unselected" and rows[0][4] == "NA"
    assert rows[1][-1] == "Exceeded" and rows[1][2] == "geo"
    assert rows[1][-2] == "-0.200"


def test_native_paired_changes_are_percentage_points_not_raw_proportions():
    _, rows, costs = comparison_rows({"models": [], "contrasts": [{"feature": "global", "split": "iid", "paired_interval": {
        "difference": -.2, "lower_95": -.3, "upper_95": -.1, "clusters": 100}}]},
        {"results": [{"model": "continuous_global_iid", "median_ms": 1, "p95_ms": 2, "calls": 1000}]})
    assert rows[0][2] == "-20.000" and rows[0][3] == "[-30.00, -10.00]"
    assert costs[0][-1] == 1000


def test_native_context_views_switch_without_completed_study(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QComboBox, QStackedWidget, QTabWidget, QTableWidget
    from illegal_parking.desktop_context_training import add_context_training_tab
    application = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    add_context_training_tab(tabs, {}, lambda headers, rows: QTableWidget(len(rows), len(headers)))
    assert tabs.count() == 1
    selector = tabs.findChild(QComboBox, "contextTrainingView")
    views = tabs.findChild(QStackedWidget, "contextTrainingViews")
    assert views.count() == 3
    for index in range(3):
        selector.setCurrentIndex(index)
        assert views.currentIndex() == index
        assert views.currentWidget().rowCount() == 0
    tabs.close()
    tabs.deleteLater()
    application.processEvents()
