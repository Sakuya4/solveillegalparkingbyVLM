import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QTableWidget, QTabWidget

from illegal_parking.desktop_app import make_table
from illegal_parking.desktop_presentation_results import add_presentation_tabs, number
from illegal_parking.project_snapshot import build_project_snapshot


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_window_results_keep_all_models_and_both_contrast_signs(app):
    snapshot = build_project_snapshot(Path(__file__).resolve().parents[1])
    tabs = QTabWidget()
    add_presentation_tabs(tabs, snapshot, make_table)
    assert tabs.count() == 3
    table = tabs.findChild(QTableWidget, "ptsWindowResults")
    assert table.rowCount() == 8 and table.columnCount() == 8
    assert table.item(0, 0).text() == "Source-frame"
    assert table.item(4, 0).text() == "Fixed-time"
    pairs = tabs.widget(0).findChildren(QTableWidget)[1]
    assert pairs.rowCount() == 4
    assert pairs.item(0, 2).text().startswith("+")
    assert pairs.item(2, 2).text().startswith("-")
    tabs.close()


def test_pending_results_do_not_invent_rows_or_zero_values(app):
    snapshot = build_project_snapshot(Path(__file__).resolve().parents[1])
    snapshot.update(presentation_events=None, presentation_controls=None)
    tabs = QTabWidget()
    add_presentation_tabs(tabs, snapshot, make_table)
    assert tabs.findChild(QTableWidget, "ptsEventResults").rowCount() == 0
    assert tabs.findChild(QTableWidget, "ptsControlResults").rowCount() == 0
    assert number(None) == "NA"
    tabs.close()


def test_event_results_distinguish_no_detection_and_budget_exceedance(app):
    snapshot = build_project_snapshot(Path(__file__).resolve().parents[1])
    outcome = {"event_recall": 0., "trigger_delay_sec": {"median": None}, "off_target_episodes_per_clip": .7}
    snapshot["presentation_events"] = {"models": [{"arm": "fixed", "feature": "candidate", "split": "iid",
                                                    "test_clips": 2, "selected_threshold": .5, "test": outcome,
                                                    "test_budget_exceeded": True}], "paired_recall_contrasts": []}
    snapshot["presentation_controls"] = None
    tabs = QTabWidget()
    add_presentation_tabs(tabs, snapshot, make_table)
    table = tabs.findChild(QTableWidget, "ptsEventResults")
    assert table.rowCount() == 1
    assert table.item(0, 5).text() == "0.000"
    assert table.item(0, 6).text() == "NA"
    assert table.item(0, 8).text() == "Yes"
    tabs.close()
