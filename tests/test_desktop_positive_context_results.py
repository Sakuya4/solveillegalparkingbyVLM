import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTableWidget, QTabWidget
from illegal_parking.desktop_app import make_table, TrafficResearchWindow
from illegal_parking.desktop_positive_context_results import add_positive_context_tab
from illegal_parking.project_snapshot import build_project_snapshot


def test_native_comparison_keeps_caps_controls_and_unsupported_prefixes():
    app = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    add_positive_context_tab(tabs, build_project_snapshot(Path(__file__).resolve().parents[1]), make_table)
    result = tabs.findChild(QTableWidget, "positiveContextResults")
    assert result.rowCount() == 8 and result.columnCount() == 8
    assert result.item(0, 2).text() == "0.450" and result.item(0, 3).text() == "0.140"
    assert result.item(0, 6).text() == "Yes" and result.item(1, 6).text() == "No"
    controls = tabs.findChild(QTableWidget, "positiveContextControls")
    assert controls.rowCount() == 4 and controls.item(0, 4).text() == "23"
    tabs.close()


def test_pending_native_study_is_not_a_zero_measurement():
    app = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    add_positive_context_tab(tabs, {}, make_table)
    assert tabs.findChild(QTableWidget, "positiveContextResults").rowCount() == 0
    tabs.close()


def test_all_policy_rows_fit_the_reference_desktop_viewport():
    app = QApplication.instance() or QApplication([])
    window = TrafficResearchWindow(Path(__file__).resolve().parents[1])
    window.navigation.setCurrentRow(1)
    tabs = window.pages.widget(1).findChild(QTabWidget)
    tabs.setCurrentIndex(6)
    window.show()
    app.processEvents()
    for name in ("positiveContextResults", "positiveContextControls"):
        table = tabs.findChild(QTableWidget, name)
        last = table.visualItemRect(table.item(table.rowCount() - 1, 0))
        assert last.bottom() < table.viewport().height()
    window.close()
