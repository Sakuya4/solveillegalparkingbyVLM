import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QStackedWidget, QTabWidget
from illegal_parking.desktop_app import make_table
from illegal_parking.desktop_event_diagnostics import add_event_diagnostic_tab
from illegal_parking.project_snapshot import build_project_snapshot


def test_native_diagnostic_modes_separate_training_rankings_controller_and_contrasts():
    app = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    add_event_diagnostic_tab(tabs, build_project_snapshot(Path(__file__).resolve().parents[1]), make_table)
    views = tabs.findChild(QStackedWidget, "eventDiagnosticViews")
    selector = tabs.findChild(QComboBox, "eventDiagnosticView")
    assert views.widget(0).rowCount() == 8
    assert views.widget(0).item(0, 2).text() == "59"
    selector.setCurrentIndex(1)
    assert views.currentIndex() == 1 and views.currentWidget().rowCount() == 16
    assert views.currentWidget().item(0, 5).text() == "34"
    selector.setCurrentIndex(2)
    assert views.currentWidget().rowCount() == 8
    tabs.close()


def test_pending_native_diagnostics_do_not_display_invented_numbers():
    app = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    add_event_diagnostic_tab(tabs, {}, make_table)
    views = tabs.findChild(QStackedWidget, "eventDiagnosticViews")
    assert all(views.widget(index).rowCount() == 0 for index in range(3))
    tabs.close()
