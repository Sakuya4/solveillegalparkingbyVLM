import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess
from PySide6.QtTest import QTest
from illegal_parking.desktop_app import TrafficResearchWindow


@pytest.fixture
def window():
    application = QApplication.instance() or QApplication([])
    widget = TrafficResearchWindow(Path(__file__).resolve().parents[1])
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    application.processEvents()


def test_desktop_loads_reference_video_and_all_research_pages(window):
    assert window.video.current_image is not None
    assert window.video.frame_count > 0
    assert window.pages.count() == 5
    for index in range(5):
        window.navigation.setCurrentRow(index)
        assert window.pages.currentIndex() == index
    assert window.provider.currentData() == "CPUExecutionProvider"


def test_video_seek_and_playback_preserve_frame_state(window):
    window.video.seek(10)
    assert window.video.frame_index == 10
    window.video.toggle_playback()
    assert window.video.timer.isActive()
    window.video.toggle_playback()
    assert not window.video.timer.isActive()
    assert not window.actions_by_name["Cancel"].isEnabled()
    assert not window.actions_by_name["Export results"].isEnabled()


def test_evidence_selection_loads_real_images(window):
    for index in range(3):
        window.evidence_choice.setCurrentIndex(index)
        assert not window.evidence_image.pixmap().isNull()
        assert window.evidence_text.text()


def test_cancel_analysis_terminates_the_active_worker(window):
    process = QProcess(window)
    window.job = process
    process.start(sys.executable, ["-c", "import time; time.sleep(60)"])
    assert process.waitForStarted(3000)
    window.cancel_analysis()
    for _ in range(40):
        if process.state() == QProcess.ProcessState.NotRunning:
            break
        QTest.qWait(100)
    assert process.state() == QProcess.ProcessState.NotRunning
    assert window.cancelled is True
