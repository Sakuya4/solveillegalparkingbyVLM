import os
import sys
import subprocess
import json
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QEvent, QProcess
from PySide6.QtTest import QTest
from illegal_parking.desktop_app import TrafficResearchWindow


@pytest.fixture(scope="session")
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application
    application.closeAllWindows()
    application.processEvents()


@pytest.fixture
def window(qt_application):
    application = qt_application
    widget = TrafficResearchWindow(Path(__file__).resolve().parents[1])
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
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


def test_loading_another_video_clears_previous_event_evidence(window):
    window.evidence_text.setText("Previous event result")
    window.load_demo()
    assert "Previous event result" not in window.evidence_text.text()
    assert window.report is None


def test_evidence_selection_loads_real_images(window):
    for index in range(3):
        window.evidence_choice.setCurrentIndex(index)
        assert not window.evidence_image.pixmap().isNull()
        assert window.evidence_text.text()


def test_completed_analysis_can_be_reopened_without_inference(window, tmp_path):
    report = {
        "input": str(window.input_path), "output_video": str(window.input_path),
        "events": [], "runtime": {"requested_provider": "CPUExecutionProvider"},
        "elapsed_sec": 1.0,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    window.load_report(path)
    assert window.report["events"] == []
    assert window.actions_by_name["Export results"].isEnabled()
    assert window.video.current_image is not None


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


@pytest.mark.parametrize("page", [0, 2, 4])
def test_native_launcher_exits_cleanly_after_rendering(tmp_path, page):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / f"page_{page}.png"
    result = subprocess.run([sys.executable, "scripts/run_desktop.py", "--page", str(page), "--screenshot", str(output)], cwd=root, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert output.stat().st_size > 10000


def test_windows_pythonw_launcher_can_complete_analysis(tmp_path):
    root = Path(__file__).resolve().parents[1]
    interpreter = Path(sys.executable).with_name("pythonw.exe")
    if not interpreter.is_file() or not (root / "yolov8n.pt").is_file():
        pytest.skip("Windows GUI interpreter and local detector are required")
    output = tmp_path / "pythonw_analysis.png"
    result = subprocess.run([str(interpreter), "scripts/run_desktop.py", "--analyze", "--screenshot", str(output)], cwd=root, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert output.stat().st_size > 10000
