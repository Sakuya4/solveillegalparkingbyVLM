import os
import sys
import subprocess
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QTabWidget
from PySide6.QtCore import QCoreApplication, QEvent, QProcess
from PySide6.QtTest import QTest
from illegal_parking.desktop_app import TrafficResearchWindow, ObservationReviewDialog


@pytest.fixture
def mjpeg_server():
    import cv2
    import numpy as np
    stopped = threading.Event()
    payload = cv2.imencode(".jpg", np.full((24, 32, 3), 80, dtype=np.uint8))[1].tobytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                while not stopped.is_set():
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.04)
            except ConnectionError:
                return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/stream"
    stopped.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


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


def test_reference_metrics_follow_the_selected_event_gate(window):
    assert window.event_profile.currentData() == 2
    assert "44.0%" in window.metric_labels[0].text()
    window.event_profile.setCurrentIndex(0)
    assert window.event_profile.currentData() == 1
    assert "75.0%" in window.metric_labels[0].text()


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


def test_visual_evidence_recomputes_current_policy_and_exposes_evidence_gaps(window):
    window.report = {"events": [{"trigger_time_sec": 2, "evidence": {"contact_sheet": str(window.root / "docs/assets/incident_vlm_clean_evidence.jpg")}, "review_disposition": "corroborated_incident", "vlm_review": {"schema_valid": True, "likely_violation": True, "confidence": 0.95, "visual_reasons": ["Vehicle interaction"], "missing_evidence": ["Collision moment absent"], "human_review_needed": True}}]}
    window.frame_choice.setCurrentIndex(0)
    window.show_evidence(0)
    assert "incomplete visual evidence" in window.evidence_text.text()
    assert "Collision moment absent" in window.evidence_text.text()
    assert "Operator review requested" in window.evidence_text.text()
    assert "corroborated incident" not in window.evidence_text.text()


def test_extended_model_tab_uses_all_matched_models_and_ranking_metrics(window):
    tabs = window.pages.widget(1).findChild(QTabWidget)
    tabs.setCurrentIndex(1)
    table = tabs.currentWidget()
    assert table.rowCount() == 8
    assert table.horizontalHeaderItem(7).text() == "AUROC"
    assert table.horizontalHeaderItem(8).text() == "AP"


def test_native_vlm_tab_displays_corrected_results_and_clustered_uncertainty(window):
    tabs = window.pages.widget(1).findChild(QTabWidget)
    assert tabs.count() == 3
    tabs.setCurrentIndex(2)
    assert window.vlm_results.rowCount() == 3
    assert window.vlm_results.item(0, 2).text() == "95%"
    assert window.vlm_results.item(0, 3).text() == "60%"
    assert window.vlm_results.item(1, 3).text() == "30%"
    assert window.vlm_results.item(2, 3).text() == "55%"


def test_edge_page_exposes_tcn_event_retention_instead_of_only_proposal_counts(window):
    tabs = window.pages.widget(4).findChild(QTabWidget)
    assert tabs.count() == 2
    table = tabs.widget(1)
    assert table.rowCount() == 24
    assert table.horizontalHeaderItem(5).text() == "Service retained"


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
    assert "No review event" in window.evidence_text.text()
    assert window.actions_by_name["Export results"].isEnabled()
    assert window.video.current_image is not None
    assert not window.actions_by_name["Review selected event"].isEnabled()
    assert not window.actions_by_name["Annotate observation"].isEnabled()


def test_native_saved_event_review_does_not_depend_on_a_local_source_video(window, tmp_path, monkeypatch):
    report = {"input": "https://camera.invalid/live", "output_video": str(window.input_path),
              "events": [{"trigger_time_sec": 1, "probability": 0.8, "evidence": {"contact_sheet": str(window.root / "docs/assets/incident_vlm_clean_evidence.jpg")}}],
              "runtime": {"requested_provider": "CPUExecutionProvider"}, "elapsed_sec": 2}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    window.load_report(path)
    assert window.input_path is None
    assert window.actions_by_name["Review selected event"].isEnabled()
    launched = []
    monkeypatch.setattr(window, "_start_job", lambda *args: launched.append(args))
    window.review_selected_event()
    script, arguments, kind = launched[0]
    assert script == "scripts/review_desktop_events.py" and kind == "review"
    assert arguments[arguments.index("--report") + 1] == str(path.resolve())
    assert arguments[arguments.index("--event-index") + 1] == "0"
    assert "--input" not in arguments


def test_report_rejects_unknown_event_gate_before_changing_ui(window, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"events": [], "runtime": {}, "min_positive_windows": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="one or two"):
        window.load_report(path)
    assert window.event_profile.currentData() == 2


def test_native_capture_worker_loads_video_and_verified_clock(window, mjpeg_server):
    window.capture_camera(mjpeg_server, "mjpeg", 3, analyze_after=False)
    assert not window.actions_by_name["Open video"].isEnabled()
    for _ in range(200):
        if window.capture_report_path is not None:
            break
        QTest.qWait(50)
    assert window.capture_report_path is not None, window.log.toPlainText()
    assert window.input_path.name == "capture.mp4"
    assert window.actions_by_name["Analyze"].isEnabled()
    assert not window.preview_timer.isActive()
    assert window.video.current_image is not None
    report = json.loads(window.capture_report_path.read_text(encoding="utf-8"))
    assert report["source_metadata"]["connections"] == 1
    assert len(report["frame_times_sec"]) == window.video.frame_count


def test_native_capture_cancel_is_cooperative(window, mjpeg_server):
    window.capture_camera(mjpeg_server, "mjpeg", 30, analyze_after=False)
    QTest.qWait(1000)
    window.cancel_analysis()
    for _ in range(160):
        if window.job.state() == QProcess.ProcessState.NotRunning:
            break
        QTest.qWait(50)
    assert window.cancelled
    assert window.job.state() == QProcess.ProcessState.NotRunning
    assert not window.actions_by_name["Analyze"].isEnabled()
    assert not list(window.output_dir.glob("camera_frames_*"))


def test_native_continuous_worker_runs_without_waiting_for_capture_completion(window, mjpeg_server):
    if not (window.root / "yolov8n.pt").is_file():
        pytest.skip("Local detector required for native streaming integration")
    window.capture_camera(mjpeg_server, "mjpeg", 3, continuous=True)
    assert window.job.property("kind") == "stream"
    for _ in range(300):
        if window.report is not None:
            break
        QTest.qWait(50)
    assert window.report is not None, window.log.toPlainText()
    assert window.report["streaming_inference"] is True
    assert window.report["source_mode"] == "mjpeg"
    assert window.report["acquisition"]["decoded_frames"] >= window.report["sampled_frames"]
    assert window.actions_by_name["Annotate observation"].isEnabled()
    dialog = ObservationReviewDialog(window.report_path, window)
    assert dialog.label.currentData() == "uncertain"
    assert not dialog.complete.isChecked()
    assert not dialog.blinded.isChecked()
    assert dialog.video.path.name == "observation_clean.mp4"
    dialog.reject()
    assert dialog.video.capture is None
    dialog.deleteLater()
    assert window.actions_by_name["Export results"].isEnabled()
    assert not window.preview_timer.isActive()


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
