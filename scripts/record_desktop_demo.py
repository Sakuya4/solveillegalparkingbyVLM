from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))


def main():
    parser = argparse.ArgumentParser(description="Record the native workbench using a completed analysis.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/desktop/workbench_demo.mp4")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=6)
    args = parser.parse_args()
    if args.fps <= 0 or args.segment_seconds <= 0:
        parser.error("Recording cadence and segment duration must be positive")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import cv2
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QTimer
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QTabWidget
    from illegal_parking.desktop_app import TrafficResearchWindow, configure_application_fonts

    cv2.setNumThreads(2)
    app = QApplication([])
    configure_application_fonts(app)
    window = TrafficResearchWindow(ROOT)
    window.load_report(args.report)
    window.show()
    app.processEvents()
    resolution = [window.width(), window.height()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (window.width(), window.height()))
    if not writer.isOpened():
        raise OSError(f"Cannot create {args.output}")
    frame_count = 0
    previous_page = None
    sequence = (0, 2, 1, 3, 4)
    timer = QTimer(window)

    def capture():
        nonlocal frame_count, previous_page
        elapsed = frame_count / args.fps
        segment = int(elapsed / args.segment_seconds)
        if segment >= len(sequence):
            timer.stop()
            window.close()
            app.quit()
            return
        page = sequence[segment]
        if page != previous_page:
            window.video.timer.stop()
            window.navigation.setCurrentRow(page)
            if page == 0:
                trigger = window.report["events"][0]["trigger_time_sec"] if window.report["events"] else 2
                window.video.seek(round(max(0, trigger - 2) * window.video.fps))
                window.video.toggle_playback()
            elif page in (1, 3):
                window.pages.widget(page).findChild(QTabWidget).setCurrentIndex(2)
            elif page == 4:
                window.pages.widget(page).findChild(QTabWidget).setCurrentIndex(1)
            previous_page = page
        if page == 2:
            within_segment = elapsed - segment * args.segment_seconds
            window.frame_choice.setCurrentIndex(min(3, 1 + int(3 * within_segment / args.segment_seconds)))
        picture = window.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
        pixels = np.frombuffer(picture.constBits(), dtype=np.uint8).reshape(picture.height(), picture.bytesPerLine())
        rgb = pixels[:, :picture.width() * 3].reshape(picture.height(), picture.width(), 3)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_count += 1

    timer.timeout.connect(capture)
    timer.start(round(1000 / args.fps))
    try:
        exit_code = app.exec()
    finally:
        timer.stop()
        writer.release()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
    manifest = {
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "capture": "Native PySide6 window rendering with recorded analysis and video playback",
        "pages": ["incident", "evidence", "model_comparison", "government_coverage", "edge_runtime"],
        "selected_tabs": {"model_comparison": 2, "government_coverage": 2, "edge_runtime": 1},
        "model_sha256": window.report["model_sha256"],
        "fps": args.fps, "frames": frame_count, "duration_sec": frame_count / args.fps,
        "resolution": resolution, "source_media_license": "CC BY-NC-SA 4.0",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
