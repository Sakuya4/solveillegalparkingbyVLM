from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application" / "DebugVersion" / "src"))


def main():
    parser = argparse.ArgumentParser(description="Launch the native traffic research workbench.")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--page", type=int, choices=range(5), default=0)
    parser.add_argument("--tab", type=int, default=0)
    parser.add_argument("--analyze", action="store_true", help="Analyze the reference clip after launch.")
    parser.add_argument("--vlm-review", action="store_true", help="Enable local Qwen evidence review for analysis.")
    parser.add_argument("--report", type=Path, help="Open a previously completed analysis report.")
    parser.add_argument("--capture-page", help="Observe a public MJPEG camera page before optional analysis.")
    parser.add_argument("--capture-duration", type=float, default=30)
    parser.add_argument("--continuous", action="store_true", help="Run causal streaming inference instead of capture followed by analysis")
    parser.add_argument("--seek-sec", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.tab < {1: 3, 3: 3, 4: 2}.get(args.page, 1):
        parser.error("Select an available tab for this page")
    if args.report and args.capture_page:
        parser.error("Choose a saved report or a camera source")
    if args.continuous and not args.capture_page:
        parser.error("--continuous requires --capture-page")
    if not 0 < args.capture_duration <= 120:
        parser.error("Camera observations must be at most 120 seconds")
    if not math.isfinite(args.seek_sec) or args.seek_sec < 0:
        parser.error("--seek-sec must be finite and nonnegative")
    if args.capture_page:
        from illegal_parking.cctv_capture import redacted_camera_url
        try:
            redacted_camera_url(args.capture_page)
        except ValueError as exc:
            parser.error(str(exc))
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QEvent, QTimer
    from PySide6.QtWidgets import QApplication, QTabWidget
    from illegal_parking.desktop_app import TrafficResearchWindow, configure_application_fonts

    app = QApplication(sys.argv[:1])
    configure_application_fonts(app)
    window = TrafficResearchWindow(ROOT)
    window.vlm_enabled.setChecked(args.vlm_review)
    if args.report:
        window.load_report(args.report)
        if args.seek_sec > 0:
            window.video.seek(min(window.video.frame_count - 1, round(args.seek_sec * window.video.fps)))
    window.navigation.setCurrentRow(args.page)
    if args.tab:
        window.pages.widget(args.page).findChild(QTabWidget).setCurrentIndex(args.tab)
    window.show()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            window.close()
            app.quit()
        if args.analyze or args.capture_page:
            def check_finished():
                ready = window.report is not None if args.analyze or args.continuous else window.capture_report_path is not None
                if ready:
                    if window.report is not None and window.report["events"]:
                        window.events.selectRow(0)
                    elif window.report is not None:
                        window.video.seek(min(window.video.frame_count - 1, round((args.seek_sec or 6) * window.video.fps)))
                    QTimer.singleShot(200, capture)
                elif window.job is not None and window.job.state() == window.job.ProcessState.NotRunning:
                    print(window.log.toPlainText(), file=sys.stderr)
                    window.close()
                    app.exit(1)
                else:
                    QTimer.singleShot(100, check_finished)
            if args.capture_page:
                QTimer.singleShot(100, lambda: window.capture_camera(args.capture_page, "page", args.capture_duration, args.analyze, continuous=args.continuous))
            else:
                QTimer.singleShot(100, window.analyze_video)
            QTimer.singleShot(300, check_finished)
            QTimer.singleShot(round((args.capture_duration + 180) * 1000), window, lambda: (window.close(), app.exit(1)))
        else:
            QTimer.singleShot(500, capture)
    elif args.capture_page:
        QTimer.singleShot(100, lambda: window.capture_camera(args.capture_page, "page", args.capture_duration, args.analyze, continuous=args.continuous))
    elif args.analyze:
        QTimer.singleShot(100, window.analyze_video)
    exit_code = app.exec()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
