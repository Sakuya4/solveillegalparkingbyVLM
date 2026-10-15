from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application" / "DebugVersion" / "src"))


def main():
    parser = argparse.ArgumentParser(description="Launch the native traffic research workbench.")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--page", type=int, choices=range(5), default=0)
    parser.add_argument("--analyze", action="store_true", help="Analyze the reference clip after launch.")
    parser.add_argument("--vlm-review", action="store_true", help="Enable local Qwen evidence review for analysis.")
    parser.add_argument("--report", type=Path, help="Open a previously completed analysis report.")
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QEvent, QTimer
    from PySide6.QtWidgets import QApplication
    from illegal_parking.desktop_app import TrafficResearchWindow, configure_application_fonts

    app = QApplication(sys.argv[:1])
    configure_application_fonts(app)
    window = TrafficResearchWindow(ROOT)
    window.vlm_enabled.setChecked(args.vlm_review)
    if args.report:
        window.load_report(args.report)
    window.navigation.setCurrentRow(args.page)
    window.show()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            window.close()
            app.quit()
        if args.analyze:
            def check_finished():
                if window.report is not None:
                    if window.report["events"]:
                        window.events.selectRow(0)
                    QTimer.singleShot(200, capture)
                elif window.actions_by_name["Analyze"].isEnabled():
                    print(window.log.toPlainText(), file=sys.stderr)
                    window.close()
                    app.exit(1)
                else:
                    QTimer.singleShot(100, check_finished)
            QTimer.singleShot(100, window.analyze_video)
            QTimer.singleShot(300, check_finished)
            QTimer.singleShot(120000, window, lambda: (window.close(), app.exit(1)))
        else:
            QTimer.singleShot(500, capture)
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
