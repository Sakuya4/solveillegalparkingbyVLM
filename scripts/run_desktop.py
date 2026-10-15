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
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from illegal_parking.desktop_app import TrafficResearchWindow

    app = QApplication(sys.argv[:1])
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ("segoeui.ttf", "seguisb.ttf", "msjh.ttc"):
            QFontDatabase.addApplicationFont(str(fonts / filename))
        app.setFont(QFont("Segoe UI", 10))
    window = TrafficResearchWindow(ROOT)
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
            QTimer.singleShot(120000, lambda: app.exit(1))
        else:
            QTimer.singleShot(500, capture)
    elif args.analyze:
        QTimer.singleShot(100, window.analyze_video)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
