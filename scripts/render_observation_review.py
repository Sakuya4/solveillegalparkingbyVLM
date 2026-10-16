import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))


def main():
    parser = argparse.ArgumentParser(description="Render the native clean-video observation annotation dialog without assigning a label.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from illegal_parking.desktop_app import ObservationReviewDialog, configure_application_fonts, STYLE
    app = QApplication([])
    configure_application_fonts(app)
    app.setStyleSheet(STYLE)
    dialog = ObservationReviewDialog(args.report)
    dialog.show()

    def capture():
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        dialog.grab().save(str(args.screenshot))
        dialog.reject()
        app.quit()

    QTimer.singleShot(400, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
