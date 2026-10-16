from __future__ import annotations

import argparse
import posixpath
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.project_snapshot import write_project_snapshot


class DashboardHandler(SimpleHTTPRequestHandler):
    def _allowed_path(self) -> str | None:
        path = posixpath.normpath(unquote(urlsplit(self.path).path))
        return path if is_allowed_dashboard_path(path) else None

    def _serve_allowed(self, head_only: bool = False) -> None:
        allowed = self._allowed_path()
        if allowed is None:
            self.send_error(404)
            return
        if allowed == "/":
            self.send_response(302)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return
        if head_only:
            super().do_HEAD()
        else:
            super().do_GET()

    def do_GET(self) -> None:
        self._serve_allowed()

    def do_HEAD(self) -> None:
        self._serve_allowed(head_only=True)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"dashboard: {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated project dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Build the snapshot and exit.")
    return parser.parse_args()


def is_allowed_dashboard_path(path: str) -> bool:
    decoded = unquote(urlsplit(path).path)
    if "\\" in decoded:
        return False
    normalized = posixpath.normpath(decoded)
    return (
        normalized == "/"
        or normalized == "/dashboard"
        or normalized.startswith("/dashboard/")
        or normalized.startswith("/docs/assets/")
        or normalized in {
            "/docs/final_evaluation_report.md",
            "/docs/project_status.md",
            "/spec.md",
        }
    )


def create_server(host: str, preferred_port: int) -> tuple[ThreadingHTTPServer, int]:
    handler = partial(DashboardHandler, directory=str(ROOT))
    for port in range(preferred_port, preferred_port + 20):
        try:
            return ThreadingHTTPServer((host, port), handler), port
        except OSError:
            continue
    raise OSError(f"No available port in range {preferred_port}-{preferred_port + 19}")


def main() -> None:
    args = parse_args()
    write_project_snapshot(ROOT, ROOT / "dashboard" / "data" / "project_snapshot.json")
    if args.check:
        print("Dashboard snapshot and assets are ready.")
        return

    server, port = create_server(args.host, args.port)
    url = f"http://{args.host}:{port}/dashboard/"
    print(f"Project dashboard: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
