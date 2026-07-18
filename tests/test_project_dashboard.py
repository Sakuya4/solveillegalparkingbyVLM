from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.run_project_dashboard import is_allowed_dashboard_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "dashboard"


def test_dashboard_snapshot_and_static_assets_are_complete() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_project_dashboard.py", "--check"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot = json.loads((DASHBOARD / "data" / "project_snapshot.json").read_text(encoding="utf-8"))

    assert len(snapshot["pipeline"]) == 6
    assert len(snapshot["event_profiles"]) == 2
    assert len(snapshot["model_comparison"]) == 6
    assert set(snapshot["reports"]) == {"final", "status", "spec"}


def test_dashboard_html_references_existing_local_files_and_views() -> None:
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    local_refs = re.findall(r'(?:href|src)="([^"#]+)"', html)

    for reference in local_refs:
        if reference.startswith(("../docs/", "../spec", "data:")):
            continue
        assert (DASHBOARD / reference).is_file(), reference

    for view in ("overview", "models", "evidence", "deployment", "readiness"):
        assert f'id="{view}-view"' in html
        assert f'data-view="{view}"' in html


def test_dashboard_has_responsive_breakpoint_and_no_gradient_decoration() -> None:
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")

    assert "@media (max-width: 760px)" in css
    assert "linear-gradient" not in css
    assert "radial-gradient" not in css


def test_dashboard_server_only_exposes_demo_and_report_assets() -> None:
    assert is_allowed_dashboard_path("/dashboard/")
    assert is_allowed_dashboard_path("/docs/assets/accident_reference.mp4")
    assert is_allowed_dashboard_path("/docs/final_evaluation_report.md")
    assert not is_allowed_dashboard_path("/.git/config")
    assert not is_allowed_dashboard_path("/.env")
    assert not is_allowed_dashboard_path("/data/raw/private.csv")
    assert not is_allowed_dashboard_path("/dashboard/../../.git/config")
    assert not is_allowed_dashboard_path("/dashboard/%2e%2e%5c.git%5cconfig")
