from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from illegal_parking.project_snapshot import build_project_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_snapshot_combines_evaluation_and_deployment_artifacts() -> None:
    snapshot = build_project_snapshot(REPO_ROOT)

    responsive = snapshot["event_profiles"][0]
    assert responsive["profile"] == "responsive"
    assert responsive["event_recall"] == pytest.approx(0.75)
    assert responsive["delay_median_sec"] == pytest.approx(0.394)
    assert len(snapshot["model_comparison"]) == 6
    assert snapshot["deployment"]["systemc"]["python_parity"] is True
    assert snapshot["government"]["coverage"]["1000"]["risk_weighted_coverage_rate"] == pytest.approx(0.1895454545)


def test_project_snapshot_exposes_honest_readiness_states_and_portable_assets() -> None:
    snapshot = build_project_snapshot(REPO_ROOT)
    states = {item["state"] for item in snapshot["readiness"]["items"]}

    assert states == {"verified", "pilot", "external"}
    assert snapshot["readiness"]["verified"] == 9
    assert snapshot["readiness"]["pilot"] == 3
    assert snapshot["readiness"]["external"] == 3
    for asset in snapshot["media"].values():
        assert not Path(asset).is_absolute()
        assert (REPO_ROOT / "dashboard" / asset).resolve().is_file()


def test_build_project_snapshot_cli_writes_dashboard_contract(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"
    subprocess.run(
        [sys.executable, "scripts/build_project_snapshot.py", "--output", str(output)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["project"]["dataset"] == "ACCIDENT fixed-CCTV benchmark"
