from pathlib import Path

from packaging.requirements import Requirement


def test_desktop_manifest_declares_bytetrack_assignment_dependency():
    root = Path(__file__).resolve().parents[1]
    requirements = [Requirement(line) for line in (root / "requirements-desktop.txt").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    lap = next(value for value in requirements if value.name == "lap")
    assert "0.5.12" in lap.specifier and "0.5.13" in lap.specifier
    assert "0.5.11" not in lap.specifier
