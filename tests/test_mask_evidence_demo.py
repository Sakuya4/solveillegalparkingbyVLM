import json
import subprocess
import sys
from pathlib import Path


def test_mask_evidence_demo_writes_visual_artifacts(tmp_path):
    script = Path("scripts/run_mask_evidence_demo.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(result.stdout)
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))

    assert stdout["mask_source"] == "bbox"
    assert evidence["mask_source"] == "bbox"
    assert evidence["vehicle_mask_area"] > 0
    assert evidence["restricted_overlap_pixels"] > 0
    assert 0.0 < evidence["restricted_overlap_ratio"] <= 1.0
    assert (tmp_path / "original.jpg").exists()
    assert (tmp_path / "bbox_overlay.jpg").exists()
    assert (tmp_path / "vehicle_mask.png").exists()
    assert (tmp_path / "restricted_mask.png").exists()
    assert (tmp_path / "overlap_overlay.jpg").exists()


def test_mask_evidence_demo_accepts_restricted_line(tmp_path):
    script = Path("scripts/run_mask_evidence_demo.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--bbox",
            "270,250,470,380",
            "--restricted-line",
            "200,340,520,340,14",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(result.stdout)

    assert stdout["mask_source"] == "bbox"
    assert stdout["restricted_line"] == [200, 340, 520, 340, 14]
    assert stdout["restricted_overlap_pixels"] > 0
    assert "restricted_rect" not in stdout


def test_mask_evidence_demo_supports_line_margin_and_privacy_regions(tmp_path):
    script = Path("scripts/run_mask_evidence_demo.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--bbox",
            "270,250,470,380",
            "--restricted-line",
            "200,340,520,340,14",
            "--restricted-line-margin-px",
            "10",
            "--blur-region",
            "300,300,420,340",
            "--hide-region",
            "350,420,700,470",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = json.loads(result.stdout)

    assert stdout["restricted_line_margin_px"] == 10
    assert stdout["restricted_overlap_pixels"] > 0
    assert (tmp_path / "original.jpg").exists()

    import cv2

    original = cv2.imread(str(tmp_path / "original.jpg"))
    hidden_region = original[430:460, 360:690]
    assert hidden_region.mean() < 50
