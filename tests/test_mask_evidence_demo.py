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
