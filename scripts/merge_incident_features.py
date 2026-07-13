from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_feature_fusion import merge_feature_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge ACCIDENT motion and trajectory features by temporal window.")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", default="data/processed/accident/fused_features.csv")
    parser.add_argument("--report", default="data/processed/accident/fused_feature_report.json")
    args = parser.parse_args()

    motion_path = _resolve(args.motion)
    trajectory_path = _resolve(args.trajectory)
    output_path = _resolve(args.output)
    report_path = _resolve(args.report)
    motion_rows = _read_rows(motion_path)
    trajectory_rows = _read_rows(trajectory_path)
    merged_rows = merge_feature_rows(motion_rows, trajectory_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0]))
        writer.writeheader()
        writer.writerows(merged_rows)
    report = {
        "motion": str(motion_path),
        "trajectory": str(trajectory_path),
        "motion_row_count": len(motion_rows),
        "trajectory_row_count": len(trajectory_rows),
        "merged_row_count": len(merged_rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Feature CSV has no rows: {path}")
    return rows


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
