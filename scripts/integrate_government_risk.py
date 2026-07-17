from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.deployment_coverage import (
    load_camera_sites,
    summarize_risk_camera_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join Taiwan A1/A2 hot spots with public enforcement-camera sites."
    )
    parser.add_argument(
        "--accident-report",
        default="data/processed/accidents/taiwan_a1_a2_hotspots_113.json",
    )
    parser.add_argument(
        "--camera-sites",
        default="data/raw/taiwan/new_taipei_illegal_parking_camera_sites.csv",
    )
    parser.add_argument(
        "--violation-report",
        default="data/processed/hotspots/taoyuan_hotspots_113.json",
    )
    parser.add_argument(
        "--output", default="docs/assets/government_risk_integration.json"
    )
    args = parser.parse_args()

    accident_report = _read_json(args.accident_report)
    violation_report = _read_json(args.violation_report)
    cameras = load_camera_sites(_resolve(args.camera_sites))
    all_hotspots = accident_report["hotspots"]
    new_taipei_hotspots = [
        hotspot for hotspot in all_hotspots if hotspot.get("city") == "新北市"
    ]
    coverage = summarize_risk_camera_coverage(new_taipei_hotspots, cameras)
    report = {
        "scope": {
            "spatial_join": "New Taipei A1/A2 top-risk grids versus New Taipei public illegal-parking camera sites",
            "violation_scenario": "Taoyuan violations are a separate municipality-level prioritization and cost scenario",
        },
        "government_records": {
            "taiwan_a1_a2": accident_report["records_loaded"],
            "taoyuan_violations": violation_report["records_loaded"],
            "new_taipei_camera_sites": len(cameras),
            "national_top_hotspots": len(all_hotspots),
            "new_taipei_hotspots_joined": len(new_taipei_hotspots),
            "out_of_city_hotspots_excluded": len(all_hotspots) - len(new_taipei_hotspots),
        },
        "risk_camera_coverage": coverage,
        "scenario_only_not_measured_outcome": {
            "a1_a2": accident_report["impact"],
            "violations": violation_report["impact"],
            "warning": (
                "Projected reductions and savings are planning assumptions. They require "
                "before/after field observations and are not measured causal outcomes."
            ),
        },
        "sources": [
            "https://data.gov.tw/dataset/172969",
            "https://data.ntpc.gov.tw/datasets/bb59a616-3572-4e92-9d09-01bf422057a6",
            "https://data.gov.tw/dataset/168176",
        ],
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_json(value: str) -> dict:
    return json.loads(_resolve(value).read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
