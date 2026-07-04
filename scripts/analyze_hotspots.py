from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.hotspots import (
    EnforcementCostConfig,
    classify_hotspots,
    estimate_intervention_impact,
    load_taoyuan_violation_records,
    summarize_road_hotspots,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze violation hot spots and estimate automated enforcement impact.")
    parser.add_argument("--input", default="data/raw/taiwan/taoyuan_traffic_violations_113.csv")
    parser.add_argument("--encoding", default="big5")
    parser.add_argument("--output", default="data/processed/hotspots/taoyuan_hotspots.json")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--high-ratio", type=float, default=0.2)
    parser.add_argument("--medium-ratio", type=float, default=0.5)
    parser.add_argument("--high-heat-reduction-rate", type=float, default=0.35)
    parser.add_argument("--minutes-per-manual-case", type=float, default=8.0)
    parser.add_argument("--hourly-labor-cost", type=float, default=550.0)
    parser.add_argument("--system-monthly-cost", type=float, default=0.0)
    args = parser.parse_args()

    records = load_taoyuan_violation_records(ROOT / args.input, encoding=args.encoding, limit=args.sample_limit)
    hotspots = classify_hotspots(
        summarize_road_hotspots(records, top_n=args.top_n),
        high_ratio=args.high_ratio,
        medium_ratio=args.medium_ratio,
    )
    impact = estimate_intervention_impact(
        hotspots,
        high_heat_reduction_rate=args.high_heat_reduction_rate,
        cost_config=EnforcementCostConfig(
            minutes_per_manual_case=args.minutes_per_manual_case,
            hourly_labor_cost=args.hourly_labor_cost,
            system_monthly_cost=args.system_monthly_cost,
        ),
    )
    report = {
        "input": args.input,
        "records_loaded": len(records),
        "hotspot_definition": {
            "group_by": ["city", "area", "road"],
            "high_ratio": args.high_ratio,
            "medium_ratio": args.medium_ratio,
            "top_n": args.top_n,
        },
        "intervention_assumption": {
            "high_heat_reduction_rate": args.high_heat_reduction_rate,
            "minutes_per_manual_case": args.minutes_per_manual_case,
            "hourly_labor_cost": args.hourly_labor_cost,
            "system_monthly_cost": args.system_monthly_cost,
        },
        "impact": impact.to_dict(),
        "hotspots": [hotspot.to_dict() for hotspot in hotspots],
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
