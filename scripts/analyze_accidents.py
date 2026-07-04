from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accidents import (
    AccidentRiskImpactConfig,
    estimate_accident_risk_impact,
    load_a1_a2_accident_records,
    summarize_accident_hotspots,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze Taiwan A1/A2 accident risk hot spots.")
    parser.add_argument("--input", default="data/raw/taiwan/taiwan_accidents_a1_a2_113.zip")
    parser.add_argument("--output", default="data/processed/accidents/taiwan_a1_a2_hotspots.json")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--grid-precision", type=int, default=3)
    parser.add_argument("--a1-weight", type=float, default=5.0)
    parser.add_argument("--a2-weight", type=float, default=1.0)
    parser.add_argument("--high-ratio", type=float, default=0.2)
    parser.add_argument("--medium-ratio", type=float, default=0.5)
    parser.add_argument("--high-risk-reduction-rate", type=float, default=0.2)
    args = parser.parse_args()

    records = load_a1_a2_accident_records(ROOT / args.input, limit=args.sample_limit)
    hotspots = summarize_accident_hotspots(
        records,
        top_n=args.top_n,
        grid_precision=args.grid_precision,
        a1_weight=args.a1_weight,
        a2_weight=args.a2_weight,
        high_ratio=args.high_ratio,
        medium_ratio=args.medium_ratio,
    )
    impact = estimate_accident_risk_impact(
        hotspots,
        config=AccidentRiskImpactConfig(high_risk_reduction_rate=args.high_risk_reduction_rate),
    )
    report = {
        "input": args.input,
        "records_loaded": len(records),
        "risk_definition": {
            "unit": "coordinate_grid",
            "grid_precision": args.grid_precision,
            "a1_weight": args.a1_weight,
            "a2_weight": args.a2_weight,
            "top_n": args.top_n,
            "high_ratio": args.high_ratio,
            "medium_ratio": args.medium_ratio,
        },
        "intervention_assumption": {
            "high_risk_reduction_rate": args.high_risk_reduction_rate,
            "target": "reduce detectable precursor events near high-risk accident grids",
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
