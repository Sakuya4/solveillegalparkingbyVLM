from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.training_analysis import analyze_training_run


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze detector training results and generate next-step recommendations.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--class-metrics-json", default=None)
    parser.add_argument("--model-name", default="detector")
    parser.add_argument("--output", default="data/processed/training/training_analysis.json")
    args = parser.parse_args()

    report = analyze_training_run(
        results_csv=ROOT / args.results_csv,
        class_metrics_json=ROOT / args.class_metrics_json if args.class_metrics_json else None,
        model_name=args.model_name,
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
