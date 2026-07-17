from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.cctv_evaluation import aggregate_normal_cctv_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate independent normal-only CCTV sessions.")
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--report-glob", action="append", default=[])
    parser.add_argument("--output", default="outputs/cctv/normal_cctv_aggregate.json")
    args = parser.parse_args()

    report_paths = [_resolve(path) for path in args.report]
    for pattern in args.report_glob:
        report_paths.extend(sorted(ROOT.glob(pattern)))
    report_paths = list(dict.fromkeys(report_paths))
    if not report_paths:
        raise SystemExit("Provide at least one --report or --report-glob")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    aggregate = aggregate_normal_cctv_reports(reports)
    payload = {
        "source_reports": [str(path) for path in report_paths],
        **aggregate,
        "interpretation": (
            "The zero-event upper bound is a one-sided 95% Poisson exposure bound; "
            "collect more camera-hours before deployment claims."
        ),
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
