from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.cctv_evaluation import aggregate_normal_cctv_reports
from illegal_parking.incident_evaluation import (
    evaluate_incident_report,
    match_report_to_accident_clip,
    summarize_incident_evaluations,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate annotation-free incident reports against ACCIDENT event timing."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--report-glob", action="append", default=[])
    parser.add_argument("--normal-report", action="append", default=[])
    parser.add_argument("--normal-report-glob", action="append", default=[])
    parser.add_argument("--early-tolerance-sec", type=float, default=1.0)
    parser.add_argument("--late-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_paths = _collect_paths(args.report, args.report_glob)
    if not report_paths:
        raise SystemExit("Provide at least one --report or --report-glob")
    clips = load_accident_manifest(_resolve(args.metadata))
    evaluations = []
    seen_clip_paths: set[str] = set()
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        clip = match_report_to_accident_clip(report, clips)
        clip_key = clip.relative_path.as_posix()
        if clip_key in seen_clip_paths:
            raise ValueError(f"Duplicate incident report for {clip_key}")
        seen_clip_paths.add(clip_key)
        evaluations.append(
            evaluate_incident_report(
                report,
                clip,
                early_tolerance_sec=args.early_tolerance_sec,
                late_tolerance_sec=args.late_tolerance_sec,
            )
        )

    payload = {
        "methodology": {
            "trigger_time": "window_end",
            "early_tolerance_sec": args.early_tolerance_sec,
            "late_tolerance_sec": args.late_tolerance_sec,
            "annotation_contract": (
                "ACCIDENT timing is used only by this post-inference evaluator; "
                "inference reports must declare annotation_free_inference=true."
            ),
            "false_alert_contract": (
                "False alerts per camera-hour are calculated only from separately reviewed normal CCTV reports."
            ),
        },
        "incident_evaluation": summarize_incident_evaluations(evaluations),
        "clips": [item.to_dict() for item in evaluations],
        "source_reports": [str(path) for path in report_paths],
    }

    normal_paths = _collect_paths(args.normal_report, args.normal_report_glob)
    if normal_paths:
        normal_reports = [json.loads(path.read_text(encoding="utf-8")) for path in normal_paths]
        payload["normal_cctv_evaluation"] = aggregate_normal_cctv_reports(normal_reports)
        payload["normal_source_reports"] = [str(path) for path in normal_paths]

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _collect_paths(explicit: list[str], patterns: list[str]) -> list[Path]:
    paths = [_resolve(value) for value in explicit]
    for pattern in patterns:
        paths.extend(sorted(ROOT.glob(pattern)))
    return list(dict.fromkeys(paths))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
