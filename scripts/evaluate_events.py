from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.evaluation import (
    evaluate_frame_events,
    load_frame_annotations,
    load_predicted_event_frames,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frame-level illegal-parking event predictions.")
    parser.add_argument("--events", required=True, help="Path to events.jsonl produced by run_edge_simulation.py.")
    parser.add_argument("--annotations", required=True, help="Path to frame-level annotations JSON.")
    parser.add_argument("--output", help="Optional path to write the metrics JSON.")
    args = parser.parse_args()

    annotations = load_frame_annotations(args.annotations)
    predicted_frames = load_predicted_event_frames(args.events)
    result = evaluate_frame_events(annotations, predicted_frames)
    payload = result.to_dict()

    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
