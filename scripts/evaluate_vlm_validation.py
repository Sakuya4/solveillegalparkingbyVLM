from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.event_validation import evaluate_vlm_review_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VLM review results against an event validation manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--provider", action="append", required=True, help="Provider name. Can be repeated.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_vlm_review_manifest(
        manifest_path=args.manifest,
        results_root=args.results_root,
        providers=args.provider,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
