from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.vlm_review import VlmReviewResult, compare_vlm_review_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare normalized VLM review results for one event.")
    parser.add_argument("--result-json", action="append", required=True, help="Path to a VLM review result JSON.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-provider", default="offline_evidence_reviewer")
    args = parser.parse_args()

    results = [
        VlmReviewResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        for path in args.result_json
    ]
    report = {
        "baseline_provider": args.baseline_provider,
        "rows": compare_vlm_review_results(results, baseline_provider=args.baseline_provider),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
