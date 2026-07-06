from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.vlm_review import VlmReviewRequest, parse_vlm_review_result_text, review_redline_parking_offline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a VLM-style review package into a decision result.")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--provider",
        default="offline",
        choices=["offline", "vlm-json"],
        help="Offline is deterministic. vlm-json normalizes a saved VLM JSON/text response.",
    )
    parser.add_argument("--vlm-response-text", help="Path to a VLM response containing the expected JSON object.")
    parser.add_argument("--vlm-provider-name", default="external_vlm", help="Provider name for --provider vlm-json.")
    args = parser.parse_args()

    payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    request = VlmReviewRequest.from_dict(payload)
    if args.provider == "offline":
        result = review_redline_parking_offline(request)
    else:
        if not args.vlm_response_text:
            raise SystemExit("--vlm-response-text is required when --provider vlm-json")
        response_text = Path(args.vlm_response_text).read_text(encoding="utf-8")
        result = parse_vlm_review_result_text(response_text, provider=args.vlm_provider_name)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
