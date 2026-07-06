from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.vlm_review import build_redline_parking_review_request


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a VLM review request from event evidence.")
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence_json).read_text(encoding="utf-8"))
    request = build_redline_parking_review_request(evidence, args.image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(request.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(request.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
