from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.vlm_review import VlmReviewRequest, build_unparsed_vlm_review_result, parse_vlm_review_result_text
from illegal_parking.vlm_transformers import TransformersVlmConfig, run_transformers_image_text_to_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Hugging Face Transformers VLM reviewer.")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--provider-name", default="qwen2_5_vl_3b")
    parser.add_argument("--device", default=None, help="Transformers pipeline device, such as 0, cpu, or cuda:0.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--result-output", required=True)
    parser.add_argument("--allow-unparsed", action="store_true", help="Write a review-needed result if the VLM response is not valid normalized JSON.")
    args = parser.parse_args()

    request = VlmReviewRequest.from_dict(json.loads(Path(args.request_json).read_text(encoding="utf-8")))
    config = TransformersVlmConfig(
        model_id=args.model_id,
        provider_name=args.provider_name,
        device=_parse_device(args.device),
        max_new_tokens=args.max_new_tokens,
    )
    raw_text = run_transformers_image_text_to_text(request, config)

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.write_text(raw_text, encoding="utf-8")

    try:
        result = parse_vlm_review_result_text(raw_text, provider=args.provider_name)
    except (ValueError, json.JSONDecodeError) as exc:
        if not args.allow_unparsed:
            raise
        result = build_unparsed_vlm_review_result(raw_text, provider=args.provider_name, error=str(exc))
    result_output = Path(args.result_output)
    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _parse_device(value: str | None) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
