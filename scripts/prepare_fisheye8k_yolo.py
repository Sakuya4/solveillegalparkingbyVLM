from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.fisheye8k import convert_fiftyone_samples_to_yolo


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Convert the Hugging Face FishEye8K FiftyOne export to YOLO format.")
    parser.add_argument("--metadata", default="data/raw/training/fisheye8k_hf_meta/samples.json")
    parser.add_argument("--image-source", default="data/raw/training/fisheye8k_hf")
    parser.add_argument("--output", default="data/processed/training/fisheye8k_yolo")
    parser.add_argument("--repo-id", default="Voxel51/fisheye8k")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-per-split", type=int, default=None)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()

    resolver = None
    if args.download_missing:
        from huggingface_hub import hf_hub_download

        cache_dir = ROOT / "data/raw/training/fisheye8k_hf_download_cache"

        def resolver(relative_path: str) -> Path | None:
            return Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=relative_path,
                    cache_dir=cache_dir,
                )
            )

    summary = convert_fiftyone_samples_to_yolo(
        metadata_path=ROOT / args.metadata,
        image_source_dir=ROOT / args.image_source,
        output_dir=ROOT / args.output,
        max_samples=args.max_samples,
        max_per_split=args.max_per_split,
        image_resolver=resolver,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
