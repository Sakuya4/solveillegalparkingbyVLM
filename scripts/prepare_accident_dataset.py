from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import (
    build_temporal_windows,
    load_accident_manifest,
    summarize_accident_manifest,
    validate_accident_files,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Validate ACCIDENT metadata and create temporal training windows.")
    parser.add_argument("--metadata", default="data/raw/accident/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident")
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--negative-gap-frames", type=int, default=8)
    parser.add_argument("--output-dir", default="data/processed/accident")
    parser.add_argument("--require-all-videos", action="store_true")
    args = parser.parse_args()

    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    output_dir = _resolve(args.output_dir)

    clips = load_accident_manifest(metadata_path)
    summary = summarize_accident_manifest(clips)
    file_validation = validate_accident_files(clips, dataset_root)
    windows = [
        window
        for clip in clips
        for window in build_temporal_windows(
            clip,
            window_frames=args.window_frames,
            negative_gap_frames=args.negative_gap_frames,
        )
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    file_payload = file_validation.to_dict()
    missing_paths = file_payload.pop("missing_paths")
    file_payload["missing_path_sample"] = missing_paths[:20]
    report = {
        "metadata": str(metadata_path),
        "dataset_root": str(dataset_root),
        "window_frames": args.window_frames,
        "negative_gap_frames": args.negative_gap_frames,
        "manifest": summary.to_dict(),
        "files": file_payload,
        "window_count": len(windows),
        "normal_window_count": sum(window.label == "normal" for window in windows),
        "incident_window_count": sum(window.label == "incident" for window in windows),
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "missing_videos.txt").write_text("\n".join(missing_paths), encoding="utf-8")
    with (output_dir / "temporal_windows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for window in windows:
            handle.write(json.dumps(window.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all_videos and file_validation.missing_count:
        return 2
    return 0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
