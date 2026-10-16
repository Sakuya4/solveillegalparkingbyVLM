from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_summary(report, report_hash, video_hash, walkthrough_hash):
    return {
        "source": {
            "dataset": "ACCIDENT",
            "clip": "real_videos/ig4k6LEEMy0_00.mp4",
            "dataset_url": "https://www.kaggle.com/datasets/picekl/accident",
            "license": "CC BY-NC-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "license_metadata_url": "https://www.kaggle.com/api/v1/datasets/view/picekl/accident",
            "license_verified_on": "2026-10-16",
            "input_sha256": report["input_sha256"],
        },
        "source_report_sha256": report_hash,
        "started_at_utc": report["started_at_utc"],
        "model_sha256": report["model_sha256"],
        "detector_sha256": report["detector_sha256"],
        "decisions_sha256": report["decisions_sha256"],
        "runtime": report["runtime"],
        "threshold": report["threshold"],
        "sequence_steps": report["sequence_steps"],
        "hop_steps": report["hop_steps"],
        "min_positive_windows": report["min_positive_windows"],
        "display_width": report["display_width"],
        "feature_width_limit": report["feature_width_limit"],
        "sampled_frames": report["sampled_frames"],
        "windows": len(report["window_starts"]),
        "event_clock": report["event_clock"],
        "playback_clock": report["playback_clock"],
        "initialization_before_observation_sec": report["initialization_before_observation_sec"],
        "processing_sec": report["processing_sec"],
        "processing_scope": "Sampled-frame processing; excludes model initialization and VLM review",
        "events": [{key: event[key] for key in (
            "event_id", "trigger_time_sec", "probability", "post_context_truncated",
            "vlm_review", "review_disposition", "review_disposition_policy_version",
        )} for event in report["events"]],
        "privacy": report["privacy"],
        "interpretation": "Single reference-case execution with preserved temporal/visual disagreement; cohort estimates are reported separately",
        "media": {
            "incident_demo.mp4": video_hash,
            "desktop_walkthrough.mp4": walkthrough_hash,
            "codec": "H.264 / yuv420p / faststart / no audio",
        },
    }


def encode(ffmpeg, source, destination, credit=False):
    filters = []
    if credit:
        font = next((path for path in (Path("C:/Windows/Fonts/arial.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) if path.is_file()), None)
        if font is None:
            raise FileNotFoundError("A publication font is required to retain visible source attribution")
        font_path = font.as_posix().replace(":", "\\:")
        filters = ["-vf", "pad=iw:ih+28:0:0:black,drawtext="
                   f"fontfile='{font_path}':text='ACCIDENT - Arkansas DOT - CC BY-NC-SA 4.0':"
                   "fontcolor=white:fontsize=12:x=8:y=h-20"]
    subprocess.run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", "-filter_threads", "2",
        "-i", str(source), *filters, "-an", "-c:v", "libx264", "-threads", "2",
        "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ], check=True)


def make_previews(ffmpeg, video, output_dir):
    subprocess.run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
        "-ss", "10.2", "-i", str(video), "-frames:v", "1", "-threads", "2",
        str(output_dir / "incident_demo_preview.jpg"),
    ], check=True)
    subprocess.run([
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
        "-ss", "6", "-t", "8", "-i", str(video), "-filter_complex_threads", "2",
        "-filter_complex", "[0:v]fps=6,scale=640:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
        "-threads", "2", str(output_dir / "incident_demo_preview.gif"),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Publish the source-hashed reference demonstration as browser-compatible video.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--walkthrough", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reference = ROOT / "docs/assets/incident_input.mp4"
    if report["input_sha256"] != sha256(reference):
        raise ValueError("This publication protocol requires the attributed ACCIDENT reference clip")
    if sha256(report["input"]) != report["input_sha256"]:
        raise ValueError("Source video has changed since inference")
    capture = json.loads(args.walkthrough.with_suffix(".json").read_text(encoding="utf-8"))
    if capture["source_report_sha256"] != sha256(args.report):
        raise ValueError("Walkthrough and inference report must refer to the same execution")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / "incident_demo.mp4"
    walkthrough = args.output_dir / "desktop_walkthrough.mp4"
    encode(args.ffmpeg, report["output_video"], video, credit=True)
    encode(args.ffmpeg, args.walkthrough, walkthrough)
    make_previews(args.ffmpeg, video, args.output_dir)
    capture["video_sha256"] = sha256(walkthrough)
    capture["publication_codec"] = "H.264 / yuv420p / faststart / no audio"
    (args.output_dir / "desktop_walkthrough.json").write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
    summary = build_summary(report, sha256(args.report), sha256(video), sha256(walkthrough))
    for name in ("incident_demo_preview.jpg", "incident_demo_preview.gif"):
        summary["media"][name] = sha256(args.output_dir / name)
    (args.output_dir / "incident_demo.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(video), "walkthrough": str(walkthrough), "events": len(report["events"])}, indent=2))


if __name__ == "__main__":
    main()
