from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_cadence import paired_cadence_windows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    return dict(zip(("min", "p05", "median", "p95", "max"), np.quantile(values, [0, .05, .5, .95, 1]).tolist()))


def inspect_presentation_clock(path, ffprobe):
    import cv2
    result = subprocess.run([str(ffprobe), "-v", "error", "-threads", "2", "-select_streams", "v:0",
                             "-show_packets", "-show_entries", "packet=pts_time", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True, timeout=60)
    timestamps = sorted(float(packet["pts_time"]) for packet in json.loads(result.stdout)["packets"])
    if not timestamps or not np.all(np.isfinite(timestamps)):
        raise ValueError("Packet timestamps must be nonempty and finite")
    times = np.asarray(timestamps) - timestamps[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Packet timestamps must be unique and increasing")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    errors = []
    index = 0
    origin = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Decoded video must report a positive FPS")
        while True:
            success, _ = cap.read()
            if not success:
                break
            if index >= len(times):
                raise ValueError("Decoded frames exceed presentation packets")
            decoded_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if origin is None:
                origin = decoded_time
            errors.append(abs(decoded_time - origin - times[index]))
            index += 1
    finally:
        cap.release()
    if index != len(times):
        raise ValueError("Decoded frames do not match presentation packets")
    return {"decoded_frames": index, "presentation_timestamps_sec": timestamps,
            "decoded_timestamp_origin_sec": origin,
            "packet_frame_alignment_verified": bool(max(errors) <= 1e-4),
            "decoded_vs_packet_max_abs_error_sec": float(max(errors)),
            "frame_index_fps_clock_max_abs_error_sec": float(np.max(np.abs(times - np.arange(index) / fps))),
            "presentation_interval_sec": quantiles(np.diff(times))}


def main():
    parser = argparse.ArgumentParser(description="Audit annotation indices, container clocks and train-only decoded presentation timestamps before cadence fitting.")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/accident/full/extracted/metadata-real.csv")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/accident/full/extracted")
    parser.add_argument("--ffprobe", type=Path, default=Path("C:/ffmpeg/bin/ffprobe.exe"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/cadence_source_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Source audits are immutable; use a new output path")
    import cv2
    cv2.setNumThreads(2)
    implementation = {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                      [Path(__file__).resolve(), ROOT / "Application/DebugVersion/src/illegal_parking/incident_cadence.py"]}
    clips = load_accident_manifest(args.metadata)
    rows, exclusions, media = [], [], []
    for index, clip in enumerate(clips):
        path = args.dataset_root / clip.relative_path
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot inspect {clip.relative_path}")
            fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        pair, reason = paired_cadence_windows(clip, fps, count)
        if reason:
            exclusions.append({"path": clip.relative_path.as_posix(), "reason": reason})
        rows.extend(pair)
        media.append({"path": clip.relative_path.as_posix(), "sha256": sha256(path),
                      "source_fps": fps, "container_frame_count": count, "metadata_frame_count": clip.frame_count,
                      "frame_count_difference": count - clip.frame_count,
                      "iid_split": clip.iid_split, "geographic_split": clip.geographic_split})
        if (index + 1) % 200 == 0:
            print(f"Inspected {index + 1}/{len(clips)} clips", flush=True)
    training = sorted((row for row in media if row["iid_split"] == row["geographic_split"] == "train"), key=lambda row: (row["source_fps"], row["path"]))
    selected = sorted({round(q * (len(training) - 1)) for q in (0, .1, .25, .5, .75, .9, 1)})
    pts_audits = [{"path": training[index]["path"], **inspect_presentation_clock(args.dataset_root / training[index]["path"], args.ffprobe)} for index in selected]
    audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "source_clock_audit_before_training_protocol",
             "software": {"opencv": cv2.__version__, "python": sys.version.split()[0], "threads": 2,
                          "ffprobe_sha256": sha256(args.ffprobe),
                          "ffprobe_version": subprocess.run([str(args.ffprobe), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]},
             "metadata_sha256": sha256(args.metadata), "input_media": media, "exclusions": exclusions,
             "requested_clips": len(clips), "nominal_pair_eligible_clips": len({row["path"] for row in rows}),
             "nominal_pair_eligible_windows": len(rows), "container_source_fps": quantiles([row["source_fps"] for row in media]),
             "frame_count_differences": dict(Counter(str(row["frame_count_difference"]) for row in media)),
             "nominal_window_duration_sec": {arm: quantiles([(row[arm + "_indices"][-1] - row[arm + "_indices"][0]) / row["source_fps"] for row in rows]) for arm in ("legacy", "fixed")},
             "train_only_presentation_clock_audits": pts_audits,
             "presentation_audit_selection": "Both official splits must be train; order by container FPS then path; unique ranks at quantiles 0, .1, .25, .5, .75, .9, 1",
             "nominal_grid": "Container average FPS and frame-index clock; 16 distinct frames, 15 flow steps, nominal target 7.5 Hz",
             "pair_support": "Preserve legacy metadata boundary; tolerate one extra container frame without consuming it in either arm; require event observability",
             "next_contract": "Freeze presentation-time sampling, label support and train/calibration/test membership before new feature extraction or fitting",
             "implementation_sha256": implementation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("requested_clips", "nominal_pair_eligible_clips", "frame_count_differences", "nominal_window_duration_sec")}, indent=2))


if __name__ == "__main__":
    main()
