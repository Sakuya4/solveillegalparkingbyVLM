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
from illegal_parking.incident_baseline import grouped_calibration_indices
from illegal_parking.incident_presentation import paired_presentation_windows, normalized_presentation_times, read_presentation_frames


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_times(path, ffprobe):
    result = subprocess.run([str(ffprobe), "-v", "error", "-threads", "2", "-select_streams", "v:0",
                             "-show_packets", "-show_streams", "-show_entries", "stream=nb_frames,codec_name:packet=pts_time",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=60)
    data = json.loads(result.stdout)
    times = sorted(float(packet["pts_time"]) for packet in data["packets"])
    normalized_presentation_times(times)
    if len(times) != int(data["streams"][0]["nb_frames"]):
        raise ValueError("Packet timestamps disagree with container frame count")
    return times


def assign_partitions(rows, scheme):
    train = np.asarray([index for index, row in enumerate(rows) if row[scheme + "_split"] == "train"])
    fit, calibration = grouped_calibration_indices(np.asarray([rows[index]["path"] for index in train]), fraction=.2, seed=42)
    partitions = {int(train[index]): "fit" for index in fit}
    partitions.update({int(train[index]): "calibration" for index in calibration})
    for index, row in enumerate(rows):
        row[scheme + "_partition"] = partitions.get(index, row[scheme + "_split"])
    for partition in ("fit", "calibration", "test"):
        if {row["target"] for row in rows if row[scheme + "_partition"] == partition} != {0, 1}:
            raise ValueError(f"Both target classes are required in {scheme}/{partition}")


def main():
    parser = argparse.ArgumentParser(description="Freeze decoded-PTS-verified paired cohorts before feature extraction or training.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/accident/full/extracted")
    parser.add_argument("--ffprobe", type=Path, default=Path("C:/ffmpeg/bin/ffprobe.exe"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/presentation_protocol.json")
    args = parser.parse_args()
    rows_path = args.output.with_suffix(".jsonl")
    if args.output.exists() or rows_path.exists():
        raise FileExistsError("Frozen protocols are immutable; use a new output path")
    import cv2
    cv2.setNumThreads(2)
    source_audit_path = ROOT / "docs/results/cadence_source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    metadata = args.dataset_root / "metadata-real.csv"
    if sha256(metadata) != source_audit["metadata_sha256"]:
        raise ValueError("Metadata differs from the audited input")
    implementation = {path.relative_to(ROOT).as_posix(): sha256(path) for path in
                      [Path(__file__).resolve(), ROOT / "scripts/train_incident_tcn.py", *
                       (ROOT / "Application/DebugVersion/src/illegal_parking" / name for name in
                        ("incident_presentation.py", "incident_candidate_roi.py", "incident_ultralytics.py", "incident_tcn.py", "incident_baseline.py"))]}
    ffprobe_sha = sha256(args.ffprobe)
    signature = hashlib.sha256(json.dumps({"implementation": implementation, "metadata": sha256(metadata), "ffprobe": ffprobe_sha}, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "outputs/accident/presentation_validation_cache" / signature
    cache.mkdir(parents=True, exist_ok=True)
    source_media = {row["path"]: row for row in source_audit["input_media"]}
    rows, exclusions, inputs = [], [], []
    for index, clip in enumerate(load_accident_manifest(metadata), 1):
        path = clip.relative_path.as_posix()
        video = args.dataset_root / clip.relative_path
        video_sha = sha256(video)
        if video_sha != source_media[path]["sha256"]:
            raise ValueError(f"Video changed since the source audit: {path}")
        checkpoint = cache / (video_sha + ".json")
        if checkpoint.exists():
            item = json.loads(checkpoint.read_text(encoding="utf-8"))
            if item["path"] != path:
                checkpoint = cache / (hashlib.sha256(path.encode()).hexdigest() + ".json")
                item = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
        else:
            item = None
        if item is None:
            item = {"path": path, "sha256": video_sha, "rows": [], "exclusion": None}
            try:
                times = probe_times(video, args.ffprobe)
                if len(times) not in (clip.frame_count, clip.frame_count + 1):
                    raise ValueError("unexpected_container_metadata_frame_count_difference")
                paired, reason = paired_presentation_windows(clip, times)
                if reason:
                    item["exclusion"] = reason
                else:
                    normalized = normalized_presentation_times(times)
                    selected = sorted({clip.accident_frame, *(value for row in paired for arm in ("legacy", "fixed") for value in row[arm + "_indices"])})
                    read_presentation_frames(video, selected, normalized[selected].tolist())
                    item["rows"] = paired
                    item["verified_frame_indices"] = selected
                item["presentation_times_sha256"] = hashlib.sha256(json.dumps(times).encode()).hexdigest()
                item["frame_count"] = len(times)
            except (ValueError, OSError) as exc:
                item["exclusion"] = str(exc).replace(str(args.dataset_root), "<dataset-root>")
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(item) + "\n", encoding="utf-8")
            temporary.replace(checkpoint)
        rows.extend(item["rows"])
        if item["exclusion"]:
            exclusions.append({"path": path, "reason": item["exclusion"]})
        inputs.append({key: item[key] for key in ("path", "sha256", "presentation_times_sha256", "frame_count") if key in item})
        if index % 50 == 0:
            print(f"Verified {index}/{len(source_media)} clips; paired rows={len(rows)}, exclusions={len(exclusions)}", flush=True)
    for scheme in ("iid", "geographic"):
        assign_partitions(rows, scheme)
    previous = json.loads((ROOT / "docs/results/continuous_event_calibration_protocol.json").read_text(encoding="utf-8"))
    protocol = {"version": 1, "status": "frozen_before_feature_extraction", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_audit_sha256": sha256(source_audit_path), "metadata_sha256": sha256(metadata), "input_media": inputs,
                "clock": "Normalized packet presentation timestamps, checked against every selected decoded frame and supplied annotation frame",
                "sampling": {"target_fps": 7.5, "sample_count": 16, "sequence_steps": 15, "nominal_duration_sec": 2,
                             "max_lateness_sec": 1 / 7.5, "comparison_epsilon_sec": 1e-6, "decoder_tolerance_sec": 1e-4,
                             "grid_origin": "first source presentation timestamp", "interpolation": False,
                             "event_anchor": "supplied annotation frame without legacy boundary clamping",
                             "negative_gap_target_slots": 4, "legacy_window_frames": 32, "legacy_stride": 2, "legacy_negative_gap_frames": 8},
                "training": {"feature_width": 320, "image_size": 320, "confidence": .15, "tracker": "bytetrack.yaml",
                             "arms": ["legacy", "fixed"], "features": ["candidate_", "global_"], "epochs": 40,
                             "channels": 32, "dropout": .1, "batch_size": 64, "learning_rate": .001,
                             "weight_decay": .0001, "seed": 42, "target_train_fpr": .2, "calibration_fraction": .2},
                "detector_sha256": sha256(ROOT / "yolov8n.pt"),
                "reference_model_sha256": {path.name: sha256(path) for path in (ROOT / "Application/desktop/models").glob("*.onnx")},
                "paired_clips": len({row["path"] for row in rows}), "paired_windows": len(rows), "exclusions": exclusions,
                "partitions": {scheme: dict(Counter(row[scheme + "_partition"] for row in rows)) for scheme in ("iid", "geographic")},
                "continuous_cohorts": {scheme: {"calibration_clip_paths": sorted({row["path"] for row in rows if row[scheme + "_partition"] == "calibration"}),
                                                 "test_clip_paths": sorted(set(previous["models"][scheme]["test_clip_paths"]) & {row["path"] for row in rows if row[scheme + "_partition"] == "test"})}
                                       for scheme in ("iid", "geographic")},
                "evaluation": {"primary": "paired source-clip window-F1 difference at training-holdout FPR=0.2", "bootstrap_resamples": 2000, "seed": 42,
                               "continuous": {"hop_steps": 15, "min_positive_windows": 2, "episode_budget_per_clip": .5,
                                              "threshold_grid": previous["threshold_grid"], "selection_rule": previous["selection_rule"]}},
                "interpretation": "Joint sampling/context and pre-event-gap policy comparison; not an isolated model architecture effect",
                "default_policy": "Retain 500-clip reference binaries and desktop defaults; do not select replacements on test scores",
                "implementation_sha256": implementation, "ffprobe_sha256": ffprobe_sha,
                "windows_file": rows_path.name}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    protocol["windows_sha256"] = sha256(rows_path)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: protocol[key] for key in ("paired_clips", "paired_windows", "partitions")}, indent=2))


if __name__ == "__main__":
    main()
