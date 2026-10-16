from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import probe_times, sha256
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES, candidate_motion_sequence_matrix, compute_candidate_motion_step
from illegal_parking.incident_presentation import normalized_presentation_times, presentation_grid, read_presentation_frames
from illegal_parking.incident_presentation_events import continuous_presentation_windows
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor
from illegal_parking.incident_ultralytics import track_vehicle_frames


def load_event_protocol(path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_continuous_pts_replay":
        raise ValueError("A frozen event protocol is required")
    if (protocol["sampling"]["target_fps"] != 7.5 or protocol["sampling"]["sample_count"] != 16
            or protocol["inference"]["hop_steps"] != 15 or protocol["inference"]["width"] != 320
            or protocol["calibration"]["min_positive_windows"] != 2 or protocol["calibration"]["episode_budget_per_clip"] != .5):
        raise ValueError("The implementation requires the declared fixed sampling/gating contract")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Pinned event implementation changed: {name}")
    for definition in protocol["models"].values():
        if sha256(ROOT / definition["path"]) != definition["sha256"]:
            raise ValueError("Pinned event model changed")
    for name, expected in protocol["reference_model_sha256"].items():
        if sha256(ROOT / "Application/desktop/models" / name) != expected:
            raise ValueError("Reference model bytes changed")
    if sha256(ROOT / "docs/results/presentation_protocol.json") != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent presentation protocol changed")
    if sha256(ROOT / "docs/results/presentation_comparison.json") != protocol["comparison_sha256"]:
        raise ValueError("Parent window comparison changed")
    if sha256(ROOT / "yolov8n.pt") != protocol["detector_sha256"]:
        raise ValueError("Frozen detector bytes changed")
    return protocol


def cohort_paths(protocol, cohort):
    if cohort not in ("calibration", "test"):
        raise ValueError("Only calibration or test cohorts are supported")
    for definition in protocol["cohorts"].values():
        a, b = definition["calibration_clip_paths"], definition["test_clip_paths"]
        if len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) & set(b):
            raise ValueError("Each split must have disjoint, unique calibration/test clips")
    return sorted({name for definition in protocol["cohorts"].values() for name in definition[cohort + "_clip_paths"]})


def clip_features(video, timestamps, detector, device):
    times = normalized_presentation_times(timestamps)
    windows, rejected = continuous_presentation_windows(timestamps)
    grid, _ = presentation_grid(times)
    # Decode from source origin, verifying even grid samples outside valid contexts.
    support = sorted(set(grid.tolist()))
    frames = dict(zip(support, read_presentation_frames(video, support, times[support], 320)))
    tensors = []
    for window in windows:
        indices = window["indices"]
        images = [frames[index] for index in indices]
        tracks = track_vehicle_frames(detector, images, indices, device=device, image_size=320, confidence=.15)
        steps = [compute_candidate_motion_step(images[offset - 1], images[offset], tracks[indices[offset]]) for offset in range(1, 16)]
        tensors.append(candidate_motion_sequence_matrix(steps))
    matrix = np.stack(tensors).astype(np.float32) if tensors else np.empty((0, 15, len(CANDIDATE_MOTION_FEATURE_NAMES)), dtype=np.float32)
    if not np.all(np.isfinite(matrix)) or matrix.shape != (len(windows), 15, len(CANDIDATE_MOTION_FEATURE_NAMES)):
        raise ValueError("Continuous feature tensors violate the frozen shape/value contract")
    return matrix, windows, rejected


def validate_batch(directory, protocol_path, protocol, cohort, selection_hash=None):
    path = directory / "batch.json"
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch["protocol_sha256"] != sha256(protocol_path) or batch["cohort"] != cohort or batch["smoke_clips"]:
        raise ValueError("Event batch is not a complete frozen-cohort replay")
    if batch["selection_sha256"] != selection_hash:
        raise ValueError("Test replay does not bind the frozen training-only selection")
    expected = cohort_paths(protocol, cohort)
    if sorted(batch["clips"]) != expected or len(batch["clips"]) != len(expected):
        raise ValueError("Event batch must retain every frozen source clip without replacement")
    reports = []
    sources = {row["path"]: row["sha256"] for row in protocol["input_media"]}
    for name in expected:
        record = batch["clips"][name]
        report_path = directory / record["report"]
        feature_path = directory / record["features"]
        if sha256(report_path) != record["report_sha256"] or sha256(feature_path) != record["features_sha256"]:
            raise ValueError("Continuous event report or features changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["protocol_sha256"] != batch["protocol_sha256"] or report["path"] != name or report["features_sha256"] != record["features_sha256"]:
            raise ValueError("Per-clip event provenance does not match the batch")
        if (report["signature"] != batch["signature"] or report["source_sha256"] != sources[name]
                or set(report["scores"]) != set(protocol["models"])):
            raise ValueError("Per-clip source, runtime signature or model set changed")
        reports.append(report)
    return batch, reports


def main():
    parser = argparse.ArgumentParser(description="Replay all frozen event clips on actual presentation timestamps.")
    parser.add_argument("--cohort", choices=("calibration", "test"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/presentation_event_protocol.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--smoke-clips", type=int)
    args = parser.parse_args()
    protocol = load_event_protocol(args.protocol)
    if args.smoke_clips is not None and (args.smoke_clips <= 0 or args.cohort != "calibration" or args.output_dir is None):
        parser.error("Smoke replay requires positive calibration clip count and a separate output directory")
    selection_path = ROOT / "outputs/accident/presentation_event_selection.json"
    selection_hash = None
    if args.cohort == "test":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["protocol_sha256"] != sha256(args.protocol):
            raise ValueError("Test replay requires the frozen training-only event selection")
        validate_batch(ROOT / "outputs/accident/presentation_events_calibration", args.protocol, protocol, "calibration")
        if (selection["calibration_batch_sha256"] != sha256(ROOT / "outputs/accident/presentation_events_calibration/batch.json")
                or set(selection["policies"]) != set(protocol["models"])):
            raise ValueError("Training-only selection is incomplete or its calibration batch changed")
        selection_hash = sha256(selection_path)
    directory = args.output_dir or ROOT / f"outputs/accident/presentation_events_{args.cohort}"
    if (directory / "batch.json").exists():
        validate_batch(directory, args.protocol, protocol, args.cohort, selection_hash)
        print("Complete event batch and hashes verified", flush=True)
        return
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    versions = {"torch": torch.__version__, "opencv": cv2.__version__, "ultralytics": ultralytics.__version__}
    if versions != protocol["extraction_versions"]:
        raise ValueError("Event extraction runtime differs from the matched training study")
    audit = json.loads((ROOT / "docs/results/cadence_source_audit.json").read_text(encoding="utf-8"))
    if cv2.__version__ != audit["software"]["opencv"]:
        raise ValueError("Decoder version differs from the verified source audit")
    ffprobe = Path("C:/ffmpeg/bin/ffprobe.exe")
    if sha256(ffprobe) != protocol["ffprobe_sha256"]:
        raise ValueError("FFprobe executable differs from the frozen clock contract")
    paths = cohort_paths(protocol, args.cohort)
    if args.smoke_clips:
        paths = paths[:args.smoke_clips]
    directory.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps({"protocol": sha256(args.protocol), "cohort": args.cohort,
                                           "versions": versions, "device": args.device, "smoke_clips": args.smoke_clips,
                                           "selection_sha256": selection_hash}, sort_keys=True).encode()).hexdigest()
    sources = {row["path"]: row for row in protocol["input_media"]}
    detector = YOLO(str(ROOT / "yolov8n.pt"))
    predictors = {key: IncidentTcnPredictor.from_checkpoint(ROOT / definition["path"], "cpu") for key, definition in protocol["models"].items()}
    records = {}
    started = datetime.now(timezone.utc).isoformat()
    for offset, name in enumerate(paths):
        stem = hashlib.sha256(name.encode()).hexdigest()
        report_path, features_path = directory / (stem + ".json"), directory / (stem + ".npz")
        video = ROOT / "data/raw/accident/full/extracted" / name
        if sha256(video) != sources[name]["sha256"]:
            raise ValueError("Event video changed after cohort freezing")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["signature"] != signature or report["features_sha256"] != sha256(features_path):
                raise ValueError("Cached continuous replay signature or features changed")
        else:
            timestamps = probe_times(video, ffprobe)
            if hashlib.sha256(json.dumps(timestamps).encode()).hexdigest() != sources[name]["presentation_times_sha256"]:
                raise ValueError("Source presentation timestamps changed")
            normalized = normalized_presentation_times(timestamps)
            annotation = protocol["annotations"][name]
            if abs(normalized[annotation["supplied_event_frame"]] - annotation["event_time_sec"]) > 1e-4:
                raise ValueError("Supplied annotation no longer maps to its frozen PTS")
            matrix, windows, rejected = clip_features(video, timestamps, detector, args.device)
            np.savez_compressed(features_path, features=matrix, feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES))
            scores = {}
            for key, predictor in predictors.items():
                columns = [CANDIDATE_MOTION_FEATURE_NAMES.index(feature) for feature in predictor.feature_names]
                scores[key] = predictor.predict_probabilities(matrix[:, :, columns]).tolist() if len(matrix) else []
            report = {"path": name, "signature": signature, "protocol_sha256": sha256(args.protocol),
                      "source_sha256": sources[name]["sha256"], "features_sha256": sha256(features_path),
                      "clock": "normalized_presentation_time", "annotation_free_inference": True,
                      "window_starts": [row["start"] for row in windows], "rejected_window_starts": rejected,
                      "window_frame_indices": [row["indices"] for row in windows],
                      "window_end_times_sec": [row["end_time_sec"] for row in windows], "scores": scores}
            temporary = report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report) + "\n", encoding="utf-8")
            temporary.replace(report_path)
        records[name] = {"report": report_path.name, "features": features_path.name,
                         "report_sha256": sha256(report_path), "features_sha256": sha256(features_path)}
        print(f"{args.cohort}: {offset + 1}/{len(paths)} clips; {len(report['window_starts'])} valid, {len(report['rejected_window_starts'])} rejected contexts", flush=True)
    batch = {"protocol_sha256": sha256(args.protocol), "cohort": args.cohort, "signature": signature,
             "selection_sha256": selection_hash, "smoke_clips": args.smoke_clips, "versions": versions,
             "device": args.device, "threads": 2, "clips": records,
             "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (directory / "batch.json").write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
