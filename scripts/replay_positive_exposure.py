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
from scripts.prepare_presentation_protocol import sha256, probe_times
from scripts.replay_presentation_events import load_event_protocol, clip_features
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
from illegal_parking.incident_presentation import normalized_presentation_times
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor


def load_study():
    path = ROOT / "docs/results/positive_exposure_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    original = ROOT / "docs/results/presentation_event_protocol.json"
    parent = load_event_protocol(original)
    if protocol["status"] != "frozen_before_new_continuous_event_replay" or protocol["parent_event_protocol_sha256"] != sha256(original):
        raise ValueError("A matching frozen positive-context study is required")
    if any(protocol[field] != parent[field] for field in ("models", "sampling", "inference", "timing", "detector_sha256", "extraction_versions")):
        raise ValueError("This follow-up must retain the original model and physical-time inference contracts")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("The frozen positive-context implementation changed")
    if sha256(ROOT / "docs/results/presentation_event_selection.json") != protocol["original_selection_sha256"]:
        raise ValueError("Original onset-only baseline thresholds changed")
    selection_path = ROOT / "docs/results/positive_exposure_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != sha256(path) or selection["calibration_batch_sha256"] != protocol["calibration_batch_sha256"]:
        raise ValueError("Training-only selection provenance changed")
    excluded = set(protocol["excluded_previously_replayed_clips"])
    for definition in protocol["cohorts"].values():
        if len(definition["test_clip_paths"]) != 100 or len(set(definition["test_clip_paths"])) != 100:
            raise ValueError("Each frozen test cohort must retain exactly 100 unique sources")
        if set(definition["test_clip_paths"]) & (excluded | set(definition["calibration_clip_paths"])):
            raise ValueError("New event replays must exclude all declared prior sources and calibration clips")
    return path, protocol, selection_path, selection


def validate_execution(protocol_path, selection_path):
    path = ROOT / "docs/results/positive_exposure_execution.json"
    execution = json.loads(path.read_text(encoding="utf-8"))
    if execution["protocol_sha256"] != sha256(protocol_path) or execution["selection_sha256"] != sha256(selection_path):
        raise ValueError("Execution appendix no longer binds the selected study")
    for name, expected in execution["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Frozen positive-context worker or publisher changed")
    return path, execution


def load_batch(protocol_path, protocol, selection_path):
    directory = ROOT / "outputs/accident/positive_exposure_test"
    batch_path = directory / "batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    execution_path, _ = validate_execution(protocol_path, selection_path)
    if (batch["protocol_sha256"] != sha256(protocol_path) or batch["selection_sha256"] != sha256(selection_path)
            or batch["execution_sha256"] != sha256(execution_path)):
        raise ValueError("New event batch has changed provenance")
    expected = {name for definition in protocol["cohorts"].values() for name in definition["test_clip_paths"]}
    if set(batch["clips"]) != expected:
        raise ValueError("New event batch must contain every frozen source without replacement")
    if batch["versions"] != protocol["extraction_versions"] or batch["device"] != "0" or batch["threads"] != 2:
        raise ValueError("Event batch runtime differs from the frozen extraction contract")
    sources = {row["path"]: row["sha256"] for row in protocol["input_media"]}
    reports = []
    for name in sorted(expected):
        record = batch["clips"][name]
        if sha256(directory / record["report"]) != record["report_sha256"] or sha256(directory / record["features"]) != record["features_sha256"]:
            raise ValueError("New event report or feature bytes changed")
        report = json.loads((directory / record["report"]).read_text(encoding="utf-8"))
        if (report["path"] != name or report["protocol_sha256"] != batch["protocol_sha256"] or report["signature"] != batch["signature"] or report["source_sha256"] != sources[name]
                or report["features_sha256"] != record["features_sha256"] or set(report["scores"]) != set(protocol["models"])):
            raise ValueError("Per-clip identity or model predictions do not bind this batch")
        reports.append(report)
    return directory, batch, reports


def main():
    parser = argparse.ArgumentParser(description="Replay the separately frozen positive-context event cohort without changing any reference model.")
    parser.add_argument("--freeze-execution", action="store_true")
    args = parser.parse_args()
    protocol_path, protocol, selection_path, _ = load_study()
    directory = ROOT / "outputs/accident/positive_exposure_test"
    execution_path = ROOT / "docs/results/positive_exposure_execution.json"
    if args.freeze_execution:
        if execution_path.exists() or directory.exists():
            raise FileExistsError("Freeze the execution appendix before any new event replay")
        record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256(protocol_path),
                  "selection_sha256": sha256(selection_path),
                  "implementation_sha256": {name: sha256(ROOT / name) for name in ("scripts/replay_positive_exposure.py", "scripts/publish_positive_exposure.py", "scripts/audit_presentation_event_predictions.py")},
                  "role": "Execution appendix to the already frozen sampling/cohort/calibration protocol; no scientific contracts changed"}
        execution_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(record, indent=2))
        return
    validate_execution(protocol_path, selection_path)
    if (directory / "batch.json").exists():
        load_batch(protocol_path, protocol, selection_path)
        print("Existing complete positive-context test batch verified", flush=True)
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
        raise ValueError("Feature runtime differs from the frozen paired study")
    ffprobe = Path("C:/ffmpeg/bin/ffprobe.exe")
    if sha256(ffprobe) != protocol["ffprobe_sha256"]:
        raise ValueError("FFprobe differs from the frozen source-clock contract")
    names = sorted({name for definition in protocol["cohorts"].values() for name in definition["test_clip_paths"]})
    sources = {row["path"]: row for row in protocol["input_media"]}
    signature = hashlib.sha256(json.dumps({"protocol": sha256(protocol_path), "selection": sha256(selection_path),
                                           "execution": sha256(execution_path), "versions": versions, "device": "0"}, sort_keys=True).encode()).hexdigest()
    detector = YOLO(str(ROOT / "yolov8n.pt"))
    predictors = {key: IncidentTcnPredictor.from_checkpoint(ROOT / definition["path"], "cpu") for key, definition in protocol["models"].items()}
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    records = {}
    for offset, name in enumerate(names):
        stem = hashlib.sha256(name.encode()).hexdigest()
        report_path, features_path = directory / (stem + ".json"), directory / (stem + ".npz")
        video = ROOT / "data/raw/accident/full/extracted" / name
        if sha256(video) != sources[name]["sha256"]:
            raise ValueError("New-cohort source video changed")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["signature"] != signature or report["features_sha256"] != sha256(features_path):
                raise ValueError("Partial replay cache changed")
        else:
            timestamps = probe_times(video, ffprobe)
            if hashlib.sha256(json.dumps(timestamps).encode()).hexdigest() != sources[name]["presentation_times_sha256"]:
                raise ValueError("New source packet timestamps changed")
            annotation = protocol["annotations"][name]
            if abs(normalized_presentation_times(timestamps)[annotation["supplied_event_frame"]] - annotation["event_time_sec"]) > 1e-4:
                raise ValueError("Supplied annotation frame no longer maps to its frozen PTS")
            features, windows, rejected = clip_features(video, timestamps, detector, "0")
            temporary = features_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=features, feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES))
            temporary.replace(features_path)
            scores = {}
            for key, predictor in predictors.items():
                columns = [CANDIDATE_MOTION_FEATURE_NAMES.index(feature) for feature in predictor.feature_names]
                scores[key] = predictor.predict_probabilities(features[:, :, columns]).tolist() if len(features) else []
            report = {"path": name, "source_sha256": sources[name]["sha256"], "signature": signature,
                      "protocol_sha256": sha256(protocol_path),
                      "features_sha256": sha256(features_path), "clock": "normalized_presentation_time", "annotation_free_inference": True,
                      "window_starts": [row["start"] for row in windows], "window_end_times_sec": [row["end_time_sec"] for row in windows],
                      "window_frame_indices": [row["indices"] for row in windows],
                      "rejected_window_starts": rejected, "scores": scores}
            temporary = report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report) + "\n", encoding="utf-8")
            temporary.replace(report_path)
        records[name] = {"report": report_path.name, "report_sha256": sha256(report_path), "features": features_path.name, "features_sha256": sha256(features_path)}
        print(f"Positive-context replay: {offset + 1}/{len(names)} sources", flush=True)
    batch = {"protocol_sha256": sha256(protocol_path), "selection_sha256": sha256(selection_path), "execution_sha256": sha256(execution_path),
             "signature": signature, "clips": records, "versions": versions, "device": "0", "threads": 2,
             "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (directory / "batch.json").write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
