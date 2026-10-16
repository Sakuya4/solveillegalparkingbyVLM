from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES, candidate_motion_sequence_matrix, compute_candidate_motion_step
from illegal_parking.incident_presentation import read_presentation_frames
from illegal_parking.incident_ultralytics import track_vehicle_frames


def load_manifest(path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    windows_path = path.parent / protocol["windows_file"]
    if protocol["status"] != "frozen_before_feature_extraction" or sha256(windows_path) != protocol["windows_sha256"]:
        raise ValueError("A frozen, hash-matched presentation manifest is required")
    rows = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != protocol["paired_windows"]:
        raise ValueError("Frozen window count differs from the protocol")
    if len({(row["path"], row["label"]) for row in rows}) != len(rows) or len({row["path"] for row in rows}) != protocol["paired_clips"]:
        raise ValueError("Frozen rows must have unique clip/label identities and matching clip counts")
    return protocol, rows


def clip_frame_support(rows):
    support = {}
    for row in rows:
        entries = [(row["supplied_event_frame"], row["event_time_sec"])]
        for arm in ("legacy", "fixed"):
            entries.extend(zip(row[arm + "_indices"], row[arm + "_times_sec"]))
        for index, timestamp in entries:
            if index in support and abs(support[index] - timestamp) > 1e-9:
                raise ValueError("A source frame has conflicting frozen presentation timestamps")
            support[index] = timestamp
    indices = sorted(support)
    return indices, [support[index] for index in indices]


def main():
    parser = argparse.ArgumentParser(description="Extract paired presentation-study features without changing reference artifacts.")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/presentation_protocol.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/accident/full/extracted")
    parser.add_argument("--arm", choices=("legacy", "fixed"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--smoke-clips", type=int, help="Use only the first N clips that are training in both official splits; not eligible for fitting")
    args = parser.parse_args()
    if args.smoke_clips is not None and args.smoke_clips <= 0:
        parser.error("Smoke clip count must be positive")
    protocol, rows = load_manifest(args.protocol)
    protocol_sha = sha256(args.protocol)
    if args.smoke_clips:
        selected = sorted({row["path"] for row in rows if row["iid_split"] == row["geographic_split"] == "train"})[:args.smoke_clips]
        rows = [row for row in rows if row["path"] in selected]
    if not rows:
        raise ValueError("The requested extraction has no eligible rows")
    sequence_path = args.output_dir / "sequences.npz"
    report_path = args.output_dir / "extraction.json"
    if sequence_path.exists() or report_path.exists():
        raise FileExistsError("Completed extraction artifacts are immutable; use a new output directory")
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.manual_seed(protocol["training"]["seed"])
    source_audit_path = ROOT / "docs/results/cadence_source_audit.json"
    audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if sha256(source_audit_path) != protocol["source_audit_sha256"] or cv2.__version__ != audit["software"]["opencv"]:
        raise ValueError("Decoder runtime differs from the verified source-audit environment")
    detector_path = ROOT / "yolov8n.pt"
    if sha256(detector_path) != protocol["detector_sha256"]:
        raise ValueError("Detector differs from the frozen protocol")
    implementation = {name: sha256(ROOT / name) for name in protocol["implementation_sha256"]}
    if implementation != protocol["implementation_sha256"]:
        raise ValueError("Frozen feature or training implementation changed")
    implementation[Path(__file__).relative_to(ROOT).as_posix()] = sha256(Path(__file__))
    versions = {"torch": torch.__version__, "opencv": cv2.__version__, "ultralytics": ultralytics.__version__}
    signature = hashlib.sha256(json.dumps({"protocol": protocol_sha, "arm": args.arm, "implementation": implementation,
                                          "versions": versions, "device": args.device}, sort_keys=True).encode()).hexdigest()
    cache = args.output_dir / "cache" / signature
    cache.mkdir(parents=True, exist_ok=True)
    inputs = {row["path"]: row["sha256"] for row in protocol["input_media"]}
    verified = set()
    grouped_rows = {}
    for row in rows:
        grouped_rows.setdefault(row["path"], []).append(row)
    decoded_path, decoded_frames = None, {}
    detector = YOLO(str(detector_path))
    tensors = []
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    for index, row in enumerate(rows):
        video = args.dataset_root / row["path"]
        if row["path"] not in verified:
            if sha256(video) != inputs[row["path"]]:
                raise ValueError("Source media differs from the frozen protocol")
            verified.add(row["path"])
        cached = cache / f"{index:05d}.npz"
        if cached.exists():
            with np.load(cached, allow_pickle=False) as data:
                if str(data["row_sha256"]) != hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest():
                    raise ValueError("Cached sequence refers to a different frozen row")
                tensor = data["features"].copy()
        else:
            indices = row[args.arm + "_indices"]
            if decoded_path != row["path"]:
                support, timestamps = clip_frame_support(grouped_rows[row["path"]])
                decoded_frames = dict(zip(support, read_presentation_frames(video, support, timestamps, protocol["training"]["feature_width"])))
                decoded_path = row["path"]
            frames = [decoded_frames[index] for index in indices]
            tracks = track_vehicle_frames(detector, frames, indices, device=args.device,
                                          image_size=protocol["training"]["image_size"], confidence=protocol["training"]["confidence"])
            steps = [compute_candidate_motion_step(frames[offset - 1], frames[offset], tracks[indices[offset]]) for offset in range(1, len(frames))]
            tensor = candidate_motion_sequence_matrix(steps)
            temporary = cached.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=tensor, row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
            temporary.replace(cached)
        if tensor.shape != (15, len(CANDIDATE_MOTION_FEATURE_NAMES)) or not np.all(np.isfinite(tensor)):
            raise ValueError("Presentation features have invalid shape or values")
        tensors.append(tensor)
        if (index + 1) % 50 == 0:
            print(f"{args.arm}: {index + 1}/{len(rows)} windows", flush=True)
    fields = ("path", "label", "target", "collision_type", "region", "quality", "iid_split", "geographic_split", "iid_partition", "geographic_partition", "event_time_sec")
    np.savez_compressed(sequence_path, features=np.stack(tensors).astype(np.float32), feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES),
                        frame_indices=np.asarray([row[args.arm + "_indices"] for row in rows]),
                        frame_times_sec=np.asarray([row[args.arm + "_times_sec"] for row in rows]),
                        protocol_sha256=protocol_sha, arm=args.arm, **{key: np.asarray([row[key] for row in rows]) for key in fields})
    report = {"protocol_sha256": protocol_sha, "sequences_sha256": sha256(sequence_path), "arm": args.arm,
              "rows": len(rows), "clips": len(verified), "smoke_clips": args.smoke_clips,
              "feature_shape": list(np.stack(tensors).shape), "implementation_sha256": implementation, "versions": versions,
              "started_at_utc": started_utc, "elapsed_sec": time.perf_counter() - started,
              "threads": 2, "device": args.device, "detector_sha256": protocol["detector_sha256"]}
    report["decode_contract"] = "Reuse the same verified per-clip support as protocol preparation; only current-arm window frames enter detection and motion features"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
