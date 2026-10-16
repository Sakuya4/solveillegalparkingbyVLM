from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

LABELS = ("uncertain", "incident_present", "normal")
MAX_OBSERVATION_GAP_SEC = 1.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review_observation(report_path: Path):
    import cv2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("streaming_inference"):
        raise ValueError("Observation review requires a continuous streaming report")
    video = Path(report["clean_video"])
    if not video.is_absolute():
        video = report_path.parent / video
    if digest(video) != report["clean_video_sha256"]:
        raise ValueError("Clean observation video hash mismatch")
    times = report["frame_times_sec"]
    if len(times) < 2 or any(not math.isfinite(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation timestamps must be finite and strictly increasing")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(times):
            raise ValueError("Clean playback frames do not match source-clock timestamps")
    finally:
        capture.release()
    return report, video.resolve()


def save_observation_review(report_path: Path, output: Path, *, reviewer_id: str, label: str,
                            all_frames_reviewed: bool, model_outputs_not_consulted: bool):
    if not reviewer_id.strip() or label not in LABELS:
        raise ValueError("A reviewer identifier and supported observation label are required")
    if not isinstance(all_frames_reviewed, bool) or not isinstance(model_outputs_not_consulted, bool):
        raise ValueError("Review attestations must be Boolean")
    report, video = load_review_observation(report_path)
    record = {"schema_version": 1, "review_method": "reviewer_video_annotation",
              "reviewer_id": reviewer_id.strip(), "label": label,
              "all_frames_reviewed": all_frames_reviewed, "model_outputs_not_consulted": model_outputs_not_consulted,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_report": str(report_path.resolve()), "source_report_sha256": digest(report_path),
              "clean_video": str(video), "clean_video_sha256": report["clean_video_sha256"],
              "sampled_frames": len(report["frame_times_sec"]),
              "clock": report["event_clock"], "max_observation_gap_sec": MAX_OBSERVATION_GAP_SEC}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    return record


def summarize_observation_reviews(paths: list[Path]):
    groups = defaultdict(list)
    excluded, sources = [], []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != 1 or record.get("review_method") != "reviewer_video_annotation" or record.get("label") not in LABELS or not isinstance(record.get("reviewer_id"), str) or not record["reviewer_id"].strip():
            raise ValueError("Unsupported or unattributed observation review")
        source = Path(record["source_report"])
        if digest(source) != record["source_report_sha256"]:
            raise ValueError("Source report changed after annotation")
        report, _ = load_review_observation(source)
        if record["clean_video_sha256"] != report["clean_video_sha256"] or record["max_observation_gap_sec"] != MAX_OBSERVATION_GAP_SEC:
            raise ValueError("Annotation media or observation-gap protocol mismatch")
        if record["sampled_frames"] != len(report["frame_times_sec"]):
            raise ValueError("Annotation frame count differs from its source report")
        sources.append({"annotation_sha256": digest(path), "source_report_sha256": digest(source), "clean_video_sha256": record["clean_video_sha256"]})
        if record.get("all_frames_reviewed") is not True or record.get("model_outputs_not_consulted") is not True or record["label"] == "uncertain":
            excluded.append({"annotation_sha256": digest(path), "reason": "Incomplete, model-informed, or uncertain review"})
            continue
        groups[record["clean_video_sha256"]].append((record, report))
    observations, conflicts = [], []
    contracts = set()
    for media_hash, entries in groups.items():
        labels = {record["label"] for record, _ in entries}
        if len(labels) > 1:
            conflicts.append(media_hash)
            continue
        if labels != {"normal"}:
            continue
        local_contracts = {(report["model_sha256"], report["detector_sha256"], report.get("feature_width_limit"), report["hop_steps"], report["min_positive_windows"], report["threshold"]) for _, report in entries}
        if len(local_contracts) != 1:
            raise ValueError("The same observation has conflicting inference contracts")
        clocks = {(tuple(report["frame_times_sec"]), tuple(event["trigger_time_sec"] for event in report["events"]), report.get("source_mode"), report["event_clock"]) for _, report in entries}
        if len(clocks) != 1:
            raise ValueError("Duplicate media has conflicting observation clocks or alert episodes")
        record, report = entries[0]
        # File replay supports workflow checks, not the camera-hour endpoint.
        if report.get("source_mode") == "file":
            excluded.append({"clean_video_sha256": media_hash, "reason": "File replay is not camera exposure"})
            continue
        if report.get("source_mode") not in {"page", "mjpeg", "snapshot"}:
            raise ValueError("Normal exposure requires an identified camera transport")
        contracts.update(local_contracts)
        times = report["frame_times_sec"]
        intervals = [(a, b) for a, b in zip(times, times[1:]) if b - a <= MAX_OBSERVATION_GAP_SEC]
        seconds = sum(b - a for a, b in intervals)
        alerts = sum(any(a <= event["trigger_time_sec"] <= b for a, b in intervals) for event in report["events"])
        observations.append({"clean_video_sha256": media_hash, "source_report_sha256": record["source_report_sha256"],
                             "observed_sec": seconds, "excluded_gap_sec": times[-1] - times[0] - seconds,
                             "normal_alert_episodes": alerts, "reviewer_records": len(entries)})
    if len(contracts) > 1:
        raise ValueError("Evaluate distinct inference contracts as separate exposure cohorts")
    seconds = sum(row["observed_sec"] for row in observations)
    count = sum(row["normal_alert_episodes"] for row in observations)
    hours = seconds / 3600
    upper = None
    if hours:
        if count == 0:
            upper = -math.log(0.05) / hours
        else:
            from scipy.stats import chi2
            upper = float(chi2.ppf(0.95, 2 * (count + 1)) / 2 / hours)
    return {"schema_version": 1, "normal_camera_hours": hours, "normal_alert_episodes": count,
            "alerts_per_normal_camera_hour": count / hours if hours else None,
            "poisson_one_sided_upper_95_per_hour": upper, "normal_observations": observations,
            "conflicting_observations": conflicts, "excluded_reviews": excluded, "source_hashes": sources,
            "max_observation_gap_sec": MAX_OBSERVATION_GAP_SEC,
            "inference_contract": list(next(iter(contracts))) if contracts else None,
            "scope": "Reviewer-attested sampled-video labels; duplicate media count once; gaps above one second, uncertain/model-informed/incomplete reviews, file replay and conflicting labels are excluded. Poisson bounds assume a stationary event rate."}
