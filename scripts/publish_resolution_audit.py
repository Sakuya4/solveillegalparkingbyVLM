from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_batch import select_evaluation_clips
from illegal_parking.incident_evaluation import evaluate_incident_report, match_report_to_accident_clip, summarize_incident_evaluations
from illegal_parking.incident_tcn_inference import persistent_positive_flags
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def load_batch(directory, clips, split):
    path = directory / "batch_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["failure_count"] or summary["completed_clips"] + summary["resumed_clips"] != 100 or summary["split_scheme"] != split or summary["sampling_seed"] != 42:
        raise ValueError("Resolution audit requires complete matched 100-clip batches")
    reports = {}
    for name in summary["report_paths"]:
        path = directory / "reports" / Path(name.replace("\\", "/")).name
        report = json.loads(path.read_text(encoding="utf-8"))
        clip = match_report_to_accident_clip(report, clips)
        key = clip.relative_path.as_posix()
        if key in reports:
            raise ValueError("Duplicate report in resolution batch")
        reports[key] = (report, clip, hashlib.sha256(path.read_bytes()).hexdigest())
    expected = {clip.relative_path.as_posix() for clip in select_evaluation_clips(clips, split_scheme=split, max_clips=100, seed=42)}
    if set(reports) != expected:
        raise ValueError("Resolution batch differs from the fixed test selection")
    return summary, reports


def align_window_grid(report, starts, hop_steps):
    original_starts = report["window_starts"]
    if len(set(original_starts)) != len(original_starts) or len(original_starts) != len(report["window_probabilities"]):
        raise ValueError("Source windows must have unique starts and aligned probabilities")
    index = {start: offset for offset, start in enumerate(original_starts)}
    if any(start not in index for start in starts):
        raise ValueError("Source report does not contain the requested window grid")
    probabilities = [report["window_probabilities"][index[start]] for start in starts]
    return {**report, "window_starts": list(starts), "window_probabilities": probabilities,
            "hop_steps": hop_steps, "window_review_flags": persistent_positive_flags(probabilities, report["threshold"], report["min_positive_windows"]).tolist()}


def main():
    parser = argparse.ArgumentParser(description="Publish paired spatial preprocessing sensitivity without replacing legacy results.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/event_resolution_audit.json")
    args = parser.parse_args()
    metadata = ROOT / "data/raw/accident/full/extracted/metadata-real.csv"
    clips = load_accident_manifest(metadata)
    rows, provenance = [], {}
    for split, old_name in (("iid", "event_batch_iid_100"), ("geographic", "event_batch_geographic_100_20261016")):
        old_directory = ROOT / "outputs/accident" / old_name
        new_directory = ROOT / "outputs/accident" / f"event_resolution320_{split}_100"
        old_summary, old = load_batch(old_directory, clips, split)
        new_summary, new = load_batch(new_directory, clips, split)
        model_hash = hashlib.sha256(Path(old_summary["model"]).read_bytes()).hexdigest()
        if new_summary["model_sha256"] != model_hash:
            raise ValueError("Resolution audit models must be identical")
        for key in sorted(old):
            a, b = old[key][0], new[key][0]
            for field in ("source_fps", "sampled_frames", "sampled_fps", "target_fps", "threshold", "sequence_steps"):
                if a[field] != b[field]:
                    raise ValueError(f"Non-spatial protocol differs: {key}, {field}")
            if b["feature_width_limit"] != 320 or b["display_width"] != 640 or b["feature_resolution_protocol"] != "source_downsample_no_upscale":
                raise ValueError("Aligned audit must use source-downsampled features and unchanged display width")
            if b["hop_steps"] != 15 or b["window_starts"] != list(range(0, b["sampled_frames"] - b["sequence_steps"], 15)):
                raise ValueError("The audit uses a fixed 15-step decision grid")
            old[key] = (align_window_grid(a, b["window_starts"], 15), old[key][1], old[key][2])
        provenance[split] = {"model_sha256": model_hash, "detector_sha256": new_summary["detector_sha256"],
                             "batch_sha256": {name: hashlib.sha256((directory / "batch_summary.json").read_bytes()).hexdigest() for name, directory in (("legacy640", old_directory), ("aligned320", new_directory))},
                             "reports": [{"clip": key, "legacy640_sha256": old[key][2], "aligned320_sha256": new[key][2]} for key in sorted(old)]}
        for profile, gate in (("responsive", 1), ("conservative", 2)):
            outcomes = []
            for batch in (old, new):
                current = []
                for key in sorted(batch):
                    original, clip, _ = batch[key]
                    report = {**original, "window_review_flags": persistent_positive_flags(original["window_probabilities"], original["threshold"], gate).tolist()}
                    current.append(evaluate_incident_report(report, clip))
                outcomes.append(current)
            a, b = [np.asarray([item.detected for item in current]) for current in outcomes]
            rows.append({"split": split, "profile": profile, "min_positive_windows": gate,
                         "legacy640": summarize_incident_evaluations(outcomes[0]), "aligned320": summarize_incident_evaluations(outcomes[1]),
                         "paired_outcomes": {"both_detected": int((a & b).sum()), "gained_at_320": int((~a & b).sum()), "lost_at_320": int((a & ~b).sum()), "neither_detected": int((~a & ~b).sum())},
                         "recall_difference_320_minus_640": paired_cluster_coverage_interval(np.ones(len(a)), b.astype(float), a.astype(float))})
    training_report = ROOT / "data/processed/accident/candidate_roi_feature_500_report.json"
    training = json.loads(training_report.read_text(encoding="utf-8"))
    result = {"metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(), "training_feature_report_sha256": hashlib.sha256(training_report.read_bytes()).hexdigest(),
              "training_preprocessing": {key: training[key] for key in ("target_width", "frame_stride", "image_size", "confidence")},
              "protocol": "Paired fixed-model spatial preprocessing sensitivity on identical sampled frames, calibrated thresholds, 15-step window hops, official test selections and -1/+3-second timing tolerances",
              "window_alignment": "Original IID640 reports use hop 5; select their existing starts divisible by 15 and recompute persistence. Geographic640 and both320 batches already use hop15. Legacy published hop5 metrics are retained separately.",
              "intervention": "Legacy features/tracking at 640-pixel display resolution versus features/tracking decoded directly at a maximum width of 320 without upscaling; display remains 640",
              "uncertainty": "Paired clip bootstrap of recall difference, 2,000 repetitions, seed 42; fixed model and threshold",
              "interpretation": "Spatial preprocessing changes the operating distribution; neither protocol is selected by test performance, and published reference binaries and legacy evaluations remain unchanged",
              "provenance": provenance, "comparisons": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "profile", "protocol", "recall", "early_episodes", "late_episodes", "delay_median_sec", "delay_p95_sec"])
        writer.writeheader()
        for row in rows:
            for name in ("legacy640", "aligned320"):
                data = row[name]
                writer.writerow({"split": row["split"], "profile": row["profile"], "protocol": name, "recall": data["event_recall"], "early_episodes": data["early_alert_episodes"], "late_episodes": data["late_alert_episodes"], "delay_median_sec": data["trigger_delay_sec"]["median"], "delay_p95_sec": data["trigger_delay_sec"]["p95"]})
    print(json.dumps([{key: row[key] for key in ("split", "profile", "paired_outcomes", "recall_difference_320_minus_640")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
