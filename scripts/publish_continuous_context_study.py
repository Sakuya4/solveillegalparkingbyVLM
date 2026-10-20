from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
import numpy as np
from scripts.run_continuous_context_study import (PROTOCOL, DIRECTORY, SELECTION, load_study, load_models,
                                                 load_batch, read_bound, sha, dump, signature, now)
from illegal_parking.incident_context_study import model_reports, operating_point, select_policy, validate_features
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor
from illegal_parking.incident_presentation_events import evaluate_presentation_event
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def csv_output(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "rows": len(rows)}


def main():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    threadpool_limits(2)
    protocol, execution, _ = load_study()
    definitions = load_models()
    selection = read_bound(SELECTION)
    cal_batch, cal = load_batch("calibration")
    test_batch, test = load_batch("test")
    if selection["calibration_batch_sha256"] != sha(DIRECTORY / "calibration/batch.json"):
        raise ValueError("Selected policies no longer bind calibration features")
    if datetime.fromisoformat(test_batch["started_at_utc"]) < datetime.fromisoformat(selection["selected_at_utc"]):
        raise ValueError("Test replay predates all frozen operating policies")
    output = ROOT / "docs/results/continuous_context_results.json"
    if any((ROOT / "docs/results" / name).exists() for name in
           ("continuous_context_results.json", "continuous_context_results.csv", "continuous_context_outcomes.csv",
            "continuous_context_scores.csv", "continuous_context_calibration.csv")):
        raise FileExistsError("Do not overwrite a published study")
    checked, max_error, contexts = 0, 0., []
    predictors = {key: IncidentTcnPredictor.from_checkpoint(ROOT / value["path"], "cpu") for key, value in definitions.items()}
    for cohort, batch, reports in (("calibration", cal_batch, cal), ("test", test_batch, test)):
        for report in reports:
            record = batch["clips"][report["path"]]
            with np.load(DIRECTORY / cohort / record["features"], allow_pickle=False) as data:
                matrix = validate_features(data["features"].copy(), len(report["window_starts"]))
                if data["feature_names"].tolist() != list(CANDIDATE_MOTION_FEATURE_NAMES):
                    raise ValueError("Saved replay feature channels changed")
            for key, predictor in predictors.items():
                columns = [CANDIDATE_MOTION_FEATURE_NAMES.index(name) for name in predictor.feature_names]
                actual = predictor.predict_probabilities(matrix[:, :, columns]) if len(matrix) else np.empty(0)
                scores = np.asarray(report["scores"][key])
                if actual.shape != scores.shape or not np.all(np.isfinite(scores)):
                    raise ValueError("Saved replay scores are malformed")
                error = float(np.max(np.abs(actual - scores))) if len(actual) else 0.
                if error > 1e-7:
                    raise ValueError("Checkpoint predictions do not reproduce")
                checked += len(actual)
                max_error = max(max_error, error)
                split = key.rsplit("_", 1)[-1]
                if report["path"] not in protocol["memberships"][split][cohort + "_clip_paths"]:
                    continue
                event = protocol["annotations"][report["path"]]["event_time_sec"]
                contexts.extend({"cohort": cohort, "model": key, "split": split, "path": report["path"],
                                 "source_sha256": report["source_sha256"], "start": start, "end_pts_sec": timestamp,
                                 "event_pts_sec": event, "probability": score}
                                for start, timestamp, score in zip(report["window_starts"], report["window_end_times_sec"], scores))
    models, traces, curves, outcomes = [], [], [], {}
    for key, definition in definitions.items():
        chosen_cal = model_reports(protocol, cal, key, "calibration")
        curve = [operating_point(chosen_cal, protocol["annotations"], threshold) for threshold in protocol["calibration"]["threshold_grid"]]
        selected = select_policy(curve, protocol["calibration"])
        if selected != selection["policies"][key]["selected"] or curve != selection["policies"][key]["curve"]:
            raise ValueError("Every calibration curve and selected threshold must reproduce")
        curves.extend({"model": key, "threshold": row["threshold"], "event_recall": row["summary"]["event_recall"],
                       "off_target_per_clip": row["summary"]["off_target_episodes_per_clip"],
                       "prefix_support": row["pre_event"]["clips_with_support"],
                       "prefix_macro": row["pre_event"]["macro_positive_fraction"]} for row in curve)
        chosen = model_reports(protocol, test, key, "test")
        threshold = selected["threshold"] if selected else None
        result = operating_point(chosen, protocol["annotations"], threshold) if selected else None
        fit = read_bound(DIRECTORY / definition["fit_report"])
        models.append({"model": key, "model_sha256": definition["sha256"], "selected_threshold": threshold,
                       "calibration": selected, "test": result, "fit": fit,
                       "test_episode_cap_exceeded": result["summary"]["off_target_episodes_per_clip"] > .5 if result else None,
                       "test_prefix_cap_exceeded": result["pre_event"]["macro_positive_fraction"] > .2 if result and result["pre_event"]["macro_positive_fraction"] is not None else None})
        if selected:
            evaluations = [evaluate_presentation_event(report, protocol["annotations"][report["path"]], threshold) for report in chosen]
            outcomes[key] = np.asarray([row.detected for row in evaluations], dtype=float)
            traces.extend(outcome_row(key, "test", report, evaluation, threshold) for report, evaluation in zip(chosen, evaluations))
        else:
            traces.extend(outcome_row(key, "test", report, None, None) for report in chosen)
    controls = {}
    for split in protocol["splits"]:
        key = next(key for key in definitions if key.endswith("_" + split))
        reports = model_reports(protocol, test, key, "test")
        controls[split] = {}
        for positive, name in ((True, "clock_only_positive"), (False, "always_negative")):
            chosen = [{**report, "window_probabilities": [float(positive)] * len(report["window_starts"])} for report in reports]
            controls[split][name] = operating_point(chosen, protocol["annotations"], .5)
            traces.extend(outcome_row(name + "_" + split, "test", report,
                                     evaluate_presentation_event(report, protocol["annotations"][report["path"]], .5), .5) for report in chosen)
    contrasts = []
    for feature in ("candidate", "global"):
        for split in protocol["splits"]:
            a, b = f"continuous_{feature}_{split}", f"anchored_{feature}_{split}"
            interval = paired_cluster_coverage_interval(np.ones(len(outcomes[a])), outcomes[a], outcomes[b]) if a in outcomes and b in outcomes else None
            contrasts.append({"feature": feature, "split": split, "contrast": "continuous minus anchored event recall", "paired_interval": interval})
    base = ROOT / "docs/results"
    files = {"outcomes": csv_output(base / "continuous_context_outcomes.csv", traces),
             "contexts": csv_output(base / "continuous_context_scores.csv", contexts),
             "calibration": csv_output(base / "continuous_context_calibration.csv", curves)}
    dump(output, {**signature(), "selection_sha256": sha(SELECTION), "models_sha256": sha(DIRECTORY / "models.json"),
                  "calibration_batch_sha256": sha(DIRECTORY / "calibration/batch.json"), "test_batch_sha256": sha(DIRECTORY / "test/batch.json"),
                  "published_at_utc": now(), "models": models, "contrasts": contrasts, "controls": controls, "files": files,
                  "prediction_verification": {"checked_probabilities": checked, "maximum_absolute_error": max_error, "tolerance": 1e-7},
                  "scope": protocol["scope"], "default_policy": protocol["default_policy"]})
    table = []
    for row in models:
        result = row["test"]
        table.append({"model": row["model"], "threshold": row["selected_threshold"],
                      "event_recall": result["summary"]["event_recall"] if result else None,
                      "off_target_per_clip": result["summary"]["off_target_episodes_per_clip"] if result else None,
                      "prefix_macro": result["pre_event"]["macro_positive_fraction"] if result else None,
                      "prefix_supported_clips": result["pre_event"]["clips_with_support"] if result else None,
                      "median_delay_sec": result["summary"]["trigger_delay_sec"]["median"] if result else None,
                      "p95_delay_sec": result["summary"]["trigger_delay_sec"]["p95"] if result else None,
                      "episode_cap_exceeded": row["test_episode_cap_exceeded"], "prefix_cap_exceeded": row["test_prefix_cap_exceeded"]})
    csv_output(output.with_suffix(".csv"), table)
    print(json.dumps({"prediction_verification": {"checked": checked, "maximum_error": max_error}, "contrasts": contrasts}, indent=2))


def outcome_row(key, cohort, report, evaluation, threshold):
    event = asdict(evaluation) if evaluation is not None else {}
    return {"model": key, "cohort": cohort, "path": report["path"], "source_sha256": report["source_sha256"],
            "threshold": threshold, "detected": int(event["detected"]) if event else None,
            "trigger_pts_sec": event.get("trigger_time_sec"), "signed_delay_sec": event.get("trigger_delay_sec"),
            "early_episodes": int(event["early_alert_episodes"]) if event else None,
            "late_episodes": int(event["late_alert_episodes"]) if event else None,
            "valid_contexts": len(report["window_starts"]), "rejected_contexts": len(report["rejected_window_starts"])}


if __name__ == "__main__":
    main()
