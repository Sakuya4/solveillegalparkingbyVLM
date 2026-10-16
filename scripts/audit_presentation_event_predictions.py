from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from scripts.evaluate_presentation_events import model_reports
from scripts.run_presentation_event_reporting import ScalarJSON, validate_amendment
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
from illegal_parking.incident_presentation_events import evaluate_presentation_event, presentation_operating_summary
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def clock_control(report, positive):
    return {**report, "window_probabilities": [float(positive)] * len(report["window_starts"])}


def control_results(reports, annotations):
    return {name: presentation_operating_summary([clock_control(report, positive) for report in reports], annotations, .5)
            for name, positive in (("clock_only_positive", True), ("always_negative", False))}


def verify_feature_predictions(directory, batch, reports, protocol):
    predictors = {key: IncidentTcnPredictor.from_checkpoint(ROOT / definition["path"], "cpu") for key, definition in protocol["models"].items()}
    maximum, checked = 0., 0
    for report in reports:
        with np.load(directory / batch["clips"][report["path"]]["features"], allow_pickle=False) as data:
            features, names = data["features"], data["feature_names"].astype(str).tolist()
        if names != list(CANDIDATE_MOTION_FEATURE_NAMES) or features.shape != (len(report["window_starts"]), 15, len(names)) or not np.all(np.isfinite(features)):
            raise ValueError("Saved features violate the continuous input contract")
        for key, predictor in predictors.items():
            columns = [names.index(name) for name in predictor.feature_names]
            actual = predictor.predict_probabilities(features[:, :, columns]) if len(features) else np.empty(0)
            saved = np.asarray(report["scores"][key], dtype=float)
            if actual.shape != saved.shape or not np.all(np.isfinite(saved)):
                raise ValueError("Saved scores do not align with recomputed predictions")
            error = float(np.max(np.abs(actual - saved))) if len(actual) else 0.
            if error > 1e-7:
                raise ValueError("Published probabilities do not reproduce from the hashed features and checkpoints")
            maximum = max(maximum, error)
            checked += len(actual)
    return {"verified_probabilities": checked, "maximum_absolute_error": maximum, "tolerance": 1e-7, "device": "cpu"}


def main():
    parser = argparse.ArgumentParser(description="Audit frozen predictions against saved features and nonvisual event controls.")
    parser.add_argument("stage", choices=("freeze", "publish"))
    args = parser.parse_args()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    threadpool_limits(2)
    protocol_path = ROOT / "docs/results/presentation_event_protocol.json"
    protocol = load_event_protocol(protocol_path)
    validate_amendment()
    plan_path = ROOT / "docs/results/presentation_event_controls_protocol.json"
    calibration_directory = ROOT / "outputs/accident/presentation_events_calibration"
    calibration_batch, calibration_reports = validate_batch(calibration_directory, protocol_path, protocol, "calibration")
    selection_path = ROOT / "outputs/accident/presentation_event_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != sha256(protocol_path) or selection["calibration_batch_sha256"] != sha256(calibration_directory / "batch.json"):
        raise ValueError("Selection provenance changed before control analysis")
    if args.stage == "freeze":
        if plan_path.exists():
            raise FileExistsError("Control specifications are immutable")
        test_directory = ROOT / "outputs/accident/presentation_events_test"
        if test_directory.exists() and any(test_directory.iterdir()):
            raise ValueError("The control specification must precede test replay")
        record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256(protocol_path),
                  "selection_sha256": sha256(selection_path), "auditor_sha256": sha256(Path(__file__)),
                  "serialization_amendment_sha256": sha256(ROOT / "docs/results/presentation_event_serialization.json"),
                  "reason": "Diagnostic addition after calibration revealed zero-threshold selections, before test replay; not part of the original pre-fitting model comparison",
                  "controls": {"clock_only_positive": "Every valid context positive, gate2, same actual PTS and missing-hop resets; no image or probability inputs",
                               "always_negative": "Every valid context negative; same cohort and denominator"},
                  "fixed_control_threshold": .5, "paired_bootstrap_resamples": 2000, "seed": 42,
                  "comparison": "Each selected model minus clock-only positive event recall, paired by source clip, all eight models",
                  "prediction_verification": "Recompute all eight checkpoint outputs from every saved feature tensor; absolute tolerance 1e-7",
                  "scope": "Diagnostic controls expose temporal dataset structure, not normal-camera operational performance"}
        plan_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(record, indent=2))
        return
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan["protocol_sha256"] != sha256(protocol_path) or plan["selection_sha256"] != sha256(selection_path)
            or plan["auditor_sha256"] != sha256(Path(__file__))
            or plan["serialization_amendment_sha256"] != sha256(ROOT / "docs/results/presentation_event_serialization.json")):
        raise ValueError("Control specification inputs or implementation changed")
    test_directory = ROOT / "outputs/accident/presentation_events_test"
    test_batch, test_reports = validate_batch(test_directory, protocol_path, protocol, "test", sha256(selection_path))
    if datetime.fromisoformat(test_batch["started_at_utc"]) < datetime.fromisoformat(plan["created_at_utc"]):
        raise ValueError("Control specification must precede test replay")
    output = ROOT / "docs/results/presentation_event_controls.json"
    if output.exists():
        raise FileExistsError("Control results are immutable")
    verification = {"calibration": verify_feature_predictions(calibration_directory, calibration_batch, calibration_reports, protocol),
                    "test": verify_feature_predictions(test_directory, test_batch, test_reports, protocol)}
    controls = {}
    for split in ("iid", "geographic"):
        key = next(key for key, definition in protocol["models"].items() if definition["split"] == split)
        controls[split] = {cohort: control_results(model_reports(protocol, reports, key, cohort), protocol["annotations"])
                           for cohort, reports in (("calibration", calibration_reports), ("test", test_reports))}
    comparisons = []
    for key in protocol["models"]:
        reports = model_reports(protocol, test_reports, key, "test")
        selected = selection["policies"][key]["selected"]
        if selected is None:
            comparisons.append({"model": key, "selected_threshold": None, "paired_recall_vs_clock": None})
            continue
        threshold = selected["threshold"]
        model_outcomes = [evaluate_presentation_event(report, protocol["annotations"][report["path"]], threshold).detected for report in reports]
        control_outcomes = [evaluate_presentation_event(clock_control(report, True), protocol["annotations"][report["path"]], .5).detected for report in reports]
        positives = [value >= threshold for report in reports for value in report["window_probabilities"]]
        if threshold == 0 and (not all(positives) or model_outcomes != control_outcomes):
            raise ValueError("Zero-threshold model must reproduce the nonvisual positive control")
        comparisons.append({"model": key, "selected_threshold": threshold, "positive_window_fraction": sum(positives) / len(positives) if positives else None,
                            "paired_recall_vs_clock": paired_cluster_coverage_interval(np.ones(len(reports)), np.asarray(model_outcomes, dtype=float), np.asarray(control_outcomes, dtype=float))})
    result = {"protocol_sha256": sha256(protocol_path), "control_protocol_sha256": sha256(plan_path),
              "serialization_amendment_sha256": plan["serialization_amendment_sha256"],
              "calibration_batch_sha256": sha256(calibration_directory / "batch.json"), "test_batch_sha256": sha256(test_directory / "batch.json"),
              "prediction_verification": verification, "controls": controls, "model_comparisons": comparisons,
              "scope": plan["scope"], "uncertainty": "Eight pointwise paired clip-bootstrap diagnostic contrasts; fixed models and thresholds; no family-wise correction"}
    output.write_text(ScalarJSON.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(ScalarJSON.dumps({"prediction_verification": verification, "model_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
