from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_rows(rows, memberships):
    selected = []
    for row in rows:
        if row["cohort"] != "calibration":
            continue
        if row["model"] not in memberships or row["path"] not in memberships[row["model"]]:
            raise ValueError("Unexpected calibration identity")
        selected.append(row)
    return selected


def main():
    base = ROOT / "docs/results"
    output = base / "event_discrimination_protocol.json"
    targets = {"inventory": base / "event_discrimination_calibration_inventory.csv",
               "contexts": base / "event_discrimination_calibration_scores.csv"}
    if output.exists() or any(path.exists() for path in targets.values()):
        raise FileExistsError("The diagnostic input and endpoint freeze is immutable")
    parent_path = base / "positive_exposure_protocol.json"
    selection_path = base / "positive_exposure_selection.json"
    trace_path = base / "positive_context_trace.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if digest(parent_path) != trace["protocol_sha256"] or digest(selection_path) != trace["selection_sha256"]:
        raise ValueError("Parent model, cohort or selection provenance changed")
    memberships = {key: parent["cohorts"][definition["split"]]["calibration_clip_paths"] for key, definition in parent["models"].items()}
    sources = {row["path"]: row["sha256"] for row in parent["input_media"]}
    extracted = {}
    for key, target in targets.items():
        source = ROOT / trace["files"][key]["path"]
        if digest(source) != trace["files"][key]["sha256"]:
            raise ValueError("Frozen public trace bytes changed")
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = calibration_rows(reader, memberships)
        if any(row["source_sha256"] != sources[row["path"]] for row in rows):
            raise ValueError("Calibration source identity changed")
        if key == "inventory" and {(row["model"], row["path"]) for row in rows} != {(key, name) for key, names in memberships.items() for name in names}:
            raise ValueError("Every corresponding training-holdout clip must be retained")
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        extracted[key] = {"path": target.relative_to(ROOT).as_posix(), "sha256": digest(target), "rows": len(rows)}
    protocol = {
        "status": "frozen_before_event_local_metric_computation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol_sha256": digest(parent_path), "selection_sha256": digest(selection_path),
        "public_trace_sha256": digest(trace_path), "preparer_sha256": digest(Path(__file__)),
        "inputs": extracted, "models": parent["models"], "model_clip_paths": memberships,
        "thresholds": {key: {"onset_only": value["onset_only_threshold"], "positive_context": value["selected"]["threshold"]} for key, value in selection["policies"].items()},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
        "labels": {"prefix": "end_pts < supplied_event_pts - 1", "target": "supplied_event_pts - 1 <= end_pts <= supplied_event_pts + 3",
                   "post": "end_pts > supplied_event_pts + 3; counted but excluded from ranking", "meaning": "Annotation-relative context endpoints, not independently annotated frame-level visual states"},
        "ranking": {"pooled": ["AUROC", "noninterpolated average precision", "target prevalence", "AP minus prevalence"],
                    "within_clip": ["AUROC", "average precision", "target prevalence", "AP minus prevalence", "max target score minus max prefix score"],
                    "support": "Both prefix and target contexts required within a clip; disclose all unsupported categories without zero imputation",
                    "aggregation": "Context-weighted pooled metrics plus equal-supported-clip means; pooled and within-clip values are distinct estimands",
                    "ties": "Default scikit-learn roc_auc_score and average_precision_score; larger score means target, never reverse orientation",
                    "constant_control": "Score 0.5 on identical prefix/target contexts; report AUROC 0.5 and AP equal prevalence when both labels occur"},
        "timing": {"hop_slots": 15, "gate": 2, "target_fps": 7.5,
                   "gate1_attainable": "At least one valid decision endpoint in the inclusive target band",
                   "gate2_attainable": "At least one endpoint in the band with an immediately preceding valid hop-15 context, independent of scores",
                   "oracle_scope": "Annotation-aided structural upper bound ignoring score discrimination and operating caps; not a deployable detector or evidence of event recognition"},
        "miss_decomposition": {"policies": ["onset_only", "positive_context"],
                               "categories_in_priority_order": ["detected", "structurally_unreachable", "no_target_positive_pair", "onset_suppression"],
                               "detected": "A run==2 episode falls in the target band; missing hops reset runs",
                               "no_target_positive_pair": "Structurally reachable but no adjacent score>=threshold pair ends in the band",
                               "onset_suppression": "A qualifying positive pair exists in the band, but sustained positivity emitted its only episode earlier; no target-band episode"},
        "contrasts": {"pairing": "Fixed-time minus source-frame, matched model feature and split, identical supported clip identities",
                      "endpoints": ["within_clip_auroc", "within_clip_average_precision"],
                      "resamples": 2000, "seed": 42, "interval": "Pointwise source-clip percentile bootstrap; conditional on existing fitted weights, no family-wise or test-generalization claim"},
        "score_summary": "Count, mean, median, minimum, maximum for prefix/target/post contexts; no score-dependent clip selection",
        "scope": "Exploratory diagnostic using only corresponding official-TRAIN calibration rows, already used for prior threshold selection. Motivated after observing previous studies, not an untouched confirmatory sample. No fitting, inversion, threshold tuning, model selection or default changes; no new test endpoints computed.",
        "references": [{"doi": "10.1371/journal.pone.0118432", "authors": ["Takaya Saito", "Marc Rehmsmeier"], "year": 2015,
                        "journal": "PLOS ONE", "volume": 10, "issue": 3, "article": "e0118432", "verified": "Publisher metadata, 16 October 2026"},
                       {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html", "role": "Exact noninterpolated AP convention"}]
    }
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"protocol_sha256": digest(output), "inputs": extracted}, indent=2))


if __name__ == "__main__":
    main()
