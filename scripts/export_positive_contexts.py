from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_presentation_protocol import sha256
from scripts.replay_positive_exposure import load_study, load_batch
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from scripts.evaluate_presentation_events import model_reports


def export_rows(protocol, reports, cohort):
    inventory, contexts = [], []
    for model, definition in protocol["models"].items():
        for report in model_reports(protocol, reports, model, cohort):
            annotation = protocol["annotations"][report["path"]]
            identity = {"cohort": cohort, "split": definition["split"], "model": model,
                        "path": report["path"], "source_sha256": report["source_sha256"]}
            inventory.append({**identity, "annotation_pts_sec": annotation["event_time_sec"],
                              "collision_type": annotation["collision_type"],
                              "valid_contexts": len(report["window_starts"]),
                              "rejected_contexts": len(report["rejected_window_starts"])})
            if not len(report["window_starts"]) == len(report["window_end_times_sec"]) == len(report["window_probabilities"]):
                raise ValueError("Public contexts must preserve every aligned score")
            for start, end, score in zip(report["window_starts"], report["window_end_times_sec"], report["window_probabilities"]):
                contexts.append({**identity, "start_slot": start, "end_pts_sec": end, "probability": score})
    return inventory, contexts


def main():
    protocol_path, protocol, selection_path, _ = load_study()
    directory, batch, reports = load_batch(protocol_path, protocol, selection_path)
    results_path = ROOT / "docs/results/positive_exposure_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results["test_batch_sha256"] != sha256(directory / "batch.json"):
        raise ValueError("Only the completed published batch can be exported")
    original_path = ROOT / "docs/results/presentation_event_protocol.json"
    original = load_event_protocol(original_path)
    calibration_directory = ROOT / "outputs/accident/presentation_events_calibration"
    _, calibration = validate_batch(calibration_directory, original_path, original, "calibration")
    if sha256(calibration_directory / "batch.json") != protocol["calibration_batch_sha256"]:
        raise ValueError("Calibration source batch changed")
    inventory, contexts = export_rows(protocol, calibration, "calibration")
    test_inventory, test_contexts = export_rows(protocol, reports, "test")
    inventory.extend(test_inventory)
    contexts.extend(test_contexts)
    files = {"inventory": ROOT / "docs/results/positive_context_inventory.csv",
             "contexts": ROOT / "docs/results/positive_context_scores.csv"}
    manifest_path = ROOT / "docs/results/positive_context_trace.json"
    if manifest_path.exists() or any(path.exists() for path in files.values()):
        raise FileExistsError("Published public score traces are immutable")
    for key, rows in (("inventory", inventory), ("contexts", contexts)):
        with files[key].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {"protocol_sha256": sha256(protocol_path), "selection_sha256": sha256(selection_path),
                "results_sha256": sha256(results_path), "test_batch_sha256": results["test_batch_sha256"],
                "calibration_batch_sha256": protocol["calibration_batch_sha256"],
                "exporter_sha256": sha256(Path(__file__)),
                "files": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for key, path in files.items()},
                "inventory_rows": len(inventory), "context_rows": len(contexts),
                "model_sha256": {key: value["sha256"] for key, value in protocol["models"].items()},
                "scope": "All model-assigned calibration/test clips, including empty and unsupported prefixes. Context rows are valid PTS decisions, not camera-hours. Constant controls are reconstructed on the same grid."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
