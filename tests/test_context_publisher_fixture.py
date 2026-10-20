import importlib.util
import json
from pathlib import Path

import numpy as np


def test_complete_publisher_fixture_checks_all_eight_models_and_empty_sources(tmp_path, monkeypatch):
    import torch
    from illegal_parking.incident_tcn import TemporalConvClassifier
    from illegal_parking.incident_tcn_inference import IncidentTcnPredictor
    from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES as names
    from illegal_parking.incident_context_study import operating_point, select_policy
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("context_publisher_fixture", root / "scripts/publish_continuous_context_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "experiment"
    directory.mkdir()
    (tmp_path / "docs/results").mkdir(parents=True)
    monkeypatch.setattr(module, "DIRECTORY", directory)
    selection_path = tmp_path / "selection.json"
    monkeypatch.setattr(module, "SELECTION", selection_path)
    monkeypatch.setattr(module, "signature", lambda: {"protocol_sha256": "fixture", "execution_sha256": "fixture"})
    monkeypatch.setattr(module, "read_bound", lambda path: json.loads(path.read_text()))
    protocol = {"memberships": {split: {"calibration_clip_paths": ["a", "b"], "test_clip_paths": ["a", "empty"]} for split in ("iid", "geographic")},
                "annotations": {name: {"path": name, "event_time_sec": 5, "collision_type": "fixture"} for name in ("a", "b", "empty")},
                "calibration": {"threshold_grid": [0, .5, 1], "episode_budget_per_clip": .5,
                                "macro_pre_event_positive_cap": 1, "minimum_supported_clips": 1},
                "splits": ["iid", "geographic"], "scope": "Synthetic test fixture", "default_policy": "No defaults changed"}
    monkeypatch.setattr(module, "load_study", lambda: (protocol, {}, []))
    definitions, predictors = {}, {}
    for arm in ("anchored", "continuous"):
        for prefix in ("candidate_", "global_"):
            columns = [index for index, name in enumerate(names) if name.startswith(prefix)]
            for split in protocol["splits"]:
                key = f"{arm}_{prefix.rstrip('_')}_{split}"
                model = TemporalConvClassifier(len(columns), channels=4)
                path = tmp_path / (key + ".pt")
                torch.save({"state_dict": model.state_dict(), "input_features": len(columns), "channels": 4, "dropout": .1,
                            "feature_names": [names[index] for index in columns], "threshold": .5,
                            "standardizer": {"feature_mean": [0] * len(columns), "feature_scale": [1] * len(columns)}}, path)
                meta = directory / (key + ".json")
                meta.write_text("{}")
                definitions[key] = {"path": path.name, "sha256": module.sha(path), "fit_report": meta.name}
                predictors[key] = IncidentTcnPredictor.from_checkpoint(path)
    monkeypatch.setattr(module, "load_models", lambda: definitions)
    module.dump(directory / "models.json", definitions)
    cohorts = {}
    for cohort, paths in (("calibration", ["a", "b"]), ("test", ["a", "empty"])):
        folder = directory / cohort
        folder.mkdir()
        reports, records = [], {}
        for name in paths:
            matrix = np.zeros((0 if name == "empty" else 4, 15, len(names)), dtype=np.float32)
            feature_path = folder / (name + ".npz")
            np.savez_compressed(feature_path, features=matrix, feature_names=np.asarray(names))
            scores = {}
            for key, predictor in predictors.items():
                columns = [names.index(value) for value in predictor.feature_names]
                scores[key] = predictor.predict_probabilities(matrix[:, :, columns]).tolist() if len(matrix) else []
            reports.append({"path": name, "source_sha256": "fixture", "annotation_free_inference": True,
                            "clock": "normalized_presentation_time", "window_starts": [0, 15, 30, 45] if len(matrix) else [],
                            "window_end_times_sec": [2, 4, 6, 8] if len(matrix) else [], "scores": scores, "rejected_window_starts": []})
            records[name] = {"features": feature_path.name}
        batch = {"started_at_utc": "2026-10-20T02:00:00+00:00", "clips": records}
        module.dump(folder / "batch.json", batch)
        cohorts[cohort] = (batch, reports)
    monkeypatch.setattr(module, "load_batch", lambda cohort: cohorts[cohort])
    policies = {}
    for key in definitions:
        reports = [{**report, "window_probabilities": report["scores"][key]} for report in cohorts["calibration"][1]]
        curve = [operating_point(reports, protocol["annotations"], threshold) for threshold in protocol["calibration"]["threshold_grid"]]
        policies[key] = {"curve": curve, "selected": select_policy(curve, protocol["calibration"])}
    module.dump(selection_path, {"selected_at_utc": "2026-10-20T01:00:00+00:00", "policies": policies,
                                 "calibration_batch_sha256": module.sha(directory / "calibration/batch.json")})
    module.main()
    result = json.loads((tmp_path / "docs/results/continuous_context_results.json").read_text())
    assert len(result["models"]) == 8 and len(result["contrasts"]) == 4
    assert result["prediction_verification"]["maximum_absolute_error"] == 0
    assert result["prediction_verification"]["checked_probabilities"] == 96
    assert all(row["test"]["summary"]["clip_count"] == 2 for row in result["models"])
    assert result["files"]["outcomes"]["rows"] == 24
