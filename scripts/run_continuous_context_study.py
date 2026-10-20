from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
import numpy as np
from scripts.prepare_presentation_protocol import probe_times
from illegal_parking.incident_context_study import fit_indices, validate_features, model_reports, operating_point, select_policy
from illegal_parking.incident_presentation import normalized_presentation_times, read_presentation_frames
from illegal_parking.incident_presentation_events import continuous_presentation_windows
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES, compute_candidate_motion_step, candidate_motion_sequence_matrix
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor

PROTOCOL = ROOT / "docs/results/continuous_training_protocol.json"
EXECUTION = ROOT / "docs/results/continuous_context_execution.json"
DIRECTORY = ROOT / "outputs/accident/continuous_context"
DATASET = ROOT / "data/raw/accident/full/extracted"
FFPROBE = Path("C:/ffmpeg/bin/ffprobe.exe")
SELECTION = ROOT / "docs/results/continuous_context_selection.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=scalar, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def versions():
    import cv2
    import torch
    import ultralytics
    return {"torch": torch.__version__, "opencv": cv2.__version__, "ultralytics": ultralytics.__version__}


def load_study():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    if execution["protocol_sha256"] != sha(PROTOCOL) or versions() != execution["versions"]:
        raise ValueError("Frozen execution protocol or runtime changed")
    for name, expected in {**protocol["implementation_sha256"], **execution["implementation_sha256"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    for name, expected in protocol["reference_model_sha256"].items():
        if sha(ROOT / "Application/desktop/models" / name) != expected:
            raise ValueError("Reference model changed")
    if sha(ROOT / "yolov8n.pt") != protocol["detector_sha256"] or sha(FFPROBE) != protocol["ffprobe_sha256"]:
        raise ValueError("Detector or timestamp probe changed")
    rows_path = PROTOCOL.with_name(protocol["fit_windows_file"])
    if sha(rows_path) != protocol["fit_windows_sha256"]:
        raise ValueError("Fitting membership changed")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    return protocol, execution, rows


def signature():
    return {"protocol_sha256": sha(PROTOCOL), "execution_sha256": sha(EXECUTION)}


def read_bound(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if any(record.get(key) != value for key, value in signature().items()):
        raise ValueError(f"Artifact belongs to a different execution: {path}")
    return record


def timestamps(protocol, name):
    source = next(row for row in protocol["input_media"] if row["path"] == name)
    video = DATASET / name
    if sha(video) != source["sha256"]:
        raise ValueError(f"Source media changed: {name}")
    raw = probe_times(video, FFPROBE)
    if len(raw) != source["frame_count"] or hashlib.sha256(json.dumps(raw).encode()).hexdigest() != source["presentation_times_sha256"]:
        raise ValueError(f"Presentation timestamps changed: {name}")
    times = normalized_presentation_times(raw)
    anchor = protocol["annotations"][name]
    if abs(times[anchor["supplied_event_frame"]] - anchor["event_time_sec"]) > 1e-4:
        raise ValueError("Annotation frame no longer maps to its supplied PTS")
    return raw, times


def verify_fit(protocol, rows, directory, *, smoke=False):
    path = directory / "decode_audit.json"
    if path.exists():
        audit = read_bound(path)
        if audit["smoke"] != smoke or audit["row_hashes"] != [row_sha(row) for row in rows] or audit["failures"]:
            raise ValueError("Existing decoder audit is incomplete or incompatible")
        for name in audit["sources"]:
            timestamps(protocol, name)
        return audit
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    records, failures = {}, []
    for offset, (name, source_rows) in enumerate(sorted(grouped.items()), 1):
        try:
            raw, times = timestamps(protocol, name)
            windows, _ = continuous_presentation_windows(raw)
            indexed = {row["start"]: row for row in windows}
            for row in source_rows:
                indices = row["indices"]
                if len(indices) != 16 or len(set(indices)) != 16 or any(a >= b for a, b in zip(indices, indices[1:])):
                    raise ValueError("Fit context has invalid frame identities")
                if indices[0] < 0 or indices[-1] >= len(times):
                    raise ValueError("Fit context lies outside decoded source support")
                if row["arm"] == "continuous":
                    expected = indexed.get(row["start"])
                    if expected is None or expected["indices"] != indices or abs(expected["end_time_sec"] - row["end_time_sec"]) > 1e-9:
                        raise ValueError("Continuous context differs from the frozen grid")
                elif not np.allclose(times[indices], row["times_sec"], atol=1e-9, rtol=0):
                    raise ValueError("Anchored context presentation support changed")
            support = sorted({protocol["annotations"][name]["supplied_event_frame"], *(i for row in source_rows for i in row["indices"])})
            read_presentation_frames(DATASET / name, support, times[support].tolist(), 320)
            records[name] = {"timestamps": raw, "decoded_support": support}
        except (ValueError, OSError) as exc:
            failures.append({"path": name, "error": str(exc).replace(str(DATASET), "<dataset-root>")})
        if offset % 50 == 0:
            print(f"decode: {offset}/{len(grouped)} sources; failures={len(failures)}", flush=True)
    audit = {**signature(), "completed_at_utc": now(), "smoke": smoke,
             "row_hashes": [row_sha(row) for row in rows], "sources": records, "failures": failures}
    dump(path, audit)
    if failures:
        raise ValueError(f"Expanded-context decoder verification failed for {len(failures)} sources; no extraction or fitting permitted")
    return audit


def extract_fit(protocol, execution, rows, directory, device, *, smoke=False):
    path = directory / "extraction.json"
    if path.exists():
        return load_fit(rows, directory, allow_smoke=smoke)[1]
    audit = read_bound(directory / "decode_audit.json")
    if audit["smoke"] != smoke or audit["failures"] or audit["row_hashes"] != [row_sha(row) for row in rows]:
        raise ValueError("Every frozen fit context must pass decoder verification before extraction")
    cache = directory / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    prior = ROOT / execution["anchored_cache"]["directory"]
    for name, expected in execution["anchored_cache"]["files"].items():
        if sha(prior / name) != expected:
            raise ValueError("Anchored feature-cache provenance changed")
    old_report = json.loads((prior / "extraction.json").read_text())
    if (old_report["smoke_clips"] is not None or old_report["versions"] != execution["versions"] or old_report["arm"] != "fixed"
            or old_report["protocol_sha256"] != protocol["parent_protocol_sha256"] or old_report["sequences_sha256"] != sha(prior / "sequences.npz")):
        raise ValueError("Anchored cache must be a complete matching fixed-time extraction")
    from illegal_parking.incident_ultralytics import track_vehicle_frames
    from ultralytics import YOLO
    detector = YOLO(str(ROOT / "yolov8n.pt"))
    started = time.perf_counter()
    started_at = now()
    reused, tensors = 0, [None] * len(rows)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["path"]].append((index, row))
    with np.load(prior / "sequences.npz", allow_pickle=False) as data:
        if data["feature_names"].astype(str).tolist() != list(CANDIDATE_MOTION_FEATURE_NAMES):
            raise ValueError("Anchored feature channel order changed")
        old_index = {(name, tuple(indices)): offset for offset, (name, indices) in enumerate(zip(data["path"].astype(str), data["frame_indices"]))}
        if len(old_index) != len(data["features"]):
            raise ValueError("Anchored feature identities are not unique")
        old_features = data["features"].copy()
    for source_index, (name, source_rows) in enumerate(sorted(grouped.items()), 1):
        raw = audit["sources"][name]["timestamps"]
        times = normalized_presentation_times(raw)
        missing = [(index, row) for index, row in source_rows if row["arm"] == "continuous" and not (cache / f"{index:05d}.npz").exists()]
        support = sorted({i for _, row in missing for i in row["indices"]})
        frames = dict(zip(support, read_presentation_frames(DATASET / name, support, times[support].tolist(), 320))) if support else {}
        for index, row in source_rows:
            file = cache / f"{index:05d}.npz"
            if file.exists():
                with np.load(file, allow_pickle=False) as saved:
                    if str(saved["row_sha256"]) != row_sha(row) or str(saved["execution_sha256"]) != sha(EXECUTION):
                        raise ValueError("Per-context cache identity changed")
                    matrix = saved["features"].copy()
            elif row["arm"] == "anchored":
                matrix = old_features[old_index[(name, tuple(row["indices"]))]].copy()
                reused += 1
            else:
                indices = row["indices"]
                images = [frames[i] for i in indices]
                tracks = track_vehicle_frames(detector, images, indices, device=device, image_size=320, confidence=.15)
                matrix = candidate_motion_sequence_matrix([compute_candidate_motion_step(images[i - 1], images[i], tracks[indices[i]]) for i in range(1, 16)])
            matrix = validate_features(np.asarray([matrix], dtype=np.float32), 1)[0]
            if not file.exists():
                np.savez_compressed(file, features=matrix, row_sha256=row_sha(row), execution_sha256=sha(EXECUTION))
            tensors[index] = matrix
        if source_index % 25 == 0:
            print(f"features: {source_index}/{len(grouped)} sources", flush=True)
    sequence = directory / "fit_sequences.npz"
    np.savez_compressed(sequence, features=validate_features(np.stack(tensors), len(rows)),
                        row_hashes=np.asarray([row_sha(row) for row in rows]), feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES))
    record = {**signature(), "started_at_utc": started_at, "completed_at_utc": now(), "elapsed_sec": time.perf_counter() - started,
              "sequences_sha256": sha(sequence), "decode_audit_sha256": sha(directory / "decode_audit.json"),
              "rows": len(rows), "smoke": smoke, "anchored_cache_rows_loaded": reused, "device": device, "versions": versions()}
    dump(path, record)
    return record


def load_fit(rows, directory=DIRECTORY, *, allow_smoke=False):
    record = read_bound(directory / "extraction.json")
    if record["smoke"] and not allow_smoke:
        raise ValueError("Smoke extraction cannot enter complete-cohort fitting")
    path = directory / "fit_sequences.npz"
    if record["sequences_sha256"] != sha(path) or record["rows"] != len(rows):
        raise ValueError("Complete fitting features changed")
    with np.load(path, allow_pickle=False) as data:
        if data["row_hashes"].tolist() != [row_sha(row) for row in rows] or data["feature_names"].tolist() != list(CANDIDATE_MOTION_FEATURE_NAMES):
            raise ValueError("Fitting feature identities changed")
        matrix = validate_features(data["features"].copy(), len(rows))
    return matrix, record


def fit_one(features, targets, settings, device):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from illegal_parking.incident_tcn import TemporalConvClassifier, fit_sequence_standardizer
    if set(np.unique(targets)) != {0, 1} or len(features) != len(targets):
        raise ValueError("Fitting requires aligned features and both classes")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    standardizer = fit_sequence_standardizer(features)
    model = TemporalConvClassifier(features.shape[-1], settings["channels"], settings["dropout"]).to(device)
    ratio = float(np.count_nonzero(targets == 0) / np.count_nonzero(targets == 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    loader = DataLoader(TensorDataset(torch.from_numpy(standardizer.transform(features)), torch.from_numpy(targets.astype(np.float32))),
                        batch_size=settings["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(settings["seed"]), num_workers=0)
    losses = []
    for _ in range(settings["epochs"]):
        model.train()
        total = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
        losses.append(total / len(targets))
    return model, standardizer, losses


def fit_models(protocol, rows):
    import torch
    matrix, extraction = load_fit(rows)
    registry_path = DIRECTORY / "models.json"
    if registry_path.exists():
        return load_models()
    models = {}
    for arm in protocol["arms"]:
        for feature in protocol["features"]:
            columns = [i for i, name in enumerate(CANDIDATE_MOTION_FEATURE_NAMES) if name.startswith(feature)]
            for split in protocol["splits"]:
                key = f"{arm}_{feature.rstrip('_')}_{split}"
                path = ROOT / "models/continuous_context" / (key + ".pt")
                meta = DIRECTORY / (key + "_fit.json")
                indices = fit_indices(rows, protocol, arm, split)
                if meta.exists():
                    record = read_bound(meta)
                    if record["model_sha256"] != sha(path) or record["sequences_sha256"] != extraction["sequences_sha256"]:
                        raise ValueError("Existing fit checkpoint provenance changed")
                else:
                    if path.exists():
                        raise FileExistsError("Unbound model checkpoint; preserve and inspect before recovery")
                    started, start_time = now(), time.perf_counter()
                    targets = np.asarray([rows[i]["target"] for i in indices])
                    model, scaler, losses = fit_one(matrix[indices][:, :, columns], targets, protocol["training"], "cuda")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                                "input_features": len(columns), "channels": protocol["training"]["channels"],
                                "dropout": protocol["training"]["dropout"], "standardizer": scaler.to_dict(),
                                "feature_names": [CANDIDATE_MOTION_FEATURE_NAMES[i] for i in columns], "threshold": .5,
                                "threshold_source": "unselected_placeholder", "deployment_eligible": False}, path)
                    record = {**signature(), "model_sha256": sha(path), "sequences_sha256": extraction["sequences_sha256"],
                              "started_at_utc": started, "completed_at_utc": now(), "elapsed_sec": time.perf_counter() - start_time,
                              "arm": arm, "feature": feature, "split": split, "fit_contexts": len(indices),
                              "positive_contexts": int(targets.sum()), "negative_contexts": int(len(targets) - targets.sum()),
                              "fit_sources": len({rows[i]["path"] for i in indices}), "epochs": len(losses), "losses": losses,
                              "device": torch.cuda.get_device_name(0), "test_scoring_performed": False}
                    dump(meta, record)
                    del model
                models[key] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "fit_report": meta.name, "fit_report_sha256": sha(meta)}
                print(f"fitted {key}: {record['fit_contexts']} contexts, final loss={record['losses'][-1]:.6f}", flush=True)
    dump(registry_path, {**signature(), "completed_at_utc": now(), "models": models})
    return models


def load_models():
    registry = read_bound(DIRECTORY / "models.json")
    if len(registry["models"]) != 8:
        raise ValueError("All eight fits must finish before calibration")
    for value in registry["models"].values():
        if sha(ROOT / value["path"]) != value["sha256"] or sha(DIRECTORY / value["fit_report"]) != value["fit_report_sha256"]:
            raise ValueError("Frozen fitted models changed")
    return registry["models"]


def load_batch(cohort):
    directory = DIRECTORY / cohort
    batch = read_bound(directory / "batch.json")
    if batch["models_sha256"] != sha(DIRECTORY / "models.json"):
        raise ValueError("Replay model registry changed")
    protocol = json.loads(PROTOCOL.read_text())
    expected = {name for definition in protocol["memberships"].values() for name in definition[cohort + "_clip_paths"]}
    if batch["cohort"] != cohort or set(batch["clips"]) != expected:
        raise ValueError("Replay must retain every frozen source without replacement")
    selected = sha(SELECTION) if cohort == "test" else None
    if batch["selection_sha256"] != selected:
        raise ValueError("Replay no longer matches its frozen operating policies")
    reports = []
    for record in batch["clips"].values():
        if sha(directory / record["report"]) != record["report_sha256"] or sha(directory / record["features"]) != record["features_sha256"]:
            raise ValueError("Replay records changed")
        report = read_bound(directory / record["report"])
        if report["selection_sha256"] != selected or report["models_sha256"] != batch["models_sha256"] or report["features_sha256"] != record["features_sha256"]:
            raise ValueError("Per-source replay provenance changed")
        reports.append(report)
    if {report["path"] for report in reports} != expected or len(reports) != len(expected):
        raise ValueError("Replay contains duplicate or substituted source identities")
    return batch, reports


def replay(protocol, execution, cohort, device):
    from ultralytics import YOLO
    from scripts.replay_presentation_events import clip_features
    models = load_models()
    selected_sha = sha(SELECTION) if cohort == "test" else None
    if cohort == "test":
        selection = read_bound(SELECTION)
        if selection["models_sha256"] != sha(DIRECTORY / "models.json") or selection["calibration_batch_sha256"] != sha(DIRECTORY / "calibration/batch.json"):
            raise ValueError("Test must follow all frozen TRAIN-only selections")
    directory = DIRECTORY / cohort
    if (directory / "batch.json").exists():
        return load_batch(cohort)
    directory.mkdir(parents=True, exist_ok=True)
    predictors = {key: IncidentTcnPredictor.from_checkpoint(ROOT / value["path"], "cpu") for key, value in models.items()}
    names = sorted({name for definition in protocol["memberships"].values() for name in definition[cohort + "_clip_paths"]})
    detector = YOLO(str(ROOT / "yolov8n.pt"))
    old_directory = ROOT / execution["calibration_cache"]["directory"]
    old = json.loads((old_directory / "batch.json").read_text()) if cohort == "calibration" else None
    if old and sha(old_directory / "batch.json") != execution["calibration_cache"]["batch_sha256"]:
        raise ValueError("Prior calibration feature registry changed")
    started, records = now(), {}
    for offset, name in enumerate(names, 1):
        stem = hashlib.sha256(name.encode()).hexdigest()
        report_path, feature_path = directory / (stem + ".json"), directory / (stem + ".npz")
        raw, times = timestamps(protocol, name)
        windows, rejected = continuous_presentation_windows(raw)
        if report_path.exists():
            report = read_bound(report_path)
            if report["features_sha256"] != sha(feature_path) or report["selection_sha256"] != selected_sha or report["models_sha256"] != sha(DIRECTORY / "models.json"):
                raise ValueError("Cached replay belongs to different features, models or policies")
            if report["window_frame_indices"] != [row["indices"] for row in windows] or report["window_end_times_sec"] != [row["end_time_sec"] for row in windows] or report["rejected_window_starts"] != rejected:
                raise ValueError("Cached replay context identities changed")
        else:
            if old:
                previous = old["clips"][name]
                old_report = old_directory / previous["report"]
                old_features = old_directory / previous["features"]
                if sha(old_report) != previous["report_sha256"] or sha(old_features) != previous["features_sha256"] or old["versions"] != versions():
                    raise ValueError("Calibration feature cache changed")
                prior = json.loads(old_report.read_text())
                if prior["window_frame_indices"] != [row["indices"] for row in windows] or prior["rejected_window_starts"] != rejected:
                    raise ValueError("Cached calibration inputs differ from continuous PTS contexts")
                with np.load(old_features, allow_pickle=False) as data:
                    if data["feature_names"].tolist() != list(CANDIDATE_MOTION_FEATURE_NAMES):
                        raise ValueError("Calibration feature channels changed")
                    matrix = validate_features(data["features"].copy(), len(windows))
            else:
                matrix, actual, actual_rejected = clip_features(DATASET / name, raw, detector, device)
                if actual != windows or actual_rejected != rejected:
                    raise ValueError("Extracted test contexts differ from the frozen physical-time grid")
            np.savez_compressed(feature_path, features=matrix, feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES))
            scores = {}
            for key, predictor in predictors.items():
                columns = [CANDIDATE_MOTION_FEATURE_NAMES.index(name) for name in predictor.feature_names]
                scores[key] = predictor.predict_probabilities(matrix[:, :, columns]).tolist() if len(matrix) else []
            report = {**signature(), "path": name, "source_sha256": sha(DATASET / name), "features_sha256": sha(feature_path),
                      "models_sha256": sha(DIRECTORY / "models.json"), "selection_sha256": selected_sha,
                      "clock": "normalized_presentation_time", "annotation_free_inference": True,
                      "window_starts": [row["start"] for row in windows], "window_frame_indices": [row["indices"] for row in windows],
                      "window_end_times_sec": [row["end_time_sec"] for row in windows], "rejected_window_starts": rejected, "scores": scores}
            dump(report_path, report)
        records[name] = {"report": report_path.name, "features": feature_path.name, "report_sha256": sha(report_path), "features_sha256": sha(feature_path)}
        print(f"{cohort}: {offset}/{len(names)} sources", flush=True)
    dump(directory / "batch.json", {**signature(), "cohort": cohort, "models_sha256": sha(DIRECTORY / "models.json"),
                                   "selection_sha256": selected_sha, "started_at_utc": started, "completed_at_utc": now(), "clips": records})
    return load_batch(cohort)


def calibrate(protocol, execution, device):
    _, reports = replay(protocol, execution, "calibration", device)
    policies = {}
    for key in load_models():
        chosen = model_reports(protocol, reports, key, "calibration")
        curve = [operating_point(chosen, protocol["annotations"], threshold) for threshold in protocol["calibration"]["threshold_grid"]]
        policies[key] = {"selected": select_policy(curve, protocol["calibration"]), "curve": curve}
    if SELECTION.exists():
        if read_bound(SELECTION)["policies"] != policies:
            raise ValueError("Frozen calibration policies do not reproduce")
        return
    dump(SELECTION, {**signature(), "selected_at_utc": now(), "models_sha256": sha(DIRECTORY / "models.json"),
                     "calibration_batch_sha256": sha(DIRECTORY / "calibration/batch.json"), "policies": policies,
                     "input_scope": "Corresponding official-TRAIN holdouts only; no new test reports read"})
    print(json.dumps({key: row["selected"]["threshold"] if row["selected"] else None for key, row in policies.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Execute the frozen continuous-context comparison without changing prior studies.")
    parser.add_argument("stage", choices=("smoke", "verify-fit", "extract-fit", "fit", "calibrate", "test"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    import cv2
    import torch
    from threadpoolctl import threadpool_limits
    from illegal_parking.gpu_job import exclusive_gpu_job
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    threadpool_limits(2)
    protocol, execution, rows = load_study()
    if args.stage != "smoke" and args.device != execution["detector_device"]:
        raise ValueError("Production execution must use its frozen detector device")
    state_path = ROOT / "outputs/research_runs/continuous_context_status.json"
    state = {"pid": os.getpid(), "child_pid": None, "stage": args.stage, "status": "running", "started_at_utc": now(), **signature()}
    with exclusive_gpu_job(ROOT):
        dump(state_path, state)
        try:
            if args.stage == "smoke":
                names = set(execution["smoke_sources"])
                selected = [row for row in rows if row["path"] in names]
                directory = DIRECTORY / "smoke"
                verify_fit(protocol, selected, directory, smoke=True)
                extract_fit(protocol, execution, selected, directory, args.device, smoke=True)
            elif args.stage == "verify-fit":
                verify_fit(protocol, rows, DIRECTORY)
            elif args.stage == "extract-fit":
                extract_fit(protocol, execution, rows, DIRECTORY, args.device)
            elif args.stage == "fit":
                fit_models(protocol, rows)
            elif args.stage == "calibrate":
                calibrate(protocol, execution, args.device)
            else:
                replay(protocol, execution, "test", args.device)
            state.update(status="complete", completed_at_utc=now())
        except BaseException as exc:
            state.update(status="failed", error=str(exc), completed_at_utc=now())
            raise
        finally:
            dump(state_path, state)


if __name__ == "__main__":
    main()
