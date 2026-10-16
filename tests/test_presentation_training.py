import json
from pathlib import Path

import numpy as np
import pytest

from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
from scripts.prepare_presentation_protocol import assign_partitions, sha256
from scripts.extract_presentation_features import load_manifest, clip_frame_support
from scripts.train_presentation_tcn import validate_frozen_sequences
from scripts.run_overnight_research import cadence_steps
from scripts.publish_presentation_comparison import paired_contrasts
from scripts import train_presentation_tcn


@pytest.fixture
def frozen(tmp_path):
    rows = [{"path": f"{index}.mp4", "label": "incident" if target else "normal", "target": target,
             "iid_split": "train" if index < 20 else "test", "geographic_split": "train" if index < 20 else "test",
             "legacy_indices": list(range(0, 32, 2)), "fixed_indices": list(range(0, 32, 2)),
             "legacy_times_sec": (np.arange(16) / 7.5).tolist(), "fixed_times_sec": (np.arange(16) / 7.5).tolist()}
            for index in range(25) for target in (0, 1)]
    for scheme in ("iid", "geographic"):
        assign_partitions(rows, scheme)
    manifest = tmp_path / "windows.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    protocol = {"status": "frozen_before_feature_extraction", "windows_file": manifest.name,
                "windows_sha256": sha256(manifest), "paired_windows": len(rows), "paired_clips": 25,
                "training": {"calibration_fraction": .2, "seed": 42, "epochs": 40, "channels": 32,
                             "dropout": .1, "batch_size": 64, "learning_rate": .001, "weight_decay": .0001, "target_train_fpr": .2},
                "clock": "test presentation clock",
                "implementation_sha256": {}, "reference_model_sha256": {}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    features = np.zeros((len(rows), 15, len(CANDIDATE_MOTION_FEATURE_NAMES)), dtype=np.float32)
    sequence = tmp_path / "sequences.npz"
    fields = ("path", "label", "target", "iid_split", "geographic_split", "iid_partition", "geographic_partition")
    np.savez_compressed(sequence, features=features, feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES),
                        protocol_sha256=sha256(path), arm="fixed", frame_indices=np.asarray([row["fixed_indices"] for row in rows]),
                        frame_times_sec=np.asarray([row["fixed_times_sec"] for row in rows]),
                        **{key: np.asarray([row[key] for row in rows]) for key in fields})
    report = {"protocol_sha256": sha256(path), "sequences_sha256": sha256(sequence), "arm": "fixed", "smoke_clips": None}
    (tmp_path / "extraction.json").write_text(json.dumps(report), encoding="utf-8")
    return path, tmp_path


def test_complete_features_reconstruct_frozen_clip_partitions(frozen):
    path, directory = frozen
    protocol, report, sequence = validate_frozen_sequences(path, directory, "iid")
    assert protocol["paired_windows"] == 50 and report["arm"] == "fixed"
    assert sequence.name == "sequences.npz"


def test_smoke_subset_cannot_be_promoted_to_the_training_cohort(frozen):
    path, directory = frozen
    report_path = directory / "extraction.json"
    report = json.loads(report_path.read_text())
    report["smoke_clips"] = 2
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete"):
        validate_frozen_sequences(path, directory, "iid")


@pytest.mark.parametrize("field", ["target", "frame_times_sec", "feature_names", "features"])
def test_hash_updated_but_contract_changed_sequences_are_rejected(frozen, field):
    path, directory = frozen
    sequence = directory / "sequences.npz"
    with np.load(sequence, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    if field == "target":
        payload[field][0] = 1
    elif field == "frame_times_sec":
        payload[field][0, 0] += .2
    elif field == "feature_names":
        payload[field][0] = "oracle_roi"
    else:
        payload[field][0, 0, 0] = np.nan
    np.savez_compressed(sequence, **payload)
    report_path = directory / "extraction.json"
    report = json.loads(report_path.read_text())
    report["sequences_sha256"] = sha256(sequence)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_frozen_sequences(path, directory, "iid")


def test_changed_manifest_is_rejected_before_data_loading(frozen):
    path, directory = frozen
    with (directory / "windows.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash-matched"):
        load_manifest(path)


def test_cadence_queue_extracts_both_arms_before_eight_matched_fits():
    steps = cadence_steps()
    assert len(steps) == 11 and len({name for name, _ in steps}) == 11
    assert all(arguments[0].endswith("extract_presentation_features.py") for _, arguments in steps[:2])
    assert all(arguments[0].endswith("train_presentation_tcn.py") for _, arguments in steps[2:10])
    assert len({(arguments[arguments.index("--features-dir") + 1], arguments[arguments.index("--feature") + 1], arguments[arguments.index("--split-scheme") + 1]) for _, arguments in steps[2:10]}) == 8
    assert steps[-1][1][0].endswith("publish_presentation_comparison.py")
    assert not any("candidate500" in token or "incident_tcn.onnx" in token for _, arguments in steps for token in arguments)


def test_publisher_preserves_both_signs_of_the_paired_f1_contrast():
    labels = np.array([0, 1, 0, 1])
    perfect = np.array([.1, .9, .1, .9])
    clips = np.array(["a", "a", "b", "b"])
    scores = {(arm, feature, split): (labels, perfect if (arm == "fixed") == (feature == "candidate") else 1 - perfect, .5, clips)
              for arm in ("legacy", "fixed") for feature in ("candidate", "global") for split in ("iid", "geographic")}
    protocol = {"evaluation": {"bootstrap_resamples": 20, "seed": 42}}
    contrasts = paired_contrasts(protocol, scores)
    assert [row["difference"] for row in contrasts] == [1, 1, -1, -1]
    assert all(row["clusters"] == 2 for row in contrasts)
    scores[("legacy", "global", "geographic")] = (1 - labels, perfect, .5, clips)
    with pytest.raises(ValueError, match="identical"):
        paired_contrasts(protocol, scores)


def test_fit_wrapper_forwards_frozen_settings_in_process_and_preserves_outputs(frozen, monkeypatch):
    import sys
    from scripts import train_incident_tcn

    path, directory = frozen
    monkeypatch.setattr(train_presentation_tcn, "ROOT", directory)
    calls = []

    def fit():
        calls.append(list(sys.argv))
        for flag in ("--model-output", "--report-output", "--predictions-output"):
            output = Path(sys.argv[sys.argv.index(flag) + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}" if flag == "--report-output" else "fixture")
        return 0

    monkeypatch.setattr(train_incident_tcn, "main", fit)
    argv = ["fit", "--protocol", str(path), "--features-dir", str(directory), "--feature", "candidate_", "--split-scheme", "iid", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    train_presentation_tcn.main()
    command, = calls
    assert command[command.index("--epochs") + 1] == "40"
    assert command[command.index("--target-train-fpr") + 1] == "0.2"
    assert command[command.index("--threads") + 1] == "2"
    assert sys.argv is argv
    with pytest.raises(FileExistsError):
        train_presentation_tcn.main()
    assert len(calls) == 1


def test_published_presentation_manifest_preserves_time_support_and_partitions():
    root = Path(__file__).resolve().parents[1]
    protocol, rows = load_manifest(root / "docs/results/presentation_protocol.json")
    assert len(protocol["input_media"]) == 2027
    assert len(rows) == protocol["paired_windows"]
    for row in rows:
        fixed = np.asarray(row["fixed_times_sec"])
        target = np.asarray(row["fixed_target_times_sec"])
        assert len(fixed) == len(target) == 16
        assert np.all(np.diff(row["fixed_indices"]) > 0)
        assert np.allclose(np.diff(target), 1 / 7.5)
        assert np.min(fixed - target) >= -1e-6
        assert np.max(fixed - target) <= 1 / 7.5 + 1e-6
        assert 0 <= row["supplied_event_frame"] < row["source_frame_count"]
        if row["target"]:
            assert fixed[0] <= row["event_time_sec"] <= fixed[-1]
            assert row["legacy_indices"][0] <= row["supplied_event_frame"] <= row["legacy_indices"][-1]
        else:
            assert fixed[-1] < row["event_time_sec"]
            assert row["legacy_times_sec"][-1] < row["event_time_sec"]
    for scheme in ("iid", "geographic"):
        expected = [row[scheme + "_partition"] for row in rows]
        assign_partitions(rows, scheme)
        assert expected == [row[scheme + "_partition"] for row in rows]
        tests = set(protocol["continuous_cohorts"][scheme]["test_clip_paths"])
        assert len(tests) <= 100
        assert tests <= {row["path"] for row in rows if row[scheme + "_partition"] == "test"}
    for name, digest in protocol["reference_model_sha256"].items():
        assert sha256(root / "Application/desktop/models" / name) == digest


def test_feature_decode_reuses_both_policy_support_without_changing_window_inputs():
    row = {"supplied_event_frame": 5, "event_time_sec": .5,
           "legacy_indices": [0, 2, 4], "legacy_times_sec": [0, .2, .4],
           "fixed_indices": [0, 3, 6], "fixed_times_sec": [0, .3, .6]}
    indices, timestamps = clip_frame_support([row])
    assert indices == [0, 2, 3, 4, 5, 6]
    assert timestamps == [0, .2, .3, .4, .5, .6]
    assert row["legacy_indices"] == [0, 2, 4] and row["fixed_indices"] == [0, 3, 6]
    with pytest.raises(ValueError, match="conflicting"):
        clip_frame_support([row, {**row, "event_time_sec": .7}])
