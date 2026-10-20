import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def worker():
    spec = importlib.util.spec_from_file_location("context_contract_worker", ROOT / "scripts/run_continuous_context_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_appendix_pins_workers_and_train_only_smoke_sources():
    protocol = json.loads((ROOT / "docs/results/continuous_training_protocol.json").read_text())
    execution = json.loads((ROOT / "docs/results/continuous_context_execution.json").read_text())
    assert execution["protocol_sha256"] == hashlib.sha256((ROOT / "docs/results/continuous_training_protocol.json").read_bytes()).hexdigest()
    assert len(execution["smoke_sources"]) == 2
    for definition in protocol["memberships"].values():
        assert set(execution["smoke_sources"]) <= set(definition["fit_clip_paths"])
        assert not set(execution["smoke_sources"]) & set(definition["calibration_clip_paths"] + definition["test_clip_paths"])
    for name, expected in execution["implementation_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_test_replay_cannot_begin_without_frozen_policies(tmp_path, monkeypatch):
    module = worker()
    monkeypatch.setattr(module, "SELECTION", tmp_path / "missing.json")
    monkeypatch.setattr(module, "DIRECTORY", tmp_path)
    monkeypatch.setattr(module, "load_models", lambda: {"fixture": {}})
    with pytest.raises(FileNotFoundError):
        module.replay({}, {}, "test", "0")
    assert not (tmp_path / "test").exists()


def test_artifact_binding_rejects_another_execution(tmp_path, monkeypatch):
    module = worker()
    monkeypatch.setattr(module, "signature", lambda: {"protocol_sha256": "p", "execution_sha256": "current"})
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"protocol_sha256": "p", "execution_sha256": "old"}))
    with pytest.raises(ValueError, match="different execution"):
        module.read_bound(path)
