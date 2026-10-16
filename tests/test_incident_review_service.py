import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from illegal_parking.incident_review_service import load_review_request, review_incident_requests, review_saved_events, QWEN_REVISION
from illegal_parking.incident_vlm_evidence import write_clean_review_package


@pytest.fixture
def saved_report(tmp_path):
    evidence = write_clean_review_package((np.zeros((20, 30, 3), dtype=np.uint8),) * 3, (0, 1, 2), tmp_path / "evidence", privacy_method="test blur")
    report = {"input": "camera", "output_video": "output.mp4", "events": [{"trigger_time_sec": 1, "probability": 0.9, "evidence": evidence}], "elapsed_sec": 3, "runtime": {}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def fake_review(root, requests):
    return [{"raw_text": "raw", "review": {"schema_valid": True, "likely_violation": True, "confidence": 0.9, "missing_evidence": ["Impact not visible"], "human_review_needed": True, "visual_reasons": ["Vehicle stopped"]}} for _ in requests]


def test_saved_review_preserves_detector_source_and_records_lineage(saved_report, tmp_path):
    before = saved_report.read_bytes()
    output = review_saved_events(tmp_path, saved_report, [0], tmp_path / "review", reviewer=fake_review)
    result = json.loads(output.read_text())
    assert saved_report.read_bytes() == before
    assert result["elapsed_sec"] == 3
    assert result["output_video"] == str((tmp_path / "output.mp4").resolve())
    assert result["events"][0]["probability"] == 0.9
    assert result["events"][0]["review_disposition"] == "incomplete_visual_evidence"
    assert result["evidence_review_runs"][0]["detector_rerun"] is False
    assert result["evidence_review_runs"][0]["source_report_sha256"] == hashlib.sha256(before).hexdigest()
    assert Path(result["events"][0]["vlm_raw_path"]).read_text() == "raw"


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [1], [0, 1, 2, 3]])
def test_saved_review_rejects_invalid_selection_before_inference(saved_report, tmp_path, indices):
    with pytest.raises(ValueError):
        review_saved_events(tmp_path, saved_report, indices, tmp_path / "review", reviewer=lambda *args: pytest.fail("Invalid event submitted"))


def test_saved_review_never_overwrites_source_or_completed_output(saved_report, tmp_path):
    with pytest.raises(FileExistsError):
        review_saved_events(tmp_path, saved_report, [0], tmp_path, reviewer=fake_review)
    review_saved_events(tmp_path, saved_report, [0], tmp_path / "review", reviewer=fake_review)
    with pytest.raises(FileExistsError):
        review_saved_events(tmp_path, saved_report, [0], tmp_path / "review", reviewer=fake_review)


def test_review_requires_privacy_and_local_images(saved_report):
    report = json.loads(saved_report.read_text())
    path = Path(report["events"][0]["evidence"]["request_json"])
    payload = json.loads(path.read_text())
    payload["evidence"]["privacy_redacted"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="privacy"):
        load_review_request(path)
    payload["evidence"]["privacy_redacted"] = True
    payload["image_paths"][0] = "https://example.org/image.jpg"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="absolute local"):
        load_review_request(path)


@pytest.mark.parametrize("raw,valid", [('not structured JSON', False), ('{"likely_violation":false,"confidence":0.8,"visual_reasons":["No interaction"],"missing_evidence":[],"human_review_needed":false}', True)])
def test_pinned_review_service_records_sources_and_parse_failures(monkeypatch, saved_report, tmp_path, raw, valid):
    torch = pytest.importorskip("torch")
    calls = []

    class Pipe:
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=QWEN_REVISION), parameters=lambda: iter([torch.zeros(1)]))

        def __call__(self, **kwargs):
            assert kwargs["generate_kwargs"]["do_sample"] is False
            return [{"generated_text": raw}]

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return Pipe()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = json.loads(saved_report.read_text())
    request = load_review_request(Path(report["events"][0]["evidence"]["request_json"]))
    result = review_incident_requests(tmp_path, [request], pipeline_factory=factory)[0]
    assert calls[0]["revision"] == QWEN_REVISION
    assert result["review"]["schema_valid"] is valid
    assert len(result["review"]["image_sha256"]) == 3
    assert result["raw_text"] == raw
