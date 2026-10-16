import json
from pathlib import Path

import pytest

from scripts.publish_desktop_demo import build_summary, encode, main, sha256


ASSETS = Path(__file__).resolve().parents[1] / "docs/assets"


def test_published_demo_preserves_reference_and_review_disagreement():
    summary = json.loads((ASSETS / "incident_demo.json").read_text(encoding="utf-8"))
    assert summary["source"]["input_sha256"] == sha256(ASSETS / "incident_input.mp4")
    capture = json.loads((ASSETS / "desktop_walkthrough.json").read_text(encoding="utf-8"))
    assert capture["source_report_sha256"] == summary["source_report_sha256"]
    for name in ("incident_demo.mp4", "desktop_walkthrough.mp4", "incident_demo_preview.jpg", "incident_demo_preview.gif"):
        assert summary["media"][name] == sha256(ASSETS / name)
    event, = summary["events"]
    assert event["trigger_time_sec"] == pytest.approx(10.0416666667)
    assert event["review_disposition"] == "temporal_visual_disagreement"
    assert event["vlm_review"]["likely_violation"] is False
    assert event["vlm_review"]["generation_contract_version"] == 2
    assert capture["frames"] == 320 and capture["duration_sec"] == 40
    assert capture["selected_tabs"] == {"model_comparison": 2, "government_coverage": 2, "edge_runtime": 1}


def test_demo_summary_does_not_publish_machine_paths_or_invent_review_agreement():
    published = json.loads((ASSETS / "incident_demo.json").read_text(encoding="utf-8"))
    report = {**published, "input_sha256": "input", "window_starts": [0],
              "input": "C:/private/source.mp4", "output_video": "C:/private/output.mp4"}
    summary = build_summary(report, "report", "video", "walkthrough")
    assert "C:/private" not in json.dumps(summary)
    assert summary["events"][0]["review_disposition"] == "temporal_visual_disagreement"
    assert summary["processing_scope"].startswith("Sampled-frame")


def test_h264_encoding_limits_threads_and_enables_browser_playback(monkeypatch):
    calls = []
    monkeypatch.setattr("scripts.publish_desktop_demo.subprocess.run", lambda command, **kwargs: calls.append((command, kwargs)))
    encode("ffmpeg", "in.mp4", "out.mp4")
    command, options = calls[0]
    assert command.count("-threads") == 2
    assert all(command[index + 1] == "2" for index, value in enumerate(command) if value == "-threads")
    assert "libx264" in command and "yuv420p" in command and "+faststart" in command
    assert options["check"] is True


def test_source_credit_is_baked_into_video(monkeypatch):
    calls = []
    monkeypatch.setattr("scripts.publish_desktop_demo.subprocess.run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    encode("ffmpeg", "in.mp4", "out.mp4", credit=True)
    command, = calls
    assert "CC BY-NC-SA 4.0" in command[command.index("-vf") + 1]
    assert command[command.index("-filter_threads") + 1] == "2"


@pytest.mark.parametrize("mismatch", ["input", "capture"])
def test_publication_rejects_mismatched_provenance_before_encoding(tmp_path, monkeypatch, mismatch):
    report = tmp_path / "report.json"
    reference = ASSETS / "incident_input.mp4"
    report.write_text(json.dumps({"input_sha256": "wrong" if mismatch == "input" else sha256(reference),
                                  "input": str(reference)}), encoding="utf-8")
    capture = tmp_path / "capture.mp4"
    capture.with_suffix(".json").write_text(json.dumps({"source_report_sha256": "wrong"}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["publish", "--report", str(report), "--walkthrough", str(capture), "--ffmpeg", "ffmpeg"])
    monkeypatch.setattr("scripts.publish_desktop_demo.encode", lambda *args, **kwargs: pytest.fail("Mismatched evidence must not be encoded"))
    with pytest.raises(ValueError, match="reference clip|same execution"):
        main()


@pytest.mark.parametrize("name", ["incident_demo.mp4", "desktop_walkthrough.mp4"])
def test_demo_video_decodes_and_contains_changing_frames(name):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    cap = cv2.VideoCapture(str(ASSETS / name))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count > 100
        success, first = cap.read()
        assert success and np.std(first) > 10
        cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
        success, middle = cap.read()
        assert success and np.mean(np.abs(first.astype(float) - middle.astype(float))) > 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        assert cap.read()[0]
    finally:
        cap.release()
