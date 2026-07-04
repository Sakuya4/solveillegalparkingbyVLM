import json

from illegal_parking.evaluation import (
    FrameAnnotation,
    evaluate_frame_events,
    load_frame_annotations,
    load_predicted_event_frames,
)


def test_load_frame_annotations_accepts_list_schema(tmp_path):
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {"frame_index": 0, "expected_violation": False},
                {"frame_index": 3, "expected_violation": True},
            ]
        ),
        encoding="utf-8",
    )

    annotations = load_frame_annotations(annotations_path)

    assert annotations == [
        FrameAnnotation(frame_index=0, expected_violation=False),
        FrameAnnotation(frame_index=3, expected_violation=True),
    ]


def test_load_frame_annotations_accepts_wrapped_schema(tmp_path):
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "camera_id": "road_1",
                "frames": [
                    {"frame_index": 1, "expected_violation": True},
                    {"frame_index": 2, "expected_violation": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    annotations = load_frame_annotations(annotations_path)

    assert annotations == [
        FrameAnnotation(frame_index=1, expected_violation=True),
        FrameAnnotation(frame_index=2, expected_violation=False),
    ]


def test_load_predicted_event_frames_reads_candidate_events(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"frame_index": 2, "is_candidate": True}),
                json.dumps({"frame_index": 4, "is_candidate": False}),
                json.dumps({"frame_index": 5, "status": "CANDIDATE"}),
            ]
        ),
        encoding="utf-8",
    )

    assert load_predicted_event_frames(events_path) == {2, 5}


def test_evaluate_frame_events_compares_annotations_to_event_frames():
    result = evaluate_frame_events(
        annotations=[
            FrameAnnotation(frame_index=0, expected_violation=False),
            FrameAnnotation(frame_index=2, expected_violation=True),
            FrameAnnotation(frame_index=3, expected_violation=False),
        ],
        predicted_event_frames={2, 3},
    )

    assert result.metrics.to_dict() == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "tn": 1,
        "precision": 0.5,
        "recall": 1.0,
        "false_positive_rate": 0.5,
    }
    assert result.evaluated_frames == 3
    assert result.predicted_event_frames == [2, 3]
