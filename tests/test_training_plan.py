from pathlib import Path

from illegal_parking.training_plan import (
    TrainingEstimateInput,
    build_yolo_data_yaml,
    estimate_training_outcome,
    inspect_yolo_dataset,
)


def test_build_yolo_data_yaml_writes_ultralytics_layout(tmp_path):
    dataset_root = tmp_path / "fisheye8k"
    output_path = tmp_path / "fisheye8k.yaml"

    config = build_yolo_data_yaml(
        dataset_root=dataset_root,
        output_path=output_path,
        class_names=["Bus", "Bike", "Car"],
    )

    text = output_path.read_text(encoding="utf-8")
    assert f"path: {dataset_root.as_posix()}" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert "nc: 3" in text
    assert '0: "Bus"' in text
    assert config.class_names == ["Bus", "Bike", "Car"]


def test_inspect_yolo_dataset_counts_images_and_labels(tmp_path):
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True)
        (dataset_root / "labels" / split).mkdir(parents=True)
    (dataset_root / "images" / "train" / "a.jpg").write_bytes(b"fake")
    (dataset_root / "images" / "train" / "b.png").write_bytes(b"fake")
    (dataset_root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (dataset_root / "images" / "val" / "c.jpg").write_bytes(b"fake")

    summary = inspect_yolo_dataset(dataset_root)

    assert summary.splits["train"].image_count == 2
    assert summary.splits["train"].label_count == 1
    assert summary.splits["train"].missing_label_count == 1
    assert summary.splits["val"].image_count == 1
    assert summary.total_images == 3


def test_estimate_training_outcome_uses_dataset_scale_and_edge_profile():
    estimate = estimate_training_outcome(
        TrainingEstimateInput(
            dataset_id="fisheye8k",
            image_count=8000,
            box_count=157000,
            model_size="yolov8n",
            image_size=640,
            edge_profile="snapdragon_orin_class",
        )
    )

    assert estimate.dataset_id == "fisheye8k"
    assert estimate.expected_map50_range[0] >= 0.6
    assert estimate.expected_map50_range[1] <= 0.9
    assert estimate.expected_event_precision_range[0] < estimate.expected_event_precision_range[1]
    assert estimate.training_run_command.startswith("yolo detect train")
