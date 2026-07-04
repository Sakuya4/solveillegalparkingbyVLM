import json

from illegal_parking.fisheye8k import convert_fiftyone_samples_to_yolo, yolo_bbox_from_fiftyone


def test_yolo_bbox_from_fiftyone_converts_top_left_to_center():
    assert yolo_bbox_from_fiftyone([0.1, 0.2, 0.4, 0.6]) == (0.3, 0.5, 0.4, 0.6)


def test_convert_fiftyone_samples_to_yolo_writes_images_labels_and_yaml(tmp_path):
    metadata_dir = tmp_path / "metadata"
    image_dir = tmp_path / "source"
    output_dir = tmp_path / "yolo"
    metadata_dir.mkdir()
    (image_dir / "data").mkdir(parents=True)
    (image_dir / "data" / "frame1.png").write_bytes(b"fake image")
    samples = {
        "samples": [
            {
                "filepath": "data/frame1.png",
                "tags": ["train"],
                "metadata": {"width": 100, "height": 100},
                "detections": {
                    "detections": [
                        {"label": "Car", "bounding_box": [0.1, 0.2, 0.4, 0.6]},
                        {"label": "Bike", "bounding_box": [0.0, 0.0, 0.2, 0.2]},
                    ]
                },
            }
        ]
    }
    (metadata_dir / "samples.json").write_text(json.dumps(samples), encoding="utf-8")

    summary = convert_fiftyone_samples_to_yolo(
        metadata_path=metadata_dir / "samples.json",
        image_source_dir=image_dir,
        output_dir=output_dir,
        max_samples=10,
    )

    assert summary.converted_images == 1
    assert summary.converted_boxes == 2
    assert (output_dir / "images" / "train" / "frame1.png").exists()
    label_text = (output_dir / "labels" / "train" / "frame1.txt").read_text(encoding="utf-8")
    assert "2 0.300000 0.500000 0.400000 0.600000" in label_text
    assert "1 0.100000 0.100000 0.200000 0.200000" in label_text
    yaml_text = (output_dir / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert '2: "Car"' in yaml_text
