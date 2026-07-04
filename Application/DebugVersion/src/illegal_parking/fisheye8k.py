from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .training_plan import TRAFFIC_OBJECT_CLASSES, build_yolo_data_yaml


@dataclass(frozen=True)
class Fisheye8kConversionSummary:
    converted_images: int
    converted_boxes: int
    skipped_missing_images: int
    skipped_unknown_labels: int
    val_source: str
    output_dir: str
    data_yaml: str

    def to_dict(self) -> dict:
        return asdict(self)


def convert_fiftyone_samples_to_yolo(
    metadata_path: str | Path,
    image_source_dir: str | Path,
    output_dir: str | Path,
    max_samples: int | None = None,
    max_per_split: int | None = None,
    split_tags: tuple[str, ...] = ("train", "val", "test"),
    use_test_as_val_if_missing: bool = True,
    image_resolver: Callable[[str], Path | None] | None = None,
) -> Fisheye8kConversionSummary:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    samples = metadata.get("samples", [])
    source_root = Path(image_source_dir)
    output_root = Path(output_dir)
    class_to_id = {name: index for index, name in enumerate(TRAFFIC_OBJECT_CLASSES)}
    counters = {
        "converted_images": 0,
        "converted_boxes": 0,
        "skipped_missing_images": 0,
        "skipped_unknown_labels": 0,
    }
    split_counts = {split: 0 for split in split_tags}

    for split in split_tags:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        if max_samples is not None and counters["converted_images"] >= max_samples:
            break
        split = _sample_split(sample, split_tags)
        if split is None:
            continue
        if max_per_split is not None and split_counts[split] >= max_per_split:
            if all(split_counts[name] >= max_per_split for name in split_tags):
                break
            continue
        relative_path = sample.get("filepath") or ""
        image_path = _resolve_image(relative_path, source_root, image_resolver)
        if image_path is None or not image_path.exists():
            counters["skipped_missing_images"] += 1
            continue

        output_image = output_root / "images" / split / Path(relative_path).name
        shutil.copy2(image_path, output_image)
        label_lines = []
        for detection in sample.get("detections", {}).get("detections", []):
            label = detection.get("label")
            if label not in class_to_id:
                counters["skipped_unknown_labels"] += 1
                continue
            bbox = yolo_bbox_from_fiftyone(detection.get("bounding_box", []))
            label_lines.append(f"{class_to_id[label]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

        label_path = output_root / "labels" / split / f"{output_image.stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        counters["converted_images"] += 1
        split_counts[split] += 1
        counters["converted_boxes"] += len(label_lines)

    data_yaml = output_root / "data.yaml"
    val_source = "images/val"
    if use_test_as_val_if_missing and split_counts.get("val", 0) == 0 and split_counts.get("test", 0) > 0:
        val_source = "images/test"
    build_yolo_data_yaml(output_root, data_yaml, class_names=TRAFFIC_OBJECT_CLASSES, val=val_source)
    return Fisheye8kConversionSummary(
        converted_images=counters["converted_images"],
        converted_boxes=counters["converted_boxes"],
        skipped_missing_images=counters["skipped_missing_images"],
        skipped_unknown_labels=counters["skipped_unknown_labels"],
        val_source=val_source,
        output_dir=output_root.as_posix(),
        data_yaml=data_yaml.as_posix(),
    )


def yolo_bbox_from_fiftyone(bbox: list[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("FiftyOne bbox must contain [x, y, width, height].")
    x, y, width, height = (_clamp(float(value)) for value in bbox)
    return (
        round(_clamp(x + width / 2), 10),
        round(_clamp(y + height / 2), 10),
        round(width, 10),
        round(height, 10),
    )


def _resolve_image(
    relative_path: str,
    source_root: Path,
    image_resolver: Callable[[str], Path | None] | None,
) -> Path | None:
    local_path = source_root / relative_path
    if local_path.exists():
        return local_path
    if image_resolver is not None:
        return image_resolver(relative_path)
    return local_path


def _sample_split(sample: dict, split_tags: tuple[str, ...]) -> str | None:
    tags = sample.get("tags") or []
    for split in split_tags:
        if split in tags:
            return split
    return None


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
