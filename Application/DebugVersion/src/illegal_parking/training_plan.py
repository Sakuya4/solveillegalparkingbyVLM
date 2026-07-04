from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


TRAFFIC_OBJECT_CLASSES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloDatasetConfig:
    dataset_root: str
    output_path: str
    class_names: list[str]
    train: str = "images/train"
    val: str = "images/val"
    test: str = "images/test"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class YoloSplitSummary:
    split: str
    image_count: int
    label_count: int
    missing_label_count: int
    extra_label_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class YoloDatasetSummary:
    dataset_root: str
    splits: dict[str, YoloSplitSummary]
    total_images: int
    total_labels: int
    ready_for_training: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["splits"] = {name: summary.to_dict() for name, summary in self.splits.items()}
        return payload


@dataclass(frozen=True)
class TrainingEstimateInput:
    dataset_id: str
    image_count: int
    box_count: int
    model_size: str = "yolov8n"
    image_size: int = 640
    edge_profile: str = "snapdragon_orin_class"


@dataclass(frozen=True)
class TrainingOutcomeEstimate:
    dataset_id: str
    model_size: str
    image_size: int
    evidence_level: str
    expected_map50_range: tuple[float, float]
    expected_event_precision_range: tuple[float, float]
    expected_edge_fps_range: tuple[float, float]
    expected_training_hours_range: tuple[float, float]
    training_run_command: str
    assumptions: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def build_yolo_data_yaml(
    dataset_root: str | Path,
    output_path: str | Path,
    class_names: list[str] | None = None,
    train: str = "images/train",
    val: str = "images/val",
    test: str = "images/test",
) -> YoloDatasetConfig:
    classes = class_names or TRAFFIC_OBJECT_CLASSES
    root = Path(dataset_root)
    output = Path(output_path)
    lines = [
        f"path: {root.as_posix()}",
        f"train: {train}",
        f"val: {val}",
        f"test: {test}",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend(f'  {index}: "{name}"' for index, name in enumerate(classes))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return YoloDatasetConfig(
        dataset_root=root.as_posix(),
        output_path=output.as_posix(),
        class_names=classes,
        train=train,
        val=val,
        test=test,
    )


def inspect_yolo_dataset(dataset_root: str | Path, splits: tuple[str, ...] = ("train", "val", "test")) -> YoloDatasetSummary:
    root = Path(dataset_root)
    split_summaries: dict[str, YoloSplitSummary] = {}
    for split in splits:
        image_dir, label_dir = _split_dirs(root, split)
        image_stems = _file_stems(image_dir, IMAGE_EXTENSIONS)
        label_stems = _file_stems(label_dir, {".txt"})
        missing = image_stems - label_stems
        extra = label_stems - image_stems
        split_summaries[split] = YoloSplitSummary(
            split=split,
            image_count=len(image_stems),
            label_count=len(label_stems),
            missing_label_count=len(missing),
            extra_label_count=len(extra),
        )

    total_images = sum(summary.image_count for summary in split_summaries.values())
    total_labels = sum(summary.label_count for summary in split_summaries.values())
    ready = total_images > 0 and all(summary.missing_label_count == 0 for summary in split_summaries.values())
    return YoloDatasetSummary(
        dataset_root=root.as_posix(),
        splits=split_summaries,
        total_images=total_images,
        total_labels=total_labels,
        ready_for_training=ready,
    )


def estimate_training_outcome(inputs: TrainingEstimateInput) -> TrainingOutcomeEstimate:
    scale_factor = _dataset_scale_factor(inputs.image_count, inputs.box_count)
    base_map50 = _base_map50(inputs.dataset_id)
    model_bonus = _model_bonus(inputs.model_size)
    resolution_bonus = 0.03 if inputs.image_size >= 960 else 0.0
    center = min(0.9, base_map50 + scale_factor + model_bonus + resolution_bonus)
    expected_map50 = (_clamp(center - 0.08, 0.0, 0.95), _clamp(center + 0.06, 0.0, 0.95))

    precision_center = max(0.45, center - 0.08)
    expected_event_precision = (
        _clamp(precision_center - 0.10, 0.0, 0.95),
        _clamp(precision_center + 0.08, 0.0, 0.95),
    )

    return TrainingOutcomeEstimate(
        dataset_id=inputs.dataset_id,
        model_size=inputs.model_size,
        image_size=inputs.image_size,
        evidence_level="estimated_before_training",
        expected_map50_range=expected_map50,
        expected_event_precision_range=expected_event_precision,
        expected_edge_fps_range=_edge_fps_range(inputs.model_size, inputs.image_size, inputs.edge_profile),
        expected_training_hours_range=_training_hours_range(inputs.image_count, inputs.image_size),
        training_run_command=(
            "yolo detect train "
            f"model={inputs.model_size}.pt data=data/processed/training/{inputs.dataset_id}.yaml "
            f"epochs=100 imgsz={inputs.image_size}"
        ),
        assumptions=[
            "The dataset is already in or converted to Ultralytics YOLO detection format.",
            "The estimate uses public traffic-camera benchmark scale, not this project's measured training result.",
            "Event precision is lower than detector mAP because illegal-parking logic also depends on tracking and rule evidence.",
            "After real training, this estimate must be replaced by validation mAP, event precision/recall, and edge FPS.",
        ],
    )


def _split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists() and (root / split / "images").exists():
        image_dir = root / split / "images"
    if not label_dir.exists() and (root / split / "labels").exists():
        label_dir = root / split / "labels"
    return image_dir, label_dir


def _file_stems(directory: Path, extensions: set[str]) -> set[str]:
    if not directory.exists():
        return set()
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }


def _dataset_scale_factor(image_count: int, box_count: int) -> float:
    if image_count >= 8000 and box_count >= 100000:
        return 0.08
    if image_count >= 3000 and box_count >= 30000:
        return 0.04
    if image_count >= 1000:
        return 0.02
    return -0.03


def _base_map50(dataset_id: str) -> float:
    normalized = dataset_id.lower()
    if normalized == "fisheye8k":
        return 0.68
    if normalized in {"fe_detrac", "fe-detrac"}:
        return 0.66
    return 0.60


def _model_bonus(model_size: str) -> float:
    normalized = model_size.lower()
    if normalized.endswith("n"):
        return -0.03
    if normalized.endswith("s"):
        return 0.0
    if normalized.endswith("m"):
        return 0.03
    if normalized.endswith("l") or normalized.endswith("x"):
        return 0.05
    return 0.0


def _edge_fps_range(model_size: str, image_size: int, edge_profile: str) -> tuple[float, float]:
    normalized = model_size.lower()
    if normalized.endswith("n"):
        base = (12.0, 28.0)
    elif normalized.endswith("s"):
        base = (8.0, 18.0)
    else:
        base = (3.0, 10.0)
    if image_size >= 960:
        return (round(base[0] * 0.45, 2), round(base[1] * 0.55, 2))
    if "cpu" in edge_profile.lower():
        return (round(base[0] * 0.35, 2), round(base[1] * 0.45, 2))
    return base


def _training_hours_range(image_count: int, image_size: int) -> tuple[float, float]:
    base = max(0.5, image_count / 8000 * 2.0)
    if image_size >= 960:
        base *= 1.8
    return (round(base, 2), round(base * 2.5, 2))


def _clamp(value: float, lower: float, upper: float) -> float:
    return round(min(max(value, lower), upper), 3)
