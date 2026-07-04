from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.training_plan import (
    TRAFFIC_OBJECT_CLASSES,
    TrainingEstimateInput,
    build_yolo_data_yaml,
    estimate_training_outcome,
    inspect_yolo_dataset,
)


DATASET_PRIORS = {
    "fisheye8k": {
        "image_count": 8000,
        "box_count": 157000,
        "source": "https://github.com/MoyoG/FishEye8K",
        "note": "Taiwan fisheye traffic-camera dataset with five object classes.",
    },
    "fe_detrac": {
        "image_count": 140000,
        "box_count": 1000000,
        "source": "https://ark.nchc.org.tw/dataset/fe-detrac",
        "note": "Fisheye traffic-camera benchmark with YOLO/COCO/VOC/MOT formats.",
    },
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Prepare YOLO dataset config and estimate training outcomes.")
    parser.add_argument("--dataset-id", choices=sorted(DATASET_PRIORS), default="fisheye8k")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-yaml", default=None)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--model-size", default="yolov8n")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--edge-profile", default="snapdragon_orin_class")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root or f"data/raw/training/{args.dataset_id}")
    output_yaml = Path(args.output_yaml or f"data/processed/training/{args.dataset_id}.yaml")
    output_report = Path(args.output_report or f"data/processed/training/{args.dataset_id}_training_plan.json")
    prior = DATASET_PRIORS[args.dataset_id]

    config = build_yolo_data_yaml(
        dataset_root=ROOT / dataset_root,
        output_path=ROOT / output_yaml,
        class_names=TRAFFIC_OBJECT_CLASSES,
    )
    summary = inspect_yolo_dataset(ROOT / dataset_root)
    image_count = summary.total_images or prior["image_count"]
    estimate = estimate_training_outcome(
        TrainingEstimateInput(
            dataset_id=args.dataset_id,
            image_count=image_count,
            box_count=prior["box_count"],
            model_size=args.model_size,
            image_size=args.image_size,
            edge_profile=args.edge_profile,
        )
    )
    report = {
        "dataset_id": args.dataset_id,
        "source": prior["source"],
        "note": prior["note"],
        "dataset_config": config.to_dict(),
        "dataset_summary": summary.to_dict(),
        "estimate": estimate.to_dict(),
        "why_estimate_is_allowed_before_training": [
            "Public benchmark metadata gives dataset scale and camera domain.",
            "YOLO transfer learning has known behavior on traffic-camera detection before local fine-tuning.",
            "This project already has event, hot-spot, A1/A2, and live-FPS baselines for downstream estimates.",
            "The report marks these numbers as estimated and keeps the same fields for later measured results.",
        ],
    }

    output_report = ROOT / output_report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
