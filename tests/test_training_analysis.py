import csv
import json

from illegal_parking.training_analysis import (
    ClassMetric,
    analyze_training_run,
    load_results_csv,
    summarize_class_metrics,
)


def test_load_results_csv_finds_best_and_last_metrics(tmp_path):
    path = tmp_path / "results.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "1",
                "metrics/precision(B)": "0.1",
                "metrics/recall(B)": "0.2",
                "metrics/mAP50(B)": "0.3",
                "metrics/mAP50-95(B)": "0.15",
            }
        )
        writer.writerow(
            {
                "epoch": "2",
                "metrics/precision(B)": "0.4",
                "metrics/recall(B)": "0.35",
                "metrics/mAP50(B)": "0.5",
                "metrics/mAP50-95(B)": "0.25",
            }
        )

    summary = load_results_csv(path)

    assert summary.best_map50_epoch == 2
    assert summary.best_map50 == 0.5
    assert summary.best_map5095 == 0.25
    assert summary.last_epoch == 2
    assert summary.last_map50 == 0.5


def test_summarize_class_metrics_marks_weak_classes_and_imbalance():
    class_summary = summarize_class_metrics(
        [
            ClassMetric(name="Car", images=100, instances=500, precision=0.7, recall=0.5, map50=0.52, map5095=0.3),
            ClassMetric(
                name="Pedestrian",
                images=40,
                instances=80,
                precision=0.05,
                recall=0.06,
                map50=0.03,
                map5095=0.01,
            ),
        ],
        weak_map50_threshold=0.2,
    )

    assert class_summary.weak_classes == ["Pedestrian"]
    assert class_summary.best_class == "Car"
    assert class_summary.worst_class == "Pedestrian"
    assert class_summary.instance_imbalance_ratio == 6.25


def test_analyze_training_run_combines_results_and_recommendations(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.4,0.3,0.29,0.14\n"
        "2,0.43,0.35,0.329,0.175\n",
        encoding="utf-8",
    )
    class_path = tmp_path / "classes.json"
    class_path.write_text(
        json.dumps(
            [
                {
                    "name": "Car",
                    "images": 1000,
                    "instances": 6090,
                    "precision": 0.592,
                    "recall": 0.514,
                    "map50": 0.517,
                    "map5095": 0.294,
                },
                {
                    "name": "Pedestrian",
                    "images": 594,
                    "instances": 1766,
                    "precision": 0.0649,
                    "recall": 0.0612,
                    "map50": 0.0304,
                    "map5095": 0.0122,
                },
            ]
        ),
        encoding="utf-8",
    )

    report = analyze_training_run(results_path, class_path, model_name="YOLOv8n")

    assert report.model_name == "YOLOv8n"
    assert report.results.best_map50 == 0.329
    assert "Pedestrian" in report.class_summary.weak_classes
    assert any("RT-DETR" in item for item in report.recommendations)
