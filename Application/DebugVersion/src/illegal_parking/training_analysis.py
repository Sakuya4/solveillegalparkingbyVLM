from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingCurveSummary:
    best_map50_epoch: int
    best_map50: float
    best_map5095_epoch: int
    best_map5095: float
    best_precision: float
    best_recall: float
    last_epoch: int
    last_map50: float
    last_map5095: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClassMetric:
    name: str
    images: int
    instances: int
    precision: float
    recall: float
    map50: float
    map5095: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClassMetricSummary:
    classes: list[ClassMetric]
    weak_classes: list[str]
    best_class: str | None
    worst_class: str | None
    instance_imbalance_ratio: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["classes"] = [item.to_dict() for item in self.classes]
        return payload


@dataclass(frozen=True)
class TrainingAnalysisReport:
    model_name: str
    results: TrainingCurveSummary
    class_summary: ClassMetricSummary | None
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "results": self.results.to_dict(),
            "class_summary": self.class_summary.to_dict() if self.class_summary else None,
            "recommendations": self.recommendations,
        }


def load_results_csv(path: str | Path) -> TrainingCurveSummary:
    rows = list(csv.DictReader(Path(path).open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("results.csv has no metric rows.")

    best_map50 = max(rows, key=lambda row: _float(row, "metrics/mAP50(B)"))
    best_map5095 = max(rows, key=lambda row: _float(row, "metrics/mAP50-95(B)"))
    best_precision = max(rows, key=lambda row: _float(row, "metrics/precision(B)"))
    best_recall = max(rows, key=lambda row: _float(row, "metrics/recall(B)"))
    last = rows[-1]
    return TrainingCurveSummary(
        best_map50_epoch=_int(best_map50, "epoch"),
        best_map50=_float(best_map50, "metrics/mAP50(B)"),
        best_map5095_epoch=_int(best_map5095, "epoch"),
        best_map5095=_float(best_map5095, "metrics/mAP50-95(B)"),
        best_precision=_float(best_precision, "metrics/precision(B)"),
        best_recall=_float(best_recall, "metrics/recall(B)"),
        last_epoch=_int(last, "epoch"),
        last_map50=_float(last, "metrics/mAP50(B)"),
        last_map5095=_float(last, "metrics/mAP50-95(B)"),
    )


def load_class_metrics(path: str | Path) -> list[ClassMetric]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ClassMetric(
            name=item["name"],
            images=int(item["images"]),
            instances=int(item["instances"]),
            precision=float(item["precision"]),
            recall=float(item["recall"]),
            map50=float(item["map50"]),
            map5095=float(item["map5095"]),
        )
        for item in payload
    ]


def summarize_class_metrics(
    classes: list[ClassMetric],
    weak_map50_threshold: float = 0.2,
) -> ClassMetricSummary:
    weak = [item.name for item in classes if item.map50 < weak_map50_threshold]
    best = max(classes, key=lambda item: item.map50).name if classes else None
    worst = min(classes, key=lambda item: item.map50).name if classes else None
    nonzero_instances = [item.instances for item in classes if item.instances > 0]
    imbalance = 0.0
    if nonzero_instances:
        imbalance = round(max(nonzero_instances) / min(nonzero_instances), 3)
    return ClassMetricSummary(
        classes=classes,
        weak_classes=weak,
        best_class=best,
        worst_class=worst,
        instance_imbalance_ratio=imbalance,
    )


def analyze_training_run(
    results_csv: str | Path,
    class_metrics_json: str | Path | None = None,
    model_name: str = "detector",
) -> TrainingAnalysisReport:
    results = load_results_csv(results_csv)
    class_summary = summarize_class_metrics(load_class_metrics(class_metrics_json)) if class_metrics_json else None
    return TrainingAnalysisReport(
        model_name=model_name,
        results=results,
        class_summary=class_summary,
        recommendations=_recommend(results, class_summary),
    )


def _recommend(results: TrainingCurveSummary, class_summary: ClassMetricSummary | None) -> list[str]:
    recommendations: list[str] = []
    if results.best_map50 < 0.5:
        recommendations.append("Train RT-DETR and D-FINE baselines because YOLOv8n mAP50 is below 0.50.")
    if results.best_map5095 < 0.25:
        recommendations.append("Prioritize localization-focused models and segmentation evidence because mAP50-95 is low.")
    if class_summary and class_summary.weak_classes:
        recommendations.append(
            "Use class-aware error analysis for weak classes: " + ", ".join(class_summary.weak_classes) + "."
        )
    if class_summary and class_summary.instance_imbalance_ratio >= 5:
        recommendations.append("Address dataset imbalance with class-aware sampling or targeted data expansion.")
    recommendations.append("Report event precision/recall in addition to detector mAP before claiming deployment value.")
    return recommendations


def _float(row: dict[str, str], key: str) -> float:
    return float((row.get(key) or "0").strip())


def _int(row: dict[str, str], key: str) -> int:
    return int(float((row.get(key) or "0").strip()))
