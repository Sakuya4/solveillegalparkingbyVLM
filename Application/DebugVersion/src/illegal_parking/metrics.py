from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


@dataclass(frozen=True)
class BinaryEventMetrics:
    counts: ConfusionCounts
    precision: float
    recall: float
    false_positive_rate: float

    def to_dict(self) -> dict:
        return {
            "tp": self.counts.tp,
            "fp": self.counts.fp,
            "fn": self.counts.fn,
            "tn": self.counts.tn,
            "precision": self.precision,
            "recall": self.recall,
            "false_positive_rate": self.false_positive_rate,
        }


def evaluate_binary_events(y_true: Iterable[bool], y_pred: Iterable[bool]) -> BinaryEventMetrics:
    tp = fp = fn = tn = 0
    for expected, predicted in zip(y_true, y_pred):
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    false_positive_rate = _safe_div(fp, fp + tn)

    return BinaryEventMetrics(
        counts=ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn),
        precision=precision,
        recall=recall,
        false_positive_rate=false_positive_rate,
    )


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
