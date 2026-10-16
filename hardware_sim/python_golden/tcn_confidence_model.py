from __future__ import annotations

from dataclasses import dataclass
import math


UNORM_MAX = 65535


def quantize_probability(value: float) -> int:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Confidence values must be finite and within [0, 1]")
    return math.floor(value * UNORM_MAX + .5)


@dataclass(frozen=True)
class ConfidenceStep:
    decision_valid: bool
    context_reset: bool
    probability: int
    threshold: int


class ConfidenceGate:
    def __init__(self):
        self.positive_run = 0

    def step(self, item: ConfidenceStep) -> tuple[int, int, int]:
        if type(item.decision_valid) is not bool or type(item.context_reset) is not bool:
            raise ValueError("Control inputs must be boolean")
        if any(type(value) is not int or not 0 <= value <= UNORM_MAX for value in (item.probability, item.threshold)):
            raise ValueError("Confidence codes must be unsigned 16-bit integers")
        review, pulse = 0, 0
        if item.context_reset:
            self.positive_run = 0
        elif item.decision_valid:
            if item.probability < item.threshold:
                self.positive_run = 0
            elif self.positive_run == 0:
                self.positive_run = 1
            else:
                review = 1
                pulse = int(self.positive_run == 1)
                self.positive_run = 2
        return review, pulse, self.positive_run


def context_decisions(starts, probabilities, threshold):
    if len(starts) != len(probabilities) or any(type(start) is not int or start < 0 or start % 15 for start in starts) or any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError("Decisions require aligned probabilities on unique increasing hop-15 slots")
    gate = ConfidenceGate()
    threshold_code = quantize_probability(threshold)
    tokens, decisions = [], []

    def append(item):
        result = gate.step(item)
        tokens.append((int(item.decision_valid), int(item.context_reset), item.probability, item.threshold, *result))
        return result

    append(ConfidenceStep(False, True, 0, threshold_code))
    previous = None
    for start, probability in zip(starts, probabilities):
        if previous is not None and start != previous + 15:
            append(ConfidenceStep(False, True, 0, threshold_code))
        # An idle cycle holds persistence; it is not a missing source context.
        append(ConfidenceStep(False, False, 0, threshold_code))
        code = quantize_probability(probability)
        result = append(ConfidenceStep(True, False, code, threshold_code))
        decisions.append({"start": start, "probability_code": code, "threshold_code": threshold_code,
                          "positive": code >= threshold_code, "review_flag": bool(result[0]), "episode_pulse": bool(result[1])})
        previous = start
    return tokens, decisions
