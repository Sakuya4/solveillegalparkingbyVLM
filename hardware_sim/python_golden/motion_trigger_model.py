from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionStep:
    frame_valid: bool
    global_motion: int
    roi_motion: int
    ack_event: bool = False


def trigger_cycles(
    sequence: list[MotionStep],
    min_roi_motion: int,
    roi_margin: int,
) -> list[int]:
    state = "armed"
    triggers: list[int] = []
    for cycle, step in enumerate(sequence):
        spike = (
            step.frame_valid
            and step.roi_motion >= min_roi_motion
            and step.roi_motion >= step.global_motion + roi_margin
        )
        if state == "armed" and spike:
            triggers.append(cycle)
            state = "latched"
        elif state == "latched" and step.ack_event:
            state = "rearm"
        elif state == "rearm" and not spike:
            state = "armed"
    return triggers
