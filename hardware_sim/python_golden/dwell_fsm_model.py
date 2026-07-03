from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DwellStep:
    track_valid: bool
    restricted_zone: bool
    stationary: bool


def first_candidate_cycle(sequence: list[DwellStep], threshold_cycles: int) -> int | None:
    dwell_count = 0
    for cycle, step in enumerate(sequence):
        if step.track_valid and step.restricted_zone and step.stationary:
            dwell_count += 1
        else:
            dwell_count = 0

        if dwell_count >= threshold_cycles:
            return cycle
    return None
