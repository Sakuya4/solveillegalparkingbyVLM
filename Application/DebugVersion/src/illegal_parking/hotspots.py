from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ViolationRecord:
    city: str
    area: str
    road: str
    law: str
    fact: str
    hour: int | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True)
class RoadHotspot:
    city: str
    area: str
    road: str
    violation_count: int
    share_of_total: float
    rank: int
    heat_level: str = "unclassified"
    peak_hour: int | None = None
    latitude: float | None = None
    longitude: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EnforcementCostConfig:
    minutes_per_manual_case: float = 8.0
    hourly_labor_cost: float = 550.0
    system_monthly_cost: float = 0.0


@dataclass(frozen=True)
class InterventionImpact:
    baseline_violations: int
    projected_violations: int
    reduced_violations: int
    reduction_rate: float
    manual_cost_before: float
    manual_cost_after: float
    gross_savings: float
    system_cost: float
    net_savings: float
    savings_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_taoyuan_violation_records(path: str | Path, encoding: str = "big5", limit: int | None = None) -> list[ViolationRecord]:
    records: list[ViolationRecord] = []
    with Path(path).open("r", encoding=encoding, errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = ViolationRecord(
                city="桃園市",
                area=(row.get("AreaName") or "").strip(),
                road=(row.get("Road") or "").strip(),
                law=(row.get("law") or "").strip(),
                fact=(row.get("fact") or "").strip(),
                hour=_parse_hour(row.get("time")),
                latitude=_parse_float(row.get("latitude")),
                longitude=_parse_float(row.get("longitude")),
            )
            if record.road:
                records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def summarize_road_hotspots(records: Iterable[ViolationRecord], top_n: int = 20) -> list[RoadHotspot]:
    record_list = list(records)
    total = len(record_list)
    grouped: dict[tuple[str, str, str], list[ViolationRecord]] = defaultdict(list)
    for record in record_list:
        grouped[(record.city, record.area, record.road)].append(record)

    ranked_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][2]))
    hotspots: list[RoadHotspot] = []
    for rank, ((city, area, road), group) in enumerate(ranked_groups[:top_n], start=1):
        hotspots.append(
            RoadHotspot(
                city=city,
                area=area,
                road=road,
                violation_count=len(group),
                share_of_total=_safe_div(len(group), total),
                rank=rank,
                peak_hour=_most_common_hour(group),
                latitude=_average([record.latitude for record in group]),
                longitude=_average([record.longitude for record in group]),
            )
        )
    return hotspots


def classify_hotspots(
    hotspots: list[RoadHotspot],
    high_ratio: float = 0.2,
    medium_ratio: float = 0.5,
) -> list[RoadHotspot]:
    if not hotspots:
        return []

    high_cutoff = max(1, round(len(hotspots) * high_ratio))
    medium_cutoff = max(high_cutoff, round(len(hotspots) * (high_ratio + medium_ratio)))
    classified: list[RoadHotspot] = []
    for index, hotspot in enumerate(hotspots, start=1):
        if index <= high_cutoff:
            heat_level = "high"
        elif index <= medium_cutoff:
            heat_level = "medium"
        else:
            heat_level = "low"
        classified.append(
            RoadHotspot(
                city=hotspot.city,
                area=hotspot.area,
                road=hotspot.road,
                violation_count=hotspot.violation_count,
                share_of_total=hotspot.share_of_total,
                rank=hotspot.rank,
                heat_level=heat_level,
                peak_hour=hotspot.peak_hour,
                latitude=hotspot.latitude,
                longitude=hotspot.longitude,
            )
        )
    return classified


def estimate_intervention_impact(
    hotspots: list[RoadHotspot],
    high_heat_reduction_rate: float,
    cost_config: EnforcementCostConfig,
) -> InterventionImpact:
    baseline = sum(hotspot.violation_count for hotspot in hotspots)
    projected = 0
    for hotspot in hotspots:
        if hotspot.heat_level == "high":
            projected += round(hotspot.violation_count * (1 - high_heat_reduction_rate))
        else:
            projected += hotspot.violation_count

    reduced = max(0, baseline - projected)
    manual_before = _manual_cost(baseline, cost_config)
    manual_after = _manual_cost(projected, cost_config)
    gross_savings = manual_before - manual_after
    net_savings = gross_savings - cost_config.system_monthly_cost
    return InterventionImpact(
        baseline_violations=baseline,
        projected_violations=projected,
        reduced_violations=reduced,
        reduction_rate=_safe_div(reduced, baseline),
        manual_cost_before=manual_before,
        manual_cost_after=manual_after,
        gross_savings=gross_savings,
        system_cost=cost_config.system_monthly_cost,
        net_savings=net_savings,
        savings_rate=_safe_div(net_savings, manual_before),
    )


def _manual_cost(violations: int, config: EnforcementCostConfig) -> float:
    return violations * (config.minutes_per_manual_case / 60.0) * config.hourly_labor_cost


def _parse_hour(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.strip()
    try:
        if ":" in normalized:
            hour = int(normalized.split(":", 1)[0])
        elif normalized.isdigit() and len(normalized) in (3, 4):
            hour = int(normalized[:-2])
        else:
            hour = int(normalized)
    except (ValueError, IndexError):
        return None
    if 0 <= hour <= 23:
        return hour
    return None


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _most_common_hour(records: list[ViolationRecord]) -> int | None:
    counter = Counter(record.hour for record in records if record.hour is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
