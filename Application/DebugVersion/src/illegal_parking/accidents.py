from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class AccidentRecord:
    accident_id: str
    category: str
    date: str
    time: str
    city: str
    area: str
    location: str
    hour: int | None
    cause: str
    longitude: float | None
    latitude: float | None
    deaths: int = 0
    injuries: int = 0


@dataclass(frozen=True)
class AccidentHotspot:
    grid_id: str
    city: str
    area: str
    representative_location: str
    accident_count: int
    a1_count: int
    a2_count: int
    deaths: int
    injuries: int
    risk_score: float
    share_of_total_risk: float
    rank: int
    heat_level: str
    peak_hour: int | None
    latitude: float | None
    longitude: float | None
    common_cause: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AccidentRiskImpactConfig:
    high_risk_reduction_rate: float = 0.2


@dataclass(frozen=True)
class AccidentRiskImpact:
    baseline_risk_score: float
    projected_risk_score: float
    reduced_risk_score: float
    reduction_rate: float
    baseline_cases: int
    projected_cases: float
    reduced_cases: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_a1_a2_accident_records(path: str | Path, limit: int | None = None) -> list[AccidentRecord]:
    records: list[AccidentRecord] = []
    seen: set[str] = set()
    with ZipFile(path) as archive:
        for name in _accident_csv_names(archive):
            with archive.open(name) as binary_file:
                text_file = io.TextIOWrapper(binary_file, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text_file)
                for row in reader:
                    record = _record_from_row(row)
                    if record.accident_id in seen:
                        continue
                    seen.add(record.accident_id)
                    records.append(record)
                    if limit is not None and len(records) >= limit:
                        return records
    return records


def summarize_accident_hotspots(
    records: list[AccidentRecord],
    top_n: int = 20,
    grid_precision: int = 3,
    a1_weight: float = 5.0,
    a2_weight: float = 1.0,
    high_ratio: float = 0.2,
    medium_ratio: float = 0.5,
) -> list[AccidentHotspot]:
    grouped: dict[str, list[AccidentRecord]] = defaultdict(list)
    for record in records:
        grouped[_grid_id(record, grid_precision)].append(record)

    scored = [
        (grid_id, group, _risk_score(group, a1_weight=a1_weight, a2_weight=a2_weight))
        for grid_id, group in grouped.items()
    ]
    scored.sort(key=lambda item: (-item[2], -len(item[1]), item[0]))
    total_risk = sum(score for _, _, score in scored)

    selected = scored[:top_n]
    high_cutoff = max(1, round(len(selected) * high_ratio)) if selected else 0
    medium_cutoff = max(high_cutoff, round(len(selected) * (high_ratio + medium_ratio))) if selected else 0

    hotspots: list[AccidentHotspot] = []
    for rank, (grid_id, group, score) in enumerate(selected, start=1):
        first = group[0]
        if rank <= high_cutoff:
            heat_level = "high"
        elif rank <= medium_cutoff:
            heat_level = "medium"
        else:
            heat_level = "low"
        hotspots.append(
            AccidentHotspot(
                grid_id=grid_id,
                city=first.city,
                area=first.area,
                representative_location=_most_common(record.location for record in group) or first.location,
                accident_count=len(group),
                a1_count=sum(1 for record in group if record.category == "A1"),
                a2_count=sum(1 for record in group if record.category == "A2"),
                deaths=sum(record.deaths for record in group),
                injuries=sum(record.injuries for record in group),
                risk_score=score,
                share_of_total_risk=_safe_div(score, total_risk),
                rank=rank,
                heat_level=heat_level,
                peak_hour=_most_common_hour(group),
                latitude=_average(record.latitude for record in group),
                longitude=_average(record.longitude for record in group),
                common_cause=_most_common(record.cause for record in group if record.cause),
            )
        )
    return hotspots


def estimate_accident_risk_impact(
    hotspots: list[AccidentHotspot],
    config: AccidentRiskImpactConfig,
) -> AccidentRiskImpact:
    baseline_risk = sum(hotspot.risk_score for hotspot in hotspots)
    projected_risk = 0.0
    baseline_cases = sum(hotspot.accident_count for hotspot in hotspots)
    projected_cases = 0.0
    for hotspot in hotspots:
        if hotspot.heat_level == "high":
            projected_risk += hotspot.risk_score * (1 - config.high_risk_reduction_rate)
            projected_cases += hotspot.accident_count * (1 - config.high_risk_reduction_rate)
        else:
            projected_risk += hotspot.risk_score
            projected_cases += hotspot.accident_count
    reduced_risk = max(0.0, baseline_risk - projected_risk)
    reduced_cases = max(0.0, baseline_cases - projected_cases)
    return AccidentRiskImpact(
        baseline_risk_score=baseline_risk,
        projected_risk_score=projected_risk,
        reduced_risk_score=reduced_risk,
        reduction_rate=_safe_div(reduced_risk, baseline_risk),
        baseline_cases=baseline_cases,
        projected_cases=projected_cases,
        reduced_cases=reduced_cases,
    )


def parse_casualties(value: str | None) -> tuple[int, int]:
    if not value:
        return 0, 0
    deaths = _extract_count(value, "死亡")
    injuries = _extract_count(value, "受傷")
    return deaths, injuries


def _accident_csv_names(archive: ZipFile) -> list[str]:
    metadata = {"manifest.csv", "schema-file.csv", "file.csv"}
    return [
        info.filename
        for info in archive.infolist()
        if info.filename.endswith(".csv") and info.filename not in metadata
    ]


def _record_from_row(row: dict[str, str]) -> AccidentRecord:
    location = (row.get("發生地點") or "").strip()
    city, area = _parse_city_area(location)
    longitude = _parse_float(row.get("經度"))
    latitude = _parse_float(row.get("緯度"))
    deaths, injuries = parse_casualties(row.get("死亡受傷人數"))
    date = (row.get("發生日期") or "").strip()
    time = (row.get("發生時間") or "").strip()
    category = (row.get("事故類別名稱") or "").strip()
    accident_id = "|".join(
        [
            category,
            date,
            time,
            location,
            str(longitude) if longitude is not None else "",
            str(latitude) if latitude is not None else "",
        ]
    )
    return AccidentRecord(
        accident_id=accident_id,
        category=category,
        date=date,
        time=time,
        city=city,
        area=area,
        location=location,
        hour=_parse_hour(time),
        cause=(row.get("肇因研判子類別名稱-主要") or "").strip(),
        longitude=longitude,
        latitude=latitude,
        deaths=deaths,
        injuries=injuries,
    )


def _parse_city_area(location: str) -> tuple[str, str]:
    city_match = re.match(r"(?P<city>.+?[縣市])", location)
    city = city_match.group("city") if city_match else ""
    rest = location[len(city) :] if city else location
    area_match = re.match(r"(?P<area>.+?[區鄉鎮市])", rest)
    area = area_match.group("area") if area_match else ""
    return city, area


def _grid_id(record: AccidentRecord, precision: int) -> str:
    if record.latitude is not None and record.longitude is not None:
        return f"{record.latitude:.{precision}f},{record.longitude:.{precision}f}"
    return f"{record.city}|{record.area}|{record.location}"


def _risk_score(group: list[AccidentRecord], a1_weight: float, a2_weight: float) -> float:
    score = 0.0
    for record in group:
        if record.category == "A1":
            score += a1_weight
        elif record.category == "A2":
            score += a2_weight
    return score


def _parse_hour(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.strip()
    try:
        if ":" in normalized:
            hour = int(normalized.split(":", 1)[0])
        elif normalized.isdigit() and len(normalized) in (3, 4, 5, 6):
            hour = int(normalized[:-4] or "0")
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


def _extract_count(value: str, label: str) -> int:
    match = re.search(rf"{label}\s*(\d+)", value)
    if not match:
        return 0
    return int(match.group(1))


def _most_common_hour(records: list[AccidentRecord]) -> int | None:
    counter = Counter(record.hour for record in records if record.hour is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _most_common(values) -> str | None:
    counter = Counter(value for value in values if value)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _average(values) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
