from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


def load_camera_sites(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sites = []
    for row in rows:
        try:
            sites.append(
                {
                    "id": row["seqno"],
                    "location": row["location"],
                    "item": row["item"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sites


def summarize_risk_camera_coverage(
    hotspots: list[dict[str, Any]],
    camera_sites: list[dict[str, Any]],
    radii_m: tuple[float, ...] = (500.0, 1000.0, 2000.0),
) -> dict[str, Any]:
    if not hotspots or not camera_sites:
        raise ValueError("hotspots and camera sites must be non-empty")
    if any(radius <= 0 for radius in radii_m):
        raise ValueError("coverage radii must be positive")

    matches = []
    for hotspot in hotspots:
        nearest = min(
            camera_sites,
            key=lambda site: haversine_m(
                float(hotspot["latitude"]),
                float(hotspot["longitude"]),
                site["latitude"],
                site["longitude"],
            ),
        )
        distance_m = haversine_m(
            float(hotspot["latitude"]),
            float(hotspot["longitude"]),
            nearest["latitude"],
            nearest["longitude"],
        )
        matches.append(
            {
                "risk_rank": int(hotspot["rank"]),
                "grid_id": hotspot["grid_id"],
                "representative_location": hotspot["representative_location"],
                "risk_score": float(hotspot["risk_score"]),
                "nearest_camera_id": nearest["id"],
                "nearest_camera_location": nearest["location"],
                "distance_m": distance_m,
            }
        )

    total_risk = sum(match["risk_score"] for match in matches)
    coverage = {}
    for radius in radii_m:
        covered = [match for match in matches if match["distance_m"] <= radius]
        covered_risk = sum(match["risk_score"] for match in covered)
        coverage[str(int(radius))] = {
            "covered_hotspots": len(covered),
            "hotspot_coverage_rate": len(covered) / len(matches),
            "covered_risk_score": covered_risk,
            "risk_weighted_coverage_rate": covered_risk / total_risk if total_risk else 0.0,
        }
    return {
        "hotspot_count": len(matches),
        "camera_site_count": len(camera_sites),
        "total_top_hotspot_risk_score": total_risk,
        "coverage_by_radius_m": coverage,
        "nearest_camera_matches": sorted(matches, key=lambda item: item["risk_rank"]),
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(value))
