from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application" / "DebugVersion" / "src"))

from illegal_parking.accidents import load_a1_a2_accident_records
from illegal_parking.deployment_coverage import load_camera_sites
from illegal_parking.risk_coverage import aggregate_risk_cells, coverage_index, evaluate_camera_selection, select_camera_budget


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal-holdout camera budget evaluation using Taiwan government records.")
    parser.add_argument("--accidents", type=Path, default=ROOT / "data/raw/taiwan/taiwan_accidents_a1_a2_113.zip")
    parser.add_argument("--cameras", type=Path, default=ROOT / "data/raw/taiwan/new_taipei_illegal_parking_camera_sites.csv")
    parser.add_argument("--city", default="新北市")
    parser.add_argument("--split-date", default="20241001")
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/results/government_camera_budget")
    args = parser.parse_args()
    if args.random_repeats <= 0:
        parser.error("--random-repeats must be positive")
    all_records = load_a1_a2_accident_records(args.accidents)
    city_records = [record for record in all_records if record.city == args.city and record.category in {"A1", "A2"} and len(record.date) == 8 and record.date.isdigit()]
    train = [record for record in city_records if record.date < args.split_date]
    holdout = [record for record in city_records if record.date >= args.split_date]
    if not train or not holdout:
        raise ValueError("Both temporal partitions must contain crash records")
    cameras = load_camera_sites(args.cameras)
    budgets = sorted({value for value in (1, 3, 5, 10, len(cameras)) if value <= len(cameras)})
    rows = []
    selections = []
    for a1_weight in (1.0, 5.0, 10.0):
        train_cells = aggregate_risk_cells(train, a1_weight=a1_weight)
        test_cells = aggregate_risk_cells(holdout, a1_weight=a1_weight)
        for radius in (500, 1000, 2000):
            train_index = coverage_index(cameras, train_cells, radius)
            test_index = coverage_index(cameras, test_cells, radius)
            risk_weights = {cell["id"]: cell["risk_weight"] for cell in train_cells}
            count_weights = {cell["id"]: cell["crash_count"] for cell in train_cells}
            for budget in budgets:
                for method in ("risk_greedy", "count_greedy", "independent_rank", "random"):
                    samples = []
                    repeats = range(args.random_repeats) if method == "random" else range(1)
                    for seed in repeats:
                        algorithm = {"risk_greedy": "weighted_greedy", "count_greedy": "weighted_greedy"}.get(method, method)
                        weights = count_weights if method == "count_greedy" else risk_weights
                        selected = select_camera_budget(train_index, weights, budget, method=algorithm, seed=seed)
                        result = evaluate_camera_selection(selected, test_index, test_cells)
                        samples.append(result["risk_coverage"])
                        if method != "random":
                            selections.append({"a1_weight": a1_weight, "radius_m": radius, "budget": budget, "method": method, "selected_camera_ids": selected, **result})
                    rows.append({"a1_weight": a1_weight, "radius_m": radius, "budget": budget, "method": method, "holdout_risk_coverage_mean": statistics.mean(samples), "holdout_risk_coverage_sd": statistics.stdev(samples) if len(samples) > 1 else 0.0, "repeats": len(samples)})
    protocol = {
        "research_question": "Can historical severity-weighted crash demand improve proximity coverage under a fixed monitoring-camera budget?",
        "city": args.city, "train_period": [min(record.date for record in train), max(record.date for record in train)],
        "holdout_period": [min(record.date for record in holdout), max(record.date for record in holdout)],
        "raw_records": len(all_records), "city_records": len(city_records),
        "train_records": len(train), "holdout_records": len(holdout),
        "train_geocoded_records": sum(cell["crash_count"] for cell in train_cells),
        "holdout_geocoded_records": sum(cell["crash_count"] for cell in test_cells),
        "camera_sites": len(cameras), "grid_precision_degrees": 3,
        "a1_weights": [1, 5, 10], "a2_weight": 1, "radii_m": [500, 1000, 2000],
        "random_seeds": [0, args.random_repeats - 1],
        "coverage_endpoint": "Proximity coverage of held-out crash severity within a fixed radius; camera orientation and visibility are separate operational attributes.",
        "camera_inventory_design": "Fixed current inventory used for retrospective temporal generalization; camera installation dates are not inferred.",
        "source_sha256": {args.accidents.name: file_hash(args.accidents), args.cameras.name: file_hash(args.cameras)},
        "sources": ["https://data.gov.tw/dataset/172969", "https://data.ntpc.gov.tw/datasets/bb59a616-3572-4e92-9d09-01bf422057a6"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_records": len(train), "holdout_records": len(holdout), "comparison_rows": len(rows), "output": str(args.output_dir)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
