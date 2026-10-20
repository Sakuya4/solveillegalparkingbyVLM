import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.audit_continuous_contexts import audit, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/results"


def test_public_context_results_reconstruct_without_models_or_media():
    if not (BASE / "continuous_context_inventory.json").exists():
        pytest.skip("Complete event study not yet published")
    result = audit()
    assert result["calibration_points"] == 808
    assert result["outcomes"] == 1200
    assert result["paired_intervals"] == 4


@pytest.mark.parametrize("fault", ["missing", "duplicate", "source", "slot", "outcome", "inventory"])
def test_rehashed_public_corruption_is_rejected(tmp_path, fault):
    if not (BASE / "continuous_context_inventory.json").exists():
        pytest.skip("Complete event study not yet published")
    destination = tmp_path / "docs/results"
    destination.mkdir(parents=True)
    names = ["continuous_training_protocol.json", "continuous_context_results.json", "continuous_context_selection.json",
             "continuous_context_inventory.json", "continuous_context_inventory.csv", "continuous_context_scores.csv",
             "continuous_context_outcomes.csv", "continuous_context_calibration.csv"]
    for name in names:
        shutil.copyfile(BASE / name, destination / name)
    study_path = destination / "continuous_context_results.json"
    manifest_path = destination / "continuous_context_inventory.json"
    study = json.loads(study_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    key = "outcomes" if fault == "outcome" else "contexts"
    record = manifest["inventory"] if fault == "inventory" else study["files"][key]
    path = tmp_path / record["path"]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if fault == "missing":
        rows.pop(0)
    elif fault == "duplicate":
        rows.append(dict(rows[0]))
    elif fault == "source":
        rows[0]["source_sha256"] = "changed"
    elif fault == "slot":
        rows[0]["start"] = "1"
    elif fault == "inventory":
        rows[0]["source_sha256"] = "changed"
    else:
        rows[0]["detected"] = str(1 - int(rows[0]["detected"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    record.update(sha256=digest(path), rows=len(rows))
    study_path.write_text(json.dumps(study), encoding="utf-8")
    manifest["results_sha256"] = digest(study_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        audit(tmp_path)


def test_published_temporal_cpu_percentiles_match_all_raw_calls():
    path = BASE / "continuous_context_latency.json"
    if not path.exists():
        pytest.skip("Latency not yet published")
    report = json.loads(path.read_text())
    assert len(report["results"]) == 8
    for row in report["results"]:
        assert len(row["timings_sec"]) == row["calls"] == 1000
        assert row["median_ms"] == pytest.approx(1000 * np.median(row["timings_sec"]))
        assert row["p95_ms"] == pytest.approx(1000 * np.percentile(row["timings_sec"], 95))
