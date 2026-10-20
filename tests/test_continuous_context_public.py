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


def test_context_study_preserves_all_bundled_reference_binaries():
    parent = json.loads((BASE / "presentation_protocol.json").read_text())
    for name, expected in parent["reference_model_sha256"].items():
        assert digest(ROOT / "Application/desktop/models" / name) == expected


def test_new_rtl_summary_covers_every_selected_public_context_and_source():
    path = BASE / "continuous_context_rtl.json"
    if not path.exists():
        pytest.skip("Controller replay not yet published")
    report = json.loads(path.read_text())
    study_path = BASE / "continuous_context_results.json"
    study = json.loads(study_path.read_text())
    protocol = json.loads((BASE / "continuous_training_protocol.json").read_text())
    assert report["results_sha256"] == digest(study_path)
    assert report["rtl_golden_mismatches"] == 0
    assert report["simulation_output"].splitlines()[0] == f'PASS tcn_confidence_gate_tb tokens={report["tokens_checked"]}'
    with (ROOT / study["files"]["contexts"]["path"]).open(newline="", encoding="utf-8") as handle:
        contexts = list(csv.DictReader(handle))
    for row in report["summaries"]:
        if row.get("status") == "unselected":
            continue
        expected = [value for value in contexts if value["cohort"] == row["cohort"] and value["model"] == row["model"]]
        split = row["model"].rsplit("_", 1)[-1]
        assert row["decisions"] == len(expected)
        assert row["source_runs"] == len(protocol["memberships"][split][row["cohort"] + "_clip_paths"])
        assert row["maximum_probability_error"] <= .5 / 65535 + 1e-12
