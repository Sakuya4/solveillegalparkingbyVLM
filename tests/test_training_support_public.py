from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1] / "docs/results"


def test_public_fit_support_reconstructs_labels_membership_and_all_denominators():
    protocol = json.loads((BASE / "continuous_training_support_protocol.json").read_text())
    result = json.loads((BASE / "continuous_training_support.json").read_text())
    assert result["protocol_sha256"] == hashlib.sha256((BASE / "continuous_training_support_protocol.json").read_bytes()).hexdigest()
    assert datetime.fromisoformat(protocol["created_at_utc"]) < datetime.fromisoformat(result["completed_at_utc"])
    parent_rows = [json.loads(line) for line in (BASE / "presentation_protocol.jsonl").read_text().splitlines()]
    memberships = {split: {row["path"] for row in parent_rows if row[split + "_partition"] == "fit"}
                   for split in ("iid", "geographic")}
    sources = {row["path"]: row for row in protocol["sources"]}
    groups = defaultdict(list)
    inventory = {}
    for key, definition in result["files"].items():
        path = BASE.parents[1] / definition["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == definition["sha256"]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == definition["rows"]
        if key == "sources":
            inventory = {row["path"]: row for row in rows}
            assert len(inventory) == len(rows) == result["fit_source_union"]
        else:
            assert len({(row["path"], row["start"]) for row in rows}) == len(rows)
            for row in rows:
                source = sources[row["path"]]
                end, event = float(row["end_time_sec"]), source["event_time_sec"]
                assert row["label"] == ("prefix" if end < event - 1 else "target" if end <= event + 3 else "post")
                assert row["fit_splits"].split(";") == source["fit_splits"]
                groups[row["path"]].append(row)
    assert set(inventory) == set(sources)
    for split, names in memberships.items():
        assert {name for name, row in sources.items() if split in row["fit_splits"]} == names
        assert result["splits"][split]["source_clips"] == len(names)
        totals = Counter()
        for name in names:
            rows = groups[name]
            counts = Counter(row["label"] for row in rows)
            starts = {int(row["start"]) for row in rows}
            targets = {int(row["start"]) for row in rows if row["label"] == "target"}
            expected = {"valid_contexts": len(rows), "prefix": counts["prefix"], "target": counts["target"], "post": counts["post"],
                        "both_classes": bool(counts["prefix"] and counts["target"]), "gate1_support": bool(targets),
                        "gate2_structural_support": any(start - 15 in starts for start in targets),
                        "two_target_contexts_adjacent": any(start - 15 in targets for start in targets),
                        "anchored_windows": len(sources[name]["anchored_windows"])}
            for key, value in expected.items():
                assert inventory[name][key] == str(value)
                totals[key] += value
        for key, value in totals.items():
            assert result["splits"][split][key] == value
    assert result["training_started"] is False
