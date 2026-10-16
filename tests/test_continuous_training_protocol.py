import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def preparer():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_continuous_training_study.py"
    spec = importlib.util.spec_from_file_location("continuous_preparer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_cohort_selection_is_deterministic_partitioned_and_excludes_replays():
    choose = preparer().new_test_sources
    rows = [{"path": str(index), "iid_partition": "test" if index < 8 else "fit"} for index in range(10)]
    a, eligible = choose(rows, "iid", {"0", "1"}, 3)
    assert choose(list(reversed(rows)), "iid", {"0", "1"}, 3) == (a, eligible)
    assert eligible == 6 and len(a) == 3
    assert not set(a) & {"0", "1", "8", "9"}
    with pytest.raises(ValueError, match="Insufficient"):
        choose(rows, "iid", {"0", "1"}, 7)


def test_frozen_training_inputs_preserve_fit_groups_and_block_previous_event_sources():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/results/continuous_training_protocol.json").read_text())
    source = root / "docs/results" / protocol["fit_windows_file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["fit_windows_sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    excluded = set(protocol["excluded_prior_event_sources"])
    for split, definition in protocol["memberships"].items():
        fit, cal, test = (set(definition[key + "_clip_paths"]) for key in ("fit", "calibration", "test"))
        assert not fit & cal and not fit & test and not cal & test
        assert len(test) == 100 and not test & excluded
        for arm, expected in (("anchored", 572 if split == "iid" else 496),
                              ("continuous", 1216 if split == "iid" else 978)):
            selected = [row for row in rows if row["arm"] == arm and split in row["fit_splits"]]
            assert len(selected) == expected
            assert {row["path"] for row in selected} <= fit
            assert {row["target"] for row in selected} == {0, 1}
            assert all(len(row["indices"]) == len(set(row["indices"])) == 16 for row in selected)
    assert protocol["training"]["model_count"] == 8
    assert protocol["training"]["epochs"] == 40
    for name, expected in protocol["implementation_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
