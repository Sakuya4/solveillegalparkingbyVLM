from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_continuous_context_study import PROTOCOL, DIRECTORY, load_study, load_models, load_batch, sha, dump


def main():
    protocol, _, _ = load_study()
    definitions = load_models()
    base = ROOT / "docs/results"
    path = base / "continuous_context_inventory.csv"
    manifest = base / "continuous_context_inventory.json"
    if path.exists() or manifest.exists():
        raise FileExistsError("Do not overwrite a public source inventory")
    rows = []
    for cohort in ("calibration", "test"):
        _, reports = load_batch(cohort)
        indexed = {row["path"]: row for row in reports}
        for model in definitions:
            split = model.rsplit("_", 1)[-1]
            for name in protocol["memberships"][split][cohort + "_clip_paths"]:
                report = indexed[name]
                annotation = protocol["annotations"][name]
                rows.append({"cohort": cohort, "model": model, "split": split, "path": name,
                             "source_sha256": report["source_sha256"],
                             "annotation_pts_sec": annotation["event_time_sec"],
                             "collision_type": annotation["collision_type"],
                             "valid_contexts": len(report["window_starts"]),
                             "rejected_contexts": len(report["rejected_window_starts"])})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(manifest, {"protocol_sha256": sha(PROTOCOL),
                    "results_sha256": sha(base / "continuous_context_results.json"),
                    "selection_sha256": sha(base / "continuous_context_selection.json"),
                    "exporter_sha256": sha(Path(__file__)),
                    "inventory": {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "rows": len(rows)},
                    "batch_sha256": {cohort: sha(DIRECTORY / cohort / "batch.json") for cohort in ("calibration", "test")},
                    "scope": "Complete corresponding model/source memberships, including zero-context sources. No new endpoints or source substitutions."})
    print(json.dumps({"inventory_rows": len(rows), "inventory_sha256": sha(path)}))


if __name__ == "__main__":
    main()
