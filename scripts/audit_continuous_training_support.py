from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import probe_times
from illegal_parking.incident_training_support import fit_source_records, audit_training_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def freeze(output):
    if output.exists():
        raise FileExistsError("Do not overwrite a frozen fit-support protocol")
    parent_path = ROOT / "docs/results/presentation_protocol.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    rows_path = parent_path.with_name(parent["windows_file"])
    if sha(rows_path) != parent["windows_sha256"]:
        raise ValueError("Parent partition bytes changed")
    for name, expected in parent["implementation_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Parent implementation changed")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    sources = fit_source_records(rows, parent["input_media"])
    implementations = [Path(__file__), ROOT / "Application/DebugVersion/src/illegal_parking/incident_training_support.py",
                       ROOT / "Application/DebugVersion/src/illegal_parking/incident_presentation_events.py"]
    dump(output, {"status": "frozen_before_fit_support_computation", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "parent_protocol_sha256": sha(parent_path), "parent_windows_sha256": sha(rows_path),
                  "ffprobe_sha256": parent["ffprobe_sha256"], "sources": sources,
                  "implementation_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in implementations},
                  "labels": {"prefix": "end_pts < supplied_event_pts - 1", "target": "supplied_event_pts - 1 <= end_pts <= supplied_event_pts + 3",
                             "post": "Excluded from proposed binary training, but counted in this audit"},
                  "context": "Parent fixed 7.5 Hz PTS grid; 16 distinct frames / 15 steps; hop 15; no interpolation; same lateness bounds",
                  "endpoints": ["Per-source valid/rejected contexts and prefix/target/post counts", "Both-class source support",
                                "Gate-1 support, gate-2 structural support, adjacent target-target support",
                                "Exact frame-identity overlap of anchored and continuous contexts", "Inherited-label versus endpoint-label disagreement"],
                  "aggregation": "All corresponding fit sources retained, including empty/unsupported clips; split-specific counts, no zero-imputed ranking or score metrics",
                  "scope": "Fit-only timestamp/label feasibility audit; no image features, predictions, optimization, calibration endpoints or test outcomes. Packet support is not expanded-context decoder verification.",
                  "comparison_intent": "Same candidate/global TCN features and architecture; annotation-anchored versus continuous-grid training is a joint phase/context/label intervention, not a pure architecture effect.",
                  "next_gate": "Freeze full fit/calibration/test protocol, settings and operating objective before any extraction/fitting; do not select defaults from observed test results"})
    print(json.dumps({"protocol_sha256": sha(output), "fit_source_union": len(sources)}))


def run(protocol_path, dataset_root, ffprobe, output):
    if output.exists() or output.with_suffix(".csv").exists() or output.with_name(output.stem + "_contexts.csv").exists():
        raise FileExistsError("Use separate paths for new audit results")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_fit_support_computation":
        raise ValueError("A predeclared fit-support protocol is required")
    if sha(ffprobe) != protocol["ffprobe_sha256"] or sha(ROOT / "docs/results/presentation_protocol.json") != protocol["parent_protocol_sha256"]:
        raise ValueError("Audited tools or parent protocol changed")
    for name, expected in protocol["implementation_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Fit audit implementation changed")
    summaries, contexts = [], []
    for index, source in enumerate(protocol["sources"], 1):
        path = dataset_root / source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError("Fit source bytes changed")
        times = probe_times(path, ffprobe)
        if len(times) != source["frame_count"] or hashlib.sha256(json.dumps(times).encode()).hexdigest() != source["presentation_times_sha256"]:
            raise ValueError("Fit source presentation support changed")
        summary, rows = audit_training_source(source, times)
        summary["fit_splits"] = ";".join(source["fit_splits"])
        summaries.append(summary)
        contexts.extend({**row, "indices": json.dumps(row["indices"]), "fit_splits": summary["fit_splits"]} for row in rows)
        if index % 100 == 0:
            print(f"Audited fit support {index}/{len(protocol['sources'])}", flush=True)
    files = {}
    for key, rows, target in (("sources", summaries, output.with_suffix(".csv")),
                              ("contexts", contexts, output.with_name(output.stem + "_contexts.csv"))):
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files[key] = {"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target), "rows": len(rows)}
    splits = {}
    for split in ("iid", "geographic"):
        rows = [row for row in summaries if split in row["fit_splits"].split(";")]
        totals = Counter()
        for row in rows:
            totals.update({key: int(value) for key, value in row.items() if key not in ("path", "fit_splits")})
        splits[split] = {"source_clips": len(rows), "empty_clips": sum(row["valid_contexts"] == 0 for row in rows), **dict(totals)}
    result = {"protocol_sha256": sha(protocol_path), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "fit_source_union": len(summaries), "files": files, "splits": splits,
              "scope": protocol["scope"], "training_started": False}
    dump(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze and audit fit-only continuous-context label support without model scores.")
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/continuous_training_support_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/continuous_training_support.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/accident/full/extracted")
    parser.add_argument("--ffprobe", type=Path, default=Path("C:/ffmpeg/bin/ffprobe.exe"))
    args = parser.parse_args()
    freeze(args.protocol) if args.mode == "freeze" else run(args.protocol, args.dataset_root, args.ffprobe, args.output)
