from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.incident_review_service import QWEN_REVISION, load_review_request, review_saved_events
from illegal_parking.vlm_transformers import GENERATION_CONTRACT_VERSION

BASE = ROOT / "docs/results"
SOURCE = ROOT / "outputs/streaming/live_20261020_daytime/report.json"
PLAN = BASE / "live_review_20261020_protocol.json"
OUTPUT = BASE / "live_review_20261020.json"
INTERMEDIATE = ROOT / "outputs/streaming/live_20261020_daytime_review_123"
FINAL = ROOT / "outputs/streaming/live_20261020_daytime_review_all"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    if path.exists():
        raise FileExistsError("Do not overwrite review artifacts")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def freeze():
    report = json.loads(SOURCE.read_text())
    if len(report["events"]) != 4:
        raise ValueError("Retain exactly the four original candidate events")
    events = []
    for index, event in enumerate(report["events"]):
        path = Path(event["evidence"]["request_json"])
        request = load_review_request(path)
        events.append({"event_index": index, "request_sha256": sha(path),
                       "image_sha256": [sha(Path(name)) for name in request.image_paths],
                       "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest()})
    paths = [Path(__file__), ROOT / "Application/DebugVersion/src/illegal_parking/incident_review_service.py",
             ROOT / "Application/DebugVersion/src/illegal_parking/vlm_transformers.py",
             ROOT / "Application/DebugVersion/src/illegal_parking/incident_vlm_evidence.py"]
    save(PLAN, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_report_sha256": sha(SOURCE), "events": events, "batches": [[0, 1, 2], [3]],
                "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "model_revision": QWEN_REVISION,
                "generation_contract_version": GENERATION_CONTRACT_VERSION, "do_sample": False,
                "max_new_tokens": 256, "seed": 42, "threads": 2,
                "implementation_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
                "scope": "All four retained daytime candidates; three clean privacy-treated images each. Fixed two-batch service limit, without detector reruns. Record malformed or inconclusive outputs without retries or event selection.",
                "label_policy": "UNASSIGNED; VLM opinions do not create independent ground truth or normal-camera exposure.",
                "publication": "Numeric dispositions, schema/service statistics and hashes only; all footage, requests and generated text remain local."})


def load_plan():
    plan = json.loads(PLAN.read_text())
    if plan["source_report_sha256"] != sha(SOURCE):
        raise ValueError("Original observation report changed")
    for name, expected in plan["implementation_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Declared review implementation changed")
    return plan


def run():
    plan = load_plan()
    if (FINAL / "report.json").exists() or (INTERMEDIATE / "report.json").exists():
        raise FileExistsError("Preserve prior review attempts before recovery")
    first = review_saved_events(ROOT, SOURCE, plan["batches"][0], INTERMEDIATE)
    review_saved_events(ROOT, first, plan["batches"][1], FINAL)


def validate_review(definition, review):
    for field, value in (("model_revision", QWEN_REVISION), ("generation_contract_version", 2),
                         ("do_sample", False), ("max_new_tokens", 256), ("seed", 42)):
        if review[field] != value:
            raise ValueError("VLM decoding provenance differs")
    for field in ("request_sha256", "image_sha256", "prompt_sha256"):
        if review[field] != definition[field]:
            raise ValueError("VLM evidence provenance differs")


def publish():
    plan = load_plan()
    source = json.loads(SOURCE.read_text())
    path = FINAL / "report.json"
    report = json.loads(path.read_text())
    if len(report["events"]) != 4 or len(report["evidence_review_runs"]) != 2:
        raise ValueError("Incomplete review chain")
    runs = report["evidence_review_runs"]
    if (runs[0]["source_report_sha256"] != sha(SOURCE)
            or runs[1]["source_report_sha256"] != sha(INTERMEDIATE / "report.json")
            or any(row["detector_rerun"] for row in runs)
            or [row["event_indices"] for row in runs] != plan["batches"]):
        raise ValueError("Review chain does not preserve the original source")
    rows = []
    for definition, event, original in zip(plan["events"], report["events"], source["events"]):
        review = event["vlm_review"]
        validate_review(definition, review)
        if event["trigger_time_sec"] != original["trigger_time_sec"] or event["probability"] != original["probability"]:
            raise ValueError("Original temporal event changed")
        rows.append({"event_index": definition["event_index"], "trigger_time_sec": event["trigger_time_sec"],
                     "schema_valid": review["schema_valid"], "likely_violation": review["likely_violation"],
                     "confidence": review["confidence"], "service_sec_excluding_model_load": review["elapsed_sec"],
                     "missing_evidence_items": len(review["missing_evidence"]),
                     "disposition": event["review_disposition"], "independent_label": "UNASSIGNED",
                     "raw_response_sha256": sha(Path(event["vlm_raw_path"])),
                     "request_sha256": review["request_sha256"], "image_sha256": review["image_sha256"]})
    save(OUTPUT, {"protocol_sha256": sha(PLAN), "source_report_sha256": sha(SOURCE), "reviewed_report_sha256": sha(path),
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(), "events": rows,
                  "batch_elapsed_sec_including_model_load": [row["elapsed_sec_including_model_load"] for row in runs],
                  "normal_exposure_added_hours": 0, "scope": plan["scope"], "label_policy": plan["label_policy"]})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "publish"))
    {"freeze": freeze, "run": run, "publish": publish}[parser.parse_args().stage]()
