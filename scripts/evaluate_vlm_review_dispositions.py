from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.incident_vlm_evidence import REVIEW_POLICY_VERSION, review_disposition


def summarize_support(rows, accepted):
    positives = sum(row["expected_collision"] for row in rows)
    true = sum(row["expected_collision"] for row, keep in zip(rows, accepted) if keep)
    count = sum(accepted)
    return {"supported_cases": count, "supported_true_cases": true, "supported_false_cases": count - true,
            "supported_positive_precision": true / count if count else None,
            "support_coverage_of_positive_cases": true / positives if positives else None,
            "unresolved_cases": len(rows) - count}


def main():
    parser = argparse.ArgumentParser(description="Evaluate selective VLM evidence support without treating deferred cases as negative predictions.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/validation/incident_vlm_paired")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/vlm_review_dispositions.json")
    args = parser.parse_args()
    summary_path = ROOT / "docs/results/incident_vlm_qwen_greedy_batch.json"
    measured = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = args.input / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, outcomes, sources = [], [], []
    for case in manifest["cases"]:
        path = args.input / case["case_id"] / "qwen_greedy_result.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["case_id"] != case["case_id"] or row["expected_collision"] != case["expected_collision"]:
            raise ValueError("Blinded review case and evaluator labels differ")
        rows.append(row)
        outcomes.append(review_disposition(True, row))
        sources.append({"case_id": row["case_id"], "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "input_signature": row["input_signature"], "expected_collision": row["expected_collision"], "disposition": outcomes[-1]})
    if len(rows) != measured["total_cases"] or not np.isclose(np.mean([row["elapsed_sec"] for row in rows]), measured["mean_review_sec"], rtol=0, atol=1e-8) or sum(row["schema_valid"] for row in rows) != measured["valid_schema_cases"]:
        raise ValueError("Review cohort does not match the published controlled Qwen study")
    prior = [row["schema_valid"] and row["likely_violation"] and row["confidence"] >= 0.7 for row in rows]
    accepted = [outcome == "consistent_visual_support" for outcome in outcomes]
    result = {"model": measured["model"], "model_revision": measured["model_revision"], "decoding": measured["decoding"], "cases": len(rows),
              "source_clips": manifest["selected_clips"], "policy_version": REVIEW_POLICY_VERSION,
              "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              "confidence_only_support": summarize_support(rows, prior), "structured_support": summarize_support(rows, accepted),
              "dispositions": dict(Counter(outcomes)), "source_cases": sources,
              "protocol": "Apply a candidate-positive review gate to the fixed blinded paired-sequence cohort; honor schema, evidence gaps, review requests, confidence and rationale without changing raw VLM predictions",
              "endpoint": "Selective positive visual support and unresolved coverage; abstentions are not true negatives and raw classification metrics remain unchanged",
              "scope": "Paired annotation-defined sequences are not a measured TCN-triggered field cohort; emitted model reasons are not independent ground truth"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("confidence_only_support", "structured_support", "dispositions")}, indent=2))


if __name__ == "__main__":
    main()
