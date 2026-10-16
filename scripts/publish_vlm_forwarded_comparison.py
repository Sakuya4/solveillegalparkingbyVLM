from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.incident_vlm_evidence import review_disposition
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_correctness(rows_a, rows_b):
    a, b = ({row["case_id"]: row for row in rows} for rows in (rows_a, rows_b))
    if a.keys() != b.keys() or any(a[key]["clip"] != b[key]["clip"] or a[key]["expected_collision"] != b[key]["expected_collision"] for key in a):
        raise ValueError("Paired comparison requires identical source clips and case labels")
    clusters = sorted({row["clip"] for row in rows_a})
    total, correct_a, correct_b = [], [], []
    for clip in clusters:
        keys = [key for key in a if a[key]["clip"] == clip]
        total.append(len(keys))
        for rows, values in ((a, correct_a), (b, correct_b)):
            values.append(sum(rows[key]["schema_valid"] and rows[key]["likely_violation"] == rows[key]["expected_collision"] for key in keys))
    return paired_cluster_coverage_interval(np.asarray(total), np.asarray(correct_a), np.asarray(correct_b))


def load_verified(summary_path, directory, provider, *, version_2):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if version_2 and (summary.get("generation_contract_version") != 2 or summary["source_manifest_sha256"] != digest(manifest_path)):
        raise ValueError("Generation contract or manifest hash mismatch")
    suffix = "forwarded_v2" if version_2 else "greedy"
    published = {row["case_id"]: row for row in summary.get("case_results", [])}
    rows, sources = [], []
    for case in manifest["cases"]:
        path = directory / case["case_id"] / f"{provider}_{suffix}_result.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["case_id"] != case["case_id"] or row["expected_collision"] != case["expected_collision"]:
            raise ValueError("Case identity or label mismatch")
        if version_2:
            request = json.loads(Path(case["request"]).read_text(encoding="utf-8"))
            fingerprint = {"request": request, "model": summary["model"], "revision": summary["model_revision"],
                           "max_new_tokens": summary["max_new_tokens"], "do_sample": summary["do_sample"],
                           "generation_contract_version": 2, "transformers_version": summary["transformers_version"],
                           "wrapper_sha256": summary["wrapper_sha256"], "pipeline_source_sha256": summary["pipeline_source_sha256"],
                           "image_hashes": [digest(Path(name)) for name in request["image_paths"]]}
            signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
            if signature != row["input_signature"] or published[case["case_id"]]["report_sha256"] != digest(path):
                raise ValueError("Case input or published result hash mismatch")
        rows.append({**row, "clip": case["clip"]})
        sources.append({"case_id": case["case_id"], "sha256": digest(path), "input_signature": row["input_signature"]})
    valid = [row for row in rows if row["schema_valid"]]
    confusion = {
        "tp": sum(row["expected_collision"] and row["likely_violation"] for row in valid),
        "fp": sum(not row["expected_collision"] and row["likely_violation"] for row in valid),
        "fn": sum(row["expected_collision"] and not row["likely_violation"] for row in valid),
        "tn": sum(not row["expected_collision"] and not row["likely_violation"] for row in valid),
    }
    correct = sum(row["schema_valid"] and row["likely_violation"] == row["expected_collision"] for row in rows)
    if len(rows) != summary["total_cases"] or len(valid) != summary["valid_schema_cases"] or confusion != summary["valid_case_confusion"] or not np.isclose(correct / len(rows), summary["end_to_end_correct_rate"]) or not np.isclose(np.mean([row["elapsed_sec"] for row in rows]), summary["mean_review_sec"], rtol=0, atol=1e-8):
        raise ValueError("Case results do not reproduce published summary metrics")
    summary = {**summary, "summary_sha256": digest(summary_path), "source_cases": sources,
               "evidence_policy_dispositions": dict(Counter(review_disposition(True, row) for row in rows)),
               "decoding_scope": "Explicit forwarded greedy generation" if version_2 else "Preserved requested-greedy run; effective override not verified"}
    return summary, rows


def main():
    definitions = (("qwen3", "qwen", ""), ("llava3", "llava", ""), ("qwen8", "qwen", "_dense8"))
    records, cases = {}, {}
    for label, provider, suffix in definitions:
        directory = ROOT / "outputs/validation" / ("incident_vlm_paired_dense8" if suffix else "incident_vlm_paired")
        for version in ("legacy", "v2"):
            stem = "forwarded_v2" if version == "v2" else "greedy"
            path = ROOT / f"docs/results/incident_vlm_{provider}{suffix}_{stem}_batch.json"
            key = f"{label}_{version}"
            records[key], cases[key] = load_verified(path, directory, provider, version_2=version == "v2")
    comparisons = {}
    for a, b in (("qwen3_v2", "llava3_v2"), ("qwen8_v2", "qwen3_v2"), ("qwen3_v2", "qwen3_legacy"), ("llava3_v2", "llava3_legacy"), ("qwen8_v2", "qwen8_legacy")):
        comparisons[f"{a}_minus_{b}"] = paired_correctness(cases[a], cases[b])
    result = {"studies": records, "paired_all_case_correctness": comparisons,
              "uncertainty": "2,000 paired source-clip bootstrap resamples, seed 42; all cases within a sampled clip retained together; ten source clusters",
              "scope": "Paired annotation-defined collision/pre-event sequences; invalid schemas count as incorrect; does not estimate field incident prevalence or independent VLM truth",
              "legacy_policy": "All original artifacts and measured service workloads remain unchanged"}
    output = ROOT / "docs/results/vlm_forwarded_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
