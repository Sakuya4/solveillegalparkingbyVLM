from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from .gpu_job import exclusive_gpu_job
from .incident_vlm_evidence import REVIEW_POLICY_VERSION, review_disposition
from .vlm_review import VlmReviewRequest, parse_vlm_review_result_text, build_unparsed_vlm_review_result
from .vlm_transformers import GENERATION_CONTRACT_VERSION, TransformersVlmConfig, run_transformers_image_text_to_text

QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review_request(path: Path) -> VlmReviewRequest:
    request = VlmReviewRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if request.task != "traffic_incident_review" or request.evidence.get("privacy_redacted") is not True:
        raise ValueError("Review requires privacy-treated traffic-incident evidence")
    if len(request.image_paths) != 3 or any(not Path(name).is_absolute() or not Path(name).is_file() for name in request.image_paths):
        raise ValueError("Review requires three existing absolute local image paths")
    return request


def review_incident_requests(root: Path, requests: list[VlmReviewRequest], *, pipeline_factory=None) -> list[dict]:
    if not requests:
        return []
    with exclusive_gpu_job(root):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "2"
        import torch
        if pipeline_factory is None:
            from transformers import pipeline
            pipeline_factory = pipeline
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = TransformersVlmConfig("Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl_3b", device=0 if torch.cuda.is_available() else "cpu")
        print("Loading pinned Qwen evidence-review model", flush=True)
        pipe = pipeline_factory("image-text-to-text", model=config.model_id, revision=QWEN_REVISION, device=config.device, dtype="auto")
        results = []
        for index, request in enumerate(requests):
            print(f"Reviewing evidence {index + 1}/{len(requests)}", flush=True)
            image_hashes = [digest(Path(name)) for name in request.image_paths]
            started = time.perf_counter()
            raw = run_transformers_image_text_to_text(request, config, pipeline_factory=lambda **kwargs: pipe)
            if image_hashes != [digest(Path(name)) for name in request.image_paths]:
                raise ValueError("Evidence images changed during review")
            try:
                review = parse_vlm_review_result_text(raw, provider=config.provider_name)
                valid = True
            except (ValueError, json.JSONDecodeError) as exc:
                review = build_unparsed_vlm_review_result(raw, provider=config.provider_name, error=str(exc))
                valid = False
            results.append({"raw_text": raw, "review": {
                **review.to_dict(), "schema_valid": valid, "elapsed_sec": time.perf_counter() - started,
                "model_id": config.model_id, "model_revision": getattr(pipe.model.config, "_commit_hash", QWEN_REVISION),
                "parameter_dtype": str(next(pipe.model.parameters()).dtype), "device": str(config.device),
                "do_sample": config.do_sample, "max_new_tokens": config.max_new_tokens, "seed": 42,
                "generation_contract_version": GENERATION_CONTRACT_VERSION,
                "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
                "image_sha256": image_hashes,
                "task_scope": "Desktop traffic-incident triage; separate from the collision-specific paired benchmark",
            }})
        return results


def review_saved_events(root: Path, report_path: Path, indices: list[int], output_dir: Path, *, reviewer=review_incident_requests) -> Path:
    if not indices or len(indices) > 3 or len(indices) != len(set(indices)):
        raise ValueError("Select one to three distinct saved events")
    output = output_dir / "report.json"
    if output.exists() or output.resolve() == report_path.resolve():
        raise FileExistsError("Choose a new output directory; source reports are preserved")
    report_hash = digest(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if any(index < 0 or index >= len(report["events"]) for index in indices):
        raise ValueError("Event index is outside the saved report")
    requests, request_hashes = [], []
    for index in indices:
        request_path = Path(report["events"][index]["evidence"]["request_json"])
        requests.append(load_review_request(request_path))
        request_hashes.append(digest(request_path))
    # Resolve retained playback paths before writing a report in a new directory.
    video = Path(report["output_video"])
    report["output_video"] = str((report_path.parent / video).resolve() if not video.is_absolute() else video)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    results = reviewer(root, requests)
    if len(results) != len(indices):
        raise ValueError("Reviewer returned an incomplete event batch")
    if digest(report_path) != report_hash or request_hashes != [digest(Path(report["events"][index]["evidence"]["request_json"])) for index in indices]:
        raise ValueError("Source report or evidence request changed during review")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, result, request_hash in zip(indices, results, request_hashes):
        event = report["events"][index]
        event["vlm_review"] = {**result["review"], "request_sha256": request_hash}
        event["review_disposition"] = review_disposition(True, event["vlm_review"])
        event["review_disposition_policy_version"] = REVIEW_POLICY_VERSION
        raw_path = output_dir / f"event_{index + 1:03d}_vlm_raw.txt"
        raw_path.write_text(result["raw_text"], encoding="utf-8")
        event["vlm_raw_path"] = str(raw_path.resolve())
    report.setdefault("evidence_review_runs", []).append({
        "source_report_sha256": report_hash, "source_report": str(report_path.resolve()),
        "event_indices": indices, "started_at_utc": started_utc,
        "elapsed_sec_including_model_load": time.perf_counter() - started,
        "policy_version": REVIEW_POLICY_VERSION, "detector_rerun": False,
    })
    report["vlm_requested"] = True
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
