from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))


def prepare(args):
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from illegal_parking.accident_dataset import load_accident_manifest
    from illegal_parking.incident_batch import select_evaluation_clips
    from illegal_parking.incident_vlm_evidence import write_clean_review_package
    from run_candidate_incident_video import _blur_plate_region

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    metadata = ROOT / "data/raw/accident/full/extracted/metadata-real.csv"
    dataset_root = metadata.parent
    clips = [clip for clip in load_accident_manifest(metadata) if clip.accident_time_sec >= 4 and clip.duration_sec - clip.accident_time_sec > 1]
    selected = select_evaluation_clips(clips, split_scheme="iid", max_clips=args.clips, seed=42)
    detector = YOLO(str(ROOT / "yolov8n.pt"))
    cases = []
    for clip in selected:
        video_path = dataset_root / clip.relative_path
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        try:
            if not capture.isOpened() or fps <= 0:
                raise ValueError(f"Cannot decode {clip.relative_path}")
            for positive, offsets in ((False, (-4, -3, -2)), (True, (-1, 0.25, 1))):
                chosen_offsets = offsets if args.frames == 3 else np.linspace(offsets[0], offsets[-1], args.frames)
                times = tuple(max(0.0, clip.accident_time_sec + offset) for offset in chosen_offsets)
                frames = []
                for timestamp in times:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Cannot decode evidence at {timestamp}")
                    frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
                    frames.append(frame)
                predictions = detector.predict(frames, device="cpu", imgsz=320, conf=0.15, classes=[2, 3, 5, 7], verbose=False)
                for frame, prediction in zip(frames, predictions):
                    height, width = frame.shape[:2]
                    for box in prediction.boxes.xyxy.cpu().numpy():
                        _blur_plate_region(frame, SimpleNamespace(x1=box[0] / width, y1=box[1] / height, x2=box[2] / width, y2=box[3] / height))
                case_id = clip.relative_path.stem + ("__collision" if positive else "__pre_event")
                case_dir = args.output / case_id
                representatives = (0, len(frames) // 2, len(frames) - 1)
                artifacts = write_clean_review_package(tuple(frames[index] for index in representatives), tuple(times[index] for index in representatives), case_dir, privacy_method="Tracked-vehicle lower-region blur; no detector overlays or model scores")
                request_path = Path(artifacts["request_json"])
                request = json.loads(request_path.read_text(encoding="utf-8"))
                if args.frames != 3:
                    image_paths = []
                    for frame_index, frame in enumerate(frames):
                        image_path = case_dir / f"frame_{frame_index:03d}.jpg"
                        if not cv2.imwrite(str(image_path), frame):
                            raise OSError(f"Could not write {image_path}")
                        image_paths.append(str(image_path.resolve()))
                    request["image_paths"] = image_paths
                    request["evidence"]["timestamps_sec"] = list(times)
                request["prompt"] = (
                    "Review these ordered traffic frames for visible evidence of an actual vehicle collision. "
                    "Pre-collision behavior, near misses, and normal stopped traffic alone do not qualify. "
                    "Do not infer hidden identifiers or traffic signals. The likely_violation field means a visible collision in this task. "
                    "Return only JSON with likely_violation (boolean), confidence (0 to 1), visual_reasons (list), "
                    "missing_evidence (list), and human_review_needed (boolean)."
                )
                request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
                cases.append({"case_id": case_id, "clip": clip.relative_path.as_posix(), "collision_type": clip.collision_type, "expected_collision": positive, "request": str(request_path.resolve()), "frame_times_sec": list(times), "source_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest()})
        finally:
            capture.release()
    manifest = {
        "protocol": "Paired annotation-defined collision and pre-event sequences from IID test clips",
        "negative_offsets_sec": [-4, -3, -2] if args.frames == 3 else list(np.linspace(-4, -2, args.frames)),
        "positive_offsets_sec": [-1, 0.25, 1] if args.frames == 3 else list(np.linspace(-1, 1, args.frames)),
        "selection_seed": 42, "selected_clips": len(selected), "cases": cases,
        "frames_per_case": args.frames,
        "label_source": "ACCIDENT collision timestamp annotations; validation frame selection is separate from the deployment trigger",
        "blinding": "Requests contain visual evidence only; labels and accident timestamps remain in this evaluator manifest",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(cases)} paired validation cases", flush=True)


def review(args):
    import torch
    from transformers import pipeline
    from illegal_parking.vlm_review import VlmReviewRequest, parse_vlm_review_result_text, build_unparsed_vlm_review_result
    from illegal_parking.vlm_transformers import GENERATION_CONTRACT_VERSION, TransformersVlmConfig, run_transformers_image_text_to_text
    import transformers

    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct" if args.provider == "qwen" else "llava-hf/llava-interleave-qwen-0.5b-hf"
    config = TransformersVlmConfig(model_id, args.provider, device=0 if torch.cuda.is_available() else "cpu", max_new_tokens=192)
    legacy = json.loads((ROOT / "docs/results" / f"incident_vlm_{args.provider}_greedy_batch.json").read_text(encoding="utf-8"))
    pipe = pipeline("image-text-to-text", model=model_id, revision=legacy["model_revision"], device=config.device, dtype="auto")
    results = []
    for index, case in enumerate(manifest["cases"]):
        request_path = Path(case["request"])
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        fingerprint = {
            "request": request_payload, "model": model_id,
            "revision": getattr(pipe.model.config, "_commit_hash", None),
            "max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample,
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "transformers_version": transformers.__version__,
            "wrapper_sha256": hashlib.sha256((ROOT / "Application/DebugVersion/src/illegal_parking/vlm_transformers.py").read_bytes()).hexdigest(),
            "pipeline_source_sha256": hashlib.sha256((Path(transformers.__file__).parent / "pipelines/image_text_to_text.py").read_bytes()).hexdigest(),
            "image_hashes": [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in request_payload["image_paths"]],
        }
        signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        result_path = args.output / case["case_id"] / f"{args.provider}_forwarded_v2_result.json"
        cached = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        if cached and cached.get("input_signature") == signature:
            row = cached
            row["expected_collision"] = case["expected_collision"]
        else:
            request = VlmReviewRequest.from_dict(request_payload)
            started = time.perf_counter()
            raw = run_transformers_image_text_to_text(request, config, pipeline_factory=lambda **kwargs: pipe)
            elapsed = time.perf_counter() - started
            try:
                assessment = parse_vlm_review_result_text(raw, provider=args.provider)
                valid = True
            except (ValueError, json.JSONDecodeError) as exc:
                assessment = build_unparsed_vlm_review_result(raw, provider=args.provider, error=str(exc))
                valid = False
            row = {"case_id": case["case_id"], "input_signature": signature, "expected_collision": case["expected_collision"], "schema_valid": valid, "elapsed_sec": elapsed, **assessment.to_dict()}
            result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            result_path.with_suffix(".txt").write_text(raw, encoding="utf-8")
        results.append(row)
        print(f'[{index + 1}/{len(manifest["cases"])}] {args.provider} {case["case_id"]} schema={row["schema_valid"]}', flush=True)
    valid = [row for row in results if row["schema_valid"]]
    tp = sum(row["expected_collision"] and row["likely_violation"] for row in valid)
    fp = sum(not row["expected_collision"] and row["likely_violation"] for row in valid)
    fn = sum(row["expected_collision"] and not row["likely_violation"] for row in valid)
    tn = sum(not row["expected_collision"] and not row["likely_violation"] for row in valid)
    summary = {
        "model": model_id, "provider": args.provider, "total_cases": len(results),
        "model_revision": getattr(pipe.model.config, "_commit_hash", None),
        "parameter_dtype": str(next(pipe.model.parameters()).dtype),
        "device": str(config.device),
        "valid_schema_cases": len(valid), "schema_valid_rate": len(valid) / len(results),
        "valid_case_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "valid_case_accuracy": (tp + tn) / len(valid) if valid else None,
        "end_to_end_correct_rate": (tp + tn) / len(results),
        "mean_review_sec": sum(row["elapsed_sec"] for row in results) / len(results),
        "operator_review_rate": sum(row["human_review_needed"] for row in results) / len(results),
        "max_new_tokens": 192, "protocol": manifest["protocol"],
        "evaluation_unit": "paired sequences clustered within source clips",
        "frames_per_case": manifest.get("frames_per_case", 3),
        "decoding": "greedy", "do_sample": False, "seed": 42,
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "transformers_version": transformers.__version__,
        "generation_parameter_route": "generate_kwargs: do_sample=False, max_new_tokens=192",
        "wrapper_sha256": fingerprint["wrapper_sha256"],
        "pipeline_source_sha256": fingerprint["pipeline_source_sha256"],
        "model_generation_defaults": pipe.model.generation_config.to_diff_dict(),
        "source_manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
        "case_results": [{"case_id": row["case_id"], "input_signature": row["input_signature"], "report_sha256": hashlib.sha256((args.output / row["case_id"] / f"{args.provider}_forwarded_v2_result.json").read_bytes()).hexdigest()} for row in results],
    }
    suffix = "" if summary["frames_per_case"] == 3 else f'_dense{summary["frames_per_case"]}'
    destination = ROOT / "docs/results" / f"incident_vlm_{args.provider}{suffix}_forwarded_v2_batch.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare and review paired clean-blind incident evidence.")
    parser.add_argument("mode", choices=("prepare", "review"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation/incident_vlm_paired")
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--provider", choices=("qwen", "llava"), default="qwen")
    parser.add_argument("--frames", type=int, choices=(3, 8), default=3)
    args = parser.parse_args()
    if args.clips <= 0:
        parser.error("--clips must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args) if args.mode == "prepare" else review(args)


if __name__ == "__main__":
    main()
