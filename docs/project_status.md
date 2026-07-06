# Project Status

## Current Completion

Estimated completion: **60% to 65%**.

This estimate is based on runnable project components, tested scripts, generated
demo evidence, and measured training results. The remaining work is mostly full
training, real VLM adapter runs, event-level validation, and hardware/runtime
profiling.

## Mainline Checklist

- [x] Define the project as an edge-camera traffic-violation event system.
- [x] Support image, video, webcam, image URL, stream URL, and public CCTV page frame sources.
- [x] Build detector, tracker, scene evidence, rule engine, and event writer pipeline.
- [x] Emit auditable event artifacts: original frame, annotated frame, crop, masks, and JSON evidence.
- [x] Add red-line/no-parking rule evidence with dwell, speed, overlap, ROI, and mask fields.
- [x] Add privacy-safe demo output for plate blur and timestamp/address removal.
- [x] Treat red-line parking as restricted on both sides of the red curb line.
- [x] Run a real user-photo SAM red-line parking demo.
- [x] Add deterministic VLM-compatible review result baseline.
- [x] Add VLM response normalization and provider comparison report tooling.
- [ ] Run a real local or cloud VLM on the same review package.
- [ ] Build a small annotated event validation set for illegal parking clips/photos.
- [ ] Measure event precision, recall, false-positive rate, and trigger delay.

## Data And Validation Checklist

- [x] Prepare Taiwan government open-data download inventory.
- [x] Use violation records for hot-spot ranking and projected manual-review savings.
- [x] Use A1/A2 accident records for risk-priority ranking.
- [x] Use public CCTV pages for runtime and frame-read validation.
- [x] Explain that government open data is for deployment/validation, not image training.
- [ ] Add before/after intervention validation once repeated monthly data is available.
- [ ] Connect detected event locations to hot-spot and A1/A2 reports in one final dashboard/report table.

## Training And Model Checklist

- [x] Add FishEye8K / FE-DETRAC planning path.
- [x] Convert FishEye8K-style FiftyOne data into YOLO format.
- [x] Run local GPU YOLOv8n quick smoke training.
- [x] Run YOLOv8n 1K/1K source-backed baseline training.
- [x] Run RT-DETR-L 1K/1K comparison training.
- [x] Document why YOLO is the speed baseline, not the final project claim.
- [x] Add SAM/SAM2-assisted evidence path.
- [ ] Download and verify the full FishEye8K / FE-DETRAC image data locally.
- [ ] Train full YOLO, RT-DETR, and D-FINE baselines.
- [ ] Add SAM-assisted pseudo-label workflow for vehicle/curb/red-line masks.
- [ ] Compare models by event quality, not only detector mAP.

## VLM Checklist

- [x] Create privacy-aware VLM review request format.
- [x] Create normalized VLM review result format.
- [x] Add offline evidence reviewer as deterministic baseline.
- [x] Add CLI for normalizing saved VLM JSON/text responses.
- [x] Add CLI for comparing VLM providers against the baseline.
- [x] Add source-backed Hugging Face Transformers VLM runner.
- [x] Run a lightweight local LLaVA/Qwen 0.5B VLM smoke test and record schema-following failure.
- [ ] Add BLIP-2 local runner or retire it if too heavy for the available GPU.
- [ ] Run one stronger modern VLM such as Qwen-VL/LLaVA-style local inference on the demo event.
- [ ] Add optional cloud VLM runner for high-quality comparison.
- [ ] Report VLM accuracy, confidence calibration, schema-following rate, latency, and human-review rate.

## Hardware Branch Checklist

- [x] Add Verilog dwell FSM.
- [x] Add Verilog bbox overlap counter.
- [x] Add Python golden models and test vectors.
- [x] Add self-checking Verilog testbenches.
- [x] Explain how hardware filtering reduces software/VLM review load.
- [ ] Add SystemC queue and throughput simulation using measured camera FPS.
- [ ] Connect hardware filter vectors to real event evidence.
- [ ] Document Snapdragon/QNN export and profiling path.

## What The Project Can Show Today

- A privacy-safe red-line parking demo from a real photo.
- SAM-prompted vehicle mask and bottom-footprint overlap evidence.
- A deterministic review decision: likely violation, confidence, reasons, and human-review flag.
- A first local VLM smoke result showing that small VLMs may run but fail the JSON review contract.
- Public data reports for violation hot spots, A1/A2 accident risk, and projected savings.
- YOLOv8n versus RT-DETR-L detector comparison on the same FishEye8K 1K/1K subset.
- Verilog-style deterministic filtering blocks for dwell and overlap logic.

## Next Best Milestones

1. Run Qwen2.5-VL-3B or another stronger local VLM on the existing user-photo review package and compare schema-following and decision quality with the offline baseline.
2. Build a 20 to 50 image/clip event validation set with labels: violation, no violation, uncertain.
3. Train or validate D-FINE/RT-DETR/YOLO on the same data slice and report event-level quality.
4. Add SystemC throughput simulation tied to measured pipeline FPS.
5. Produce the final report table: detector metrics, VLM review metrics, event metrics, hot-spot reduction estimate, and hardware/runtime cost.
