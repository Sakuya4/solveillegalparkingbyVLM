# Project Status

## Current Completion

Existing red-line/event platform: **about 85% complete**.

New ACCIDENT incident mainline: **about 90% complete**. All 2,027 real clips
have been processed into 3,387 windows with zero decode failures. Full IID and
geographic global-motion baselines, a causal TCN, a 500-clip trajectory/TTC
ablation, annotation-free tracker/motion candidate ROIs, optional SAM2
refinement, train-holdout threshold calibration, a frozen VideoMAE comparison,
a public-CCTV pilot and session aggregator, a real model-output demo, an
incident VLM evidence package, edge-trace replay, Verilog trigger logic, and
Python/SystemC queue models are implemented. A formal 100-clip event batch,
partial VideoMAE fine-tuning, same-sample SAM2 ablation, clean-blind VLM case,
five-minute government CCTV session, ONNX parity check, and A1/A2 camera-site
coverage report are now complete.

Current positioning: **edge-first traffic incident detection and risk
prioritization for roadside cameras**. Red-line parking is now a completed event
case. ACCIDENT-based temporal incident detection is the new training mainline.

The platform estimate is based on runnable components, tests, demo evidence,
measured detector training, and a real Qwen2.5-VL-3B review. The new mainline is
reported separately so the added research scope is visible.

## Completed Platform Checklist

- [x] Define the project as an application-facing smart-city violation monitoring system.
- [x] Complete red-line parking as the first event case.
- [x] Support image, video, webcam, image URL, stream URL, and public CCTV page frame sources.
- [x] Build detector, tracker, scene evidence, rule engine, and event writer pipeline.
- [x] Emit auditable event artifacts: original frame, annotated frame, crop, masks, and JSON evidence.
- [x] Add red-line/no-parking rule evidence with dwell, speed, overlap, ROI, and mask fields.
- [x] Add privacy-safe demo output for plate blur and timestamp/address removal.
- [x] Treat red-line parking as restricted on both sides of the red curb line.
- [x] Run a real user-photo SAM red-line parking demo.
- [x] Add deterministic VLM-compatible review result baseline.
- [x] Add VLM response normalization and provider comparison report tooling.
- [x] Run a real local VLM on the same review package.
- [x] Add event validation manifest schema and simulated workflow.
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

## ACCIDENT Mainline Checklist

- [x] Select a public fixed-CCTV incident benchmark.
- [x] Verify ACCIDENT access, CC BY-NC-SA 4.0 license, size, and metadata.
- [x] Download real/synthetic metadata locally.
- [x] Download and verify the archive containing all real and synthetic videos.
- [x] Add validated temporal-window and video-file inventory generation.
- [x] Add global/ROI motion feature extraction.
- [x] Add lightweight incident model training and IID/geographic OOD reports.
- [x] Run a 500-clip real-video smoke benchmark with zero decode failures.
- [x] Run full 2,027-clip global motion IID/geographic baselines.
- [x] Extract YOLOv8n/ByteTrack trajectory and image-plane TTC features.
- [x] Train a causal TCN on deployment-safe global motion sequences.
- [x] Separate deployment-safe metrics from oracle accident-ROI upper bounds.
- [x] Generate online tracker/motion candidate ROIs without accident annotations.
- [x] Add optional SAM2 box-prompt refinement with an edge-safe fallback.
- [x] Run a 500-clip candidate ROI logistic/TCN ablation and emit edge traces.
- [x] Calibrate candidate model thresholds on a grouped training holdout.
- [x] Render privacy-treated model output without accident bbox input.
- [x] Train and evaluate a frozen VideoMAE embedding comparison head.
- [x] Compare candidate ROI and VideoMAE fusion under fixed/calibrated thresholds.
- [x] Package confirmed incidents as ordered privacy-treated VLM evidence.
- [x] Add post-inference event recall and trigger-delay evaluation tooling.
- [ ] Reproduce the official heuristic and VLM smoke baselines.
- [x] Fine-tune the final VideoMAE encoder block with train-holdout early stopping.
- [x] Evaluate a representative IID event batch for recall and latency.
- [ ] Run the same event-level batch on the geographic OOD split.
- [x] Integrate incident output with the existing VLM evidence workflow.
- [x] Run a 10-clip SAM2 proposal ablation and a clean-blind real VLM case comparison.
- [ ] Expand the VLM comparison to a labeled validation set.

## VLM Checklist

- [x] Create privacy-aware VLM review request format.
- [x] Create normalized VLM review result format.
- [x] Add offline evidence reviewer as deterministic baseline.
- [x] Add CLI for normalizing saved VLM JSON/text responses.
- [x] Add CLI for comparing VLM providers against the baseline.
- [x] Add source-backed Hugging Face Transformers VLM runner.
- [x] Run a lightweight local LLaVA/Qwen 0.5B VLM smoke test and record schema-following failure.
- [x] Run Qwen2.5-VL-3B on the privacy-safe red-line demo event.
- [x] Add ordered before/trigger/after incident requests and offline dispatch.
- [ ] Add BLIP-2 local runner or retire it if too heavy for the available GPU.
- [x] Record single-event VLM provider comparison table.
- [ ] Add optional cloud VLM runner for high-quality comparison.
- [ ] Report VLM accuracy, confidence calibration, schema-following rate, latency, and human-review rate over an annotated validation set.

## Hardware Branch Checklist

- [x] Add Verilog dwell FSM.
- [x] Add Verilog bbox overlap counter.
- [x] Add Python golden models and test vectors.
- [x] Add self-checking Verilog testbenches.
- [x] Explain how hardware filtering reduces software/VLM review load.
- [x] Add SystemC queue and throughput simulation using measured camera FPS.
- [x] Connect model candidate traces to Python and SystemC queue inputs.
- [x] Run a short official Taichung CCTV domain-shift pilot.
- [x] Add multi-session CCTV aggregation and zero-event 95% exposure bounds.
- [x] Compile and run SystemC 3.0.2 against a real candidate trace.
- [x] Export TCN ONNX, verify parity, and document the QNN submission path.
- [ ] Profile latency and power after QAI Hub credentials or hardware are available.

## What The Project Can Show Today

- A privacy-safe red-line parking demo from a real photo.
- SAM-prompted vehicle mask and bottom-footprint overlap evidence.
- A deterministic review decision: likely violation, confidence, reasons, and human-review flag.
- A first local VLM smoke result showing that small VLMs may run but fail the JSON review contract.
- A stronger Qwen2.5-VL-3B local VLM result that agrees with the evidence baseline on the red-line demo.
- A simulated event validation workflow for testing reports before enough real data exists.
- Public data reports for violation hot spots, A1/A2 accident risk, and projected savings.
- YOLOv8n versus RT-DETR-L detector comparison on the same FishEye8K 1K/1K subset.
- Verilog-style deterministic filtering blocks for dwell and overlap logic.
- A full real-video causal TCN benchmark using no oracle ROI: IID F1 0.594 and
  geographic F1 0.629 over 3,387 windows.
- A measured oracle-ROI gap and a 500-clip trajectory/TTC fusion ablation.
- An annotation-free 500-clip online ROI benchmark: candidate-only logistic F1
  0.581 IID / 0.615 geographic and candidate TCN F1 0.629 / 0.668.
- A shared-NPU capacity result showing RT-DETR-L saturation above two 15 FPS
  camera streams under the measured-latency queue assumptions.
- A real annotation-free model-output video and a public-CCTV normal-stream
  pilot with raw and persistence-gated alerts reported separately.
- A candidate-trace queue replay showing zero NPU frame drops but 68.3% review
  queue drops under a four-camera stress profile.
- A frozen VideoMAE comparison over 822 windows: fixed-threshold F1 0.646 IID /
  0.610 geographic, plus calibrated candidate/VideoMAE fusion.
- A one-clip event-timing integration result with a measured +0.792 second
  trigger delay using the sequence-window end as decision time.
- An incident evidence package with ordered before/trigger/after frames and a
  task-aware VLM review contract that keeps human review mandatory.

## Next Best Milestones

1. Extend normal-only CCTV validation from seconds to independently reviewed hours.
2. Fine-tune VideoMAE and compare detector front ends on IID/OOD event metrics.
3. Run SAM2-assisted evidence and real VLM comparisons on annotated incidents.
4. Install SystemC and run cross-model trace parity checks.
5. Join event results with Taiwan A1/A2, violations, VD flow, and CCTV coverage.
