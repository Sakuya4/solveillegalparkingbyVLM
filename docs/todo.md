# Project TODO

## Completed Case: Red-Line Parking

- [x] Build edge simulation from image, video, webcam, URL, and CCTV sources.
- [x] Detect dwell and two-sided red-line overlap as one temporal event.
- [x] Generate privacy-safe evidence packages.
- [x] Add SAM mask evidence and local Qwen2.5-VL review.
- [x] Compare YOLOv8n and RT-DETR-L detector baselines.
- [x] Add event evaluation, hot-spot analysis, and A1/A2 risk analysis.

## Phase 1: Public-Dataset Incident Model

- [x] Select a public fixed-CCTV dataset: ACCIDENT.
- [x] Verify Kaggle access, license, size, metadata, and split fields.
- [x] Download real/synthetic metadata and annotation classes locally.
- [x] Download and checksum the full 16.76 GB dataset.
- [x] Add an ACCIDENT dataset adapter and clip sampler.
- [x] Validate official metadata and preserve two end-boundary label corrections.
- [x] Add ROI/global optical-flow and frame-difference feature extraction.
- [x] Add a serializable lightweight incident baseline with IID/OOD reporting.
- [x] Run the optical-flow smoke baseline on 500 real clips.
- [x] Run full 2,027-clip global motion IID/geographic baselines.
- [ ] Reproduce naive and bbox-dynamics smoke baselines.
- [x] Extract detector/tracker trajectory and image-plane TTC features.
- [x] Train a causal TCN temporal baseline on global motion sequences.
- [ ] Train an LSTM comparison baseline.
- [x] Measure the gap between deployable global features and oracle accident ROI.
- [x] Replace oracle ROI with online tracker/motion candidate proposals.
- [x] Add optional SAM2 box-prompt refinement with a no-SAM2 fallback.
- [x] Run a 500-clip online ROI logistic/TCN ablation and export edge traces.
- [x] Calibrate model thresholds against a target false-positive budget using a grouped train holdout.
- [x] Train a frozen VideoMAE embedding comparison head.
- [x] Compare candidate ROI and VideoMAE feature fusion.
- [x] Fine-tune the final VideoMAE encoder block with grouped-holdout early stopping.
- [ ] Compare YOLO, RT-DETR, and D-FINE as detector front ends.
- [x] Generate ordered privacy-treated incident evidence for VLM review.
- [x] Run a same-sample SAM2 proposal ablation and clean-blind VLM case comparison.
- [x] Add event recall, trigger-delay, and off-target alert evaluation tooling.
- [x] Publish a 100-clip IID event recall, trigger-delay, type, and off-target alert report.
- [ ] Add a matching geographic OOD event-level batch report.

## Phase 2: Edge And Hardware-Aware Validation

- [x] Export the calibrated TCN to ONNX and verify 128-sample runtime parity.
- [ ] Quantize and profile the model on an authenticated QNN target.
- [ ] Add Verilog acceleration and time-to-collision trigger logic.
- [x] Add a Verilog ROI-versus-global motion-spike trigger.
- [x] Add a shared NPU/review queue golden simulation and camera-capacity sweep.
- [ ] Feed real model traces into Verilog golden vectors.
- [x] Add SystemC camera, shared-NPU, and review/alert queues.
- [x] Feed real candidate ROI traces into Python and SystemC queue inputs.
- [x] Simulate latency, backpressure, frame dropping, and utilization for
  multiple cameras.
- [ ] Add CPU/VLM stages and calibrated power proxies to the SystemC model.
- [x] Document the Snapdragon QNN/AI Hub compilation path and credential boundary.
- [x] Spatially join Taiwan A1/A2 risk grids with public enforcement-camera locations.
- [ ] Add VD flow and event-alert coordinates to the same spatial join.
- [x] Run an initial live public-CCTV runtime and domain-shift check.
- [x] Aggregate matching CCTV sessions with a zero-event 95% exposure bound.
- [ ] Extend normal-only CCTV validation to independently reviewed camera-hours.
- [x] Produce the accuracy, edge cost, hot-spot coverage, and response-time report.

## Difficult-Image And Review Validation

- [x] Run an official NAFNet checkpoint on privacy-safe synthetic motion blur.
- [x] Audit the official SM3Det release, configuration, domain, compute, and license.
- [x] Separate annotated VLM evidence from clean blinded visual review.
- [x] Compare Qwen2.5-VL-3B and LLaVA/Qwen-0.5B schema behavior on one incident.
- [ ] Expand clean-blind VLM evaluation from one event to an annotated set.

## Final Claim Gate

- [ ] Do not claim reduced A1/A2 casualties without before/after field data.
- [ ] Claim only measured detection delay, event quality, camera-hour false
  alarms, risk coverage, throughput, and review-cost savings.
