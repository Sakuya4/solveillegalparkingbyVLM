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
- [ ] Train or fine-tune a VideoMAE comparison model.
- [ ] Compare YOLO, RT-DETR, and D-FINE as detector front ends.
- [ ] Generate incident evidence with SAM2 and VLM review.
- [ ] Report IID/OOD event recall, false alarms/hour, temporal error, type
  accuracy, and latency.

## Phase 2: Edge And Hardware-Aware Validation

- [ ] Export and quantize the selected models for an edge runtime.
- [ ] Add Verilog acceleration and time-to-collision trigger logic.
- [x] Add a Verilog ROI-versus-global motion-spike trigger.
- [x] Add a shared NPU/review queue golden simulation and camera-capacity sweep.
- [ ] Feed real model traces into Verilog golden vectors.
- [x] Add SystemC camera, shared-NPU, and review/alert queues.
- [x] Feed real candidate ROI traces into Python and SystemC queue inputs.
- [x] Simulate latency, backpressure, frame dropping, and utilization for
  multiple cameras.
- [ ] Add CPU/VLM stages and calibrated power proxies to the SystemC model.
- [ ] Document the Snapdragon QNN/AI Hub compilation and profiling path.
- [ ] Join alerts with Taiwan A1/A2, violations, VD flow, and CCTV locations.
- [x] Run an initial live public-CCTV runtime and domain-shift check.
- [ ] Extend normal-only CCTV validation to independently reviewed camera-hours.
- [ ] Produce the final accuracy, edge cost, hot-spot coverage, and response-time
  report.

## Final Claim Gate

- [ ] Do not claim reduced A1/A2 casualties without before/after field data.
- [ ] Claim only measured detection delay, event quality, camera-hour false
  alarms, risk coverage, throughput, and review-cost savings.
