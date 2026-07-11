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
- [ ] Download and checksum the full 16.76 GB dataset.
- [ ] Add an ACCIDENT dataset adapter and clip sampler.
- [ ] Reproduce naive, optical-flow, and bbox-dynamics smoke baselines.
- [ ] Extract detector/tracker trajectory features around accident timestamps.
- [ ] Train TCN/LSTM temporal baselines.
- [ ] Train or fine-tune a VideoMAE comparison model.
- [ ] Compare YOLO, RT-DETR, and D-FINE as detector front ends.
- [ ] Generate incident evidence with SAM2 and VLM review.
- [ ] Report IID/OOD event recall, false alarms/hour, temporal error, type
  accuracy, and latency.

## Phase 2: Edge And Hardware-Aware Validation

- [ ] Export and quantize the selected models for an edge runtime.
- [ ] Add Verilog motion, acceleration, and time-to-collision trigger logic.
- [ ] Feed real model traces into Verilog golden vectors.
- [ ] Add SystemC camera, CPU, NPU, VLM, and alert queues.
- [ ] Simulate latency, backpressure, frame dropping, utilization, and power
  proxies for multiple cameras.
- [ ] Document the Snapdragon QNN/AI Hub compilation and profiling path.
- [ ] Join alerts with Taiwan A1/A2, violations, VD flow, and CCTV locations.
- [ ] Run live public-CCTV runtime and domain-shift checks.
- [ ] Produce the final accuracy, edge cost, hot-spot coverage, and response-time
  report.

## Final Claim Gate

- [ ] Do not claim reduced A1/A2 casualties without before/after field data.
- [ ] Claim only measured detection delay, event quality, camera-hour false
  alarms, risk coverage, throughput, and review-cost savings.
