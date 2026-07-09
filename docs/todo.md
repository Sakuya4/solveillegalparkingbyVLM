# Project TODO

## Mainline: Smart-City Traffic Violation Monitoring System

- [x] Reposition the project as an application-facing smart-city violation monitoring system.
- [x] Keep red-line parking as the first complete event case.
- [x] Build edge simulation from image/video/webcam sources.
- [x] Emit auditable candidate events with evidence artifacts.
- [x] Add frame-level event evaluation.
- [x] Prepare Taiwan open-data sources locally without committing raw data.
- [x] Add live CCTV validation source for public camera pages.
- [x] Add violation hot-spot analysis and cost-savings estimation.
- [x] Add A1/A2 accident risk hot-spot analysis.
- [x] Add FishEye8K / FE-DETRAC YOLO dataset planner and estimated training report.
- [x] Add FishEye8K FiftyOne-to-YOLO converter.
- [x] Run local GPU YOLOv8n sanity training on a FishEye8K quick subset.
- [x] Add detector training error analysis and weak-class recommendations.
- [x] Run RT-DETR-L detector comparison on the same FishEye8K 1K/1K subset.
- [x] Add SAM-ready mask evidence interface for detector bbox prompts.
- [x] Add runnable SAM/bbox mask-evidence demo artifacts.
- [x] Add public real-photo illegal-parking demo command.
- [x] Run real SAM checkpoint demo with red-line footprint evidence.
- [x] Add privacy redaction and two-sided red-line contact band for demos.
- [x] Add VLM review request package for red-line parking events.
- [x] Add deterministic VLM-compatible decision baseline.
- [x] Add VLM response normalization and provider comparison tooling.
- [x] Add source-backed Transformers VLM runner for Qwen/LLaVA-style models.
- [x] Run lightweight local LLaVA/Qwen 0.5B VLM smoke and record schema failure.
- [x] Run Qwen2.5-VL-3B local review on the privacy-safe red-line demo.
- [x] Add single-event VLM comparison result table.
- [x] Add event validation manifest schema and simulated evaluation workflow.
- [ ] Download full FishEye8K / FE-DETRAC image data and verify YOLO labels locally.
- [ ] Train full detector baseline and fine-tuned traffic-camera model.
- [ ] Add D-FINE comparison path.
- [ ] Run VLM adapters over a labeled event validation set.
- [ ] Add SAM-assisted pseudo-label workflow for vehicle/red-line masks.
- [ ] Compare YOLO-only, fine-tuned detector, and SAM-assisted event logic.
- [ ] Build a final report table for accuracy, latency, violation reduction, accident-risk coverage, and estimated savings.

## Branch: Hardware-Aware Event Filter

- [x] Add Verilog dwell FSM.
- [x] Add Verilog bbox overlap counter.
- [x] Add Python golden models and vectors.
- [x] Add self-checking Verilog testbenches.
- [x] Document how the hardware-aware branch supports detector error control.
- [ ] Add SystemC queue/throughput simulation using measured live-camera FPS.
- [ ] Connect hardware filter vectors to real pipeline event evidence.
- [ ] Document Snapdragon/QNN export and profiling path.

## Validation Milestones

- [x] Runtime validation: read public CCTV frames and report effective FPS.
- [x] Open-data validation: rank roads by violation hot spots.
- [x] Risk validation: rank A1/A2 accident hot spots by coordinate grid.
- [ ] Intervention validation: compare before/after windows once repeated monthly data is available.
- [x] Training validation: report mAP, recall, FPS, and edge latency for first trained models.
- [ ] Event validation: report precision, recall, false-positive rate, and trigger delay on annotated clips.
