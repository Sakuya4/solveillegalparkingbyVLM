# Project TODO

## Mainline: Taiwan Violation Event System

- [x] Build edge simulation from image/video/webcam sources.
- [x] Emit auditable candidate events with evidence artifacts.
- [x] Add frame-level event evaluation.
- [x] Prepare Taiwan open-data sources locally without committing raw data.
- [x] Add live CCTV validation source for public camera pages.
- [x] Add violation hot-spot analysis and cost-savings estimation.
- [x] Add A1/A2 accident risk hot-spot analysis.
- [ ] Add FishEye8K / FE-DETRAC training dataset conversion.
- [ ] Train detector baseline and fine-tuned traffic-camera model.
- [ ] Add SAM-assisted pseudo-label workflow for vehicle/red-line masks.
- [ ] Compare YOLO-only, fine-tuned detector, and SAM-assisted event logic.
- [ ] Build a final report table for accuracy, latency, violation reduction, accident-risk coverage, and estimated savings.

## Branch: Hardware-Aware Event Filter

- [x] Add Verilog dwell FSM.
- [x] Add Verilog bbox overlap counter.
- [x] Add Python golden models and vectors.
- [x] Add self-checking Verilog testbenches.
- [ ] Add SystemC queue/throughput simulation using measured live-camera FPS.
- [ ] Connect hardware filter vectors to real pipeline event evidence.
- [ ] Document Snapdragon/QNN export and profiling path.

## Validation Milestones

- [x] Runtime validation: read public CCTV frames and report effective FPS.
- [x] Open-data validation: rank roads by violation hot spots.
- [x] Risk validation: rank A1/A2 accident hot spots by coordinate grid.
- [ ] Intervention validation: compare before/after windows once repeated monthly data is available.
- [ ] Training validation: report mAP, recall, FPS, and edge latency for trained models.
- [ ] Event validation: report precision, recall, false-positive rate, and trigger delay on annotated clips.
