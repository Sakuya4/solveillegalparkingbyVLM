# Mainline Application

## Goal

Build a Taiwan traffic-violation event system that can run at an edge camera,
produce auditable evidence, and compare its output with public traffic records.

The mainline is no longer only red-line parking. Red-line parking remains the
first rule, but the project becomes a deployable event pipeline:

```text
camera/video/image folder
  -> vehicle detector
  -> tracker
  -> scene/rule evidence
  -> event candidate writer
  -> VLM or human review
  -> city-level validation and reporting
```

## What the Finished Application Looks Like

For a government or campus deployment, each camera runs an edge process:

1. Read frames from RTSP, video files, image folders, or a webcam simulator.
2. Detect vehicles and vulnerable road users with a traffic-camera detector.
3. Track objects over time and estimate whether they are stationary.
4. Check no-parking evidence such as red-line overlap or configured ROIs.
5. Emit one candidate event per track with:
   - original frame
   - annotated frame
   - vehicle crop
   - rule evidence JSON
   - coarse or SAM-assisted vehicle/restricted-zone mask evidence
   - frame index and camera ID
6. Send only candidate events to VLM or human review.
7. Compare event distribution with public violation and accident hot spots.

The practical value is that the system does not claim every detection is a
ticket. It produces reviewable evidence and measurable event quality.

## SAM-Assisted Evidence

The mainline now separates object detection from rule evidence:

```text
detector bbox
  -> vehicle mask provider
  -> restricted-zone mask overlap
  -> auditable event evidence
```

The default provider uses the detector bbox as a coarse vehicle mask, so the
pipeline still works without a large segmentation checkpoint. When
`segment-anything` and a SAM checkpoint are available, `SamPromptMaskProvider`
can use the detector bbox as a prompt and return a tighter vehicle mask.

This matters because the project should not only say "a vehicle was detected."
It can report how much of the vehicle evidence overlaps the red line or
configured no-parking region. That is closer to what a reviewer or government
operator needs for an auditable violation event.

References:

- Segment Anything official repository: https://github.com/facebookresearch/segment-anything
- SAM 2 official repository: https://github.com/facebookresearch/sam2

Run the demo:

```powershell
python scripts\run_mask_evidence_demo.py --output-dir outputs\mask_evidence_demo
```

It writes:

- `original.jpg`
- `bbox_overlay.jpg`
- `vehicle_mask.png`
- `restricted_mask.png`
- `overlap_overlay.jpg`
- `evidence.json`

The default demo uses a synthetic camera frame and bbox-mask fallback. For a
real frame, pass `--image`, `--bbox x1,y1,x2,y2`, and optionally
`--restricted-rect x1,y1,x2,y2` or `--restricted-line x1,y1,x2,y2,width`.
Use `--restricted-line` for red-line parking photos where the no-parking
evidence is a painted curb line. To use SAM, also pass `--sam-checkpoint`.
The overlay uses yellow for the SAM/bbox vehicle mask, red for the restricted
line, blue for the vehicle bottom footprint, and green for the footprint-line
overlap.

Public real-photo demo:

```powershell
New-Item -ItemType Directory -Force -Path outputs\public_illegal_parking_demo
Invoke-WebRequest `
  -Uri "https://upload.wikimedia.org/wikipedia/commons/8/83/FedEx_driver_parked_in_bike_lane.jpg" `
  -OutFile "outputs\public_illegal_parking_demo\source.jpg"

python scripts\run_mask_evidence_demo.py `
  --image outputs\public_illegal_parking_demo\source.jpg `
  --bbox 485,175,1230,930 `
  --restricted-rect 430,760,1290,1040 `
  --output-dir outputs\public_illegal_parking_demo
```

The image is a Wikimedia Commons real-world illegal-parking example: a delivery
truck parked in a bike lane in Washington, D.C. It is licensed CC BY-SA 4.0 by
Sdkb, so it is safer for an open demo than reusing news-site images with unclear
rights.

SAM checkpoint demo:

```powershell
python -m pip install git+https://github.com/facebookresearch/segment-anything.git

New-Item -ItemType Directory -Force -Path models\sam
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
  -OutFile "models\sam\sam_vit_b_01ec64.pth"

python scripts\run_mask_evidence_demo.py `
  --image path\to\redline-parking.jpg `
  --bbox 95,220,525,640 `
  --restricted-line 205,585,138,952,24 `
  --sam-checkpoint models\sam\sam_vit_b_01ec64.pth `
  --sam-model-type vit_b `
  --device cuda `
  --output-dir outputs\user_redline_sam_demo
```

The SAM body mask can be accurate while direct body/red-line overlap remains
small, because the red line is often under the vehicle or beside the tire. The
demo therefore reports both body-mask overlap and bottom-footprint overlap.
For red-line parking review, `footprint_overlap_pixels` and
`footprint_overlap_ratio` are the more useful contact evidence fields.

## Data Roles

| Role | Sources | Purpose |
| --- | --- | --- |
| Vision training | FishEye8K, FE-DETRAC | Train/evaluate detector on Taiwan traffic-camera viewpoints. |
| Violation records | Taipei and Taoyuan traffic violation open data | Build violation type, road, time, and hot-spot baselines. |
| Camera deployment | New Taipei illegal-parking automatic enforcement sites | Ground the demo in real enforcement deployment patterns. |
| Risk validation | National Police Agency A1/A2 accident records | Compare detected events with higher-risk road segments. |

## Prepared Local Data

Run:

```powershell
python scripts\prepare_taiwan_data.py --overwrite
```

The script reads `data/sources/taiwan_transport_sources.json`, downloads default
government open-data sources into `data/raw/taiwan/`, and writes
`data/raw/taiwan/inventory.json`.

Raw data is intentionally not tracked by git.

Training setup and expected pre-training outputs are documented in
`docs/training_plan.md`.

Model comparison strategy is documented in `docs/model_comparison.md`.

## Validation Plan

The project can now be verified at three levels:

1. Model quality
   - Dataset: FishEye8K / FE-DETRAC.
   - Metrics: mAP, precision, recall, FPS, edge latency.

2. Event quality
   - Dataset: replayed images/video with frame-level annotations.
   - Tooling: `scripts/run_edge_simulation.py` and `scripts/evaluate_events.py`.
   - Metrics: event precision, recall, false-positive rate.

3. City-level usefulness
   - Dataset: Taipei/Taoyuan violations, New Taipei camera sites, A1/A2 accidents.
   - Metrics: hot-spot overlap, time-of-day consistency, high-risk-road coverage,
     projected violation reduction, and estimated manual-review cost savings.

4. Live-image usefulness
   - Source: public CCTV pages or approved government CCTV feeds.
   - Tooling: `scripts/run_edge_simulation.py --source cctv-page`.
   - Metrics: effective FPS, read latency, event trigger delay, evidence write latency.

Example:

```powershell
python scripts\run_edge_simulation.py `
  --source cctv-page `
  --path "https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129" `
  --max-frames 5 `
  --read-timeout-sec 5 `
  --source-fps 1 `
  --target-fps 1 `
  --detector none
```

Use `--source image-url` for direct JPEG/PNG snapshot URLs and
`--source stream-url` for RTSP, MJPEG, or HLS URLs that OpenCV can open.

Live CCTV validation proves runtime behavior, throughput, and frame-read
stability. It does not by itself prove violation accuracy, because public CCTV
feeds may not contain labeled violations. Accuracy still comes from replayed
annotated clips and model datasets.

## Hot-Spot Reduction Standard

Public violation records can define the deployment target:

```text
high-heat roads before deployment
  -> automated event detection and review
  -> fewer repeated violations
  -> medium/low heat after deployment
```

Run:

```powershell
python scripts\analyze_hotspots.py `
  --input data/raw/taiwan/taoyuan_traffic_violations_113.csv `
  --top-n 30 `
  --high-heat-reduction-rate 0.35 `
  --minutes-per-manual-case 8 `
  --hourly-labor-cost 550 `
  --system-monthly-cost 50000
```

The output estimates:

- baseline violations in the selected hot spots
- projected violations after automated enforcement
- reduced violations
- manual handling cost before/after
- net savings after system operating cost

This turns hot-spot data into a measurable project target instead of only a
map visualization.

## A1/A2 Accident Risk Standard

A1/A2 data should be used as a risk baseline, not as a direct promise that one
camera immediately prevents crashes. The practical target is:

```text
A1/A2 high-risk grids
  -> deploy cameras or replay validation clips near those grids
  -> detect precursor events such as illegal stopping, lane blockage, red-line parking, or poor sight-line behavior
  -> reduce repeated precursor events
  -> compare later A1/A2 and violation records against the baseline
```

Run:

```powershell
python scripts\analyze_accidents.py `
  --input data/raw/taiwan/taiwan_accidents_a1_a2_113.zip `
  --top-n 30 `
  --grid-precision 3 `
  --a1-weight 5 `
  --a2-weight 1 `
  --high-risk-reduction-rate 0.2
```

The report ranks coordinate-grid hot spots by weighted risk. A1 cases carry a
higher score than A2 cases, so fatal-risk locations move up the deployment
priority list even when the raw case count is smaller.

## Hardware Branch Relationship

The Verilog/SystemC branch implements the deterministic event filter:

```text
software detector/tracker evidence
  -> bbox overlap counter
  -> dwell FSM
  -> candidate event
```

This does not replace YOLO or VLM. It makes the rule stage explainable and
testable like a digital IC block, while the software pipeline handles perception
and reporting.
