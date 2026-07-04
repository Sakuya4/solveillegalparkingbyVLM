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
   - frame index and camera ID
6. Send only candidate events to VLM or human review.
7. Compare event distribution with public violation and accident hot spots.

The practical value is that the system does not claim every detection is a
ticket. It produces reviewable evidence and measurable event quality.

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
