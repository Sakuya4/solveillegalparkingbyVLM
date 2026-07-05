# Hardware Simulation Notes

This folder sketches the hardware-aware part of the illegal-parking project.
The goal is not to move YOLO or VLM into RTL. The goal is to show which
parts of the edge pipeline are simple, deterministic, and suitable for
hardware-style simulation.

## Scope

The hardware side focuses on event filtering before expensive AI review:

- red-pixel and bbox-band overlap counting
- dwell-time accumulation
- event candidate FSM

The software side remains responsible for:

- YOLO vehicle detection
- VLM review
- reporting and human review workflow

## Files

```text
hardware_sim/
  rtl/
    dwell_fsm.v
    bbox_overlap_counter.v
  testbench/
    dwell_fsm_tb.v
    bbox_overlap_counter_tb.v
  systemc/
    event_pipeline_sim.cpp
  python_golden/
    dwell_fsm_model.py
    overlap_counter_model.py
  test_vectors/
    dwell_fsm_vectors.json
    overlap_counter_vectors.json
```

## How This Helps the Project

The project can now describe the edge device as a mixed software/hardware
pipeline:

1. Python/OpenCV/YOLO generates detections and scene evidence.
2. A hardware-style event filter tracks dwell time and restricted-zone overlap.
3. Only candidate events are forwarded to VLM and human review.

This is similar in spirit to digital IC simulation: define stimuli, simulate
state transitions, compare against a golden model, and only then think about
hardware deployment.

## Mainline Interface

The hardware branch consumes compact evidence from the software pipeline:

```text
track_valid      YOLO/tracker has a stable vehicle track
restricted_zone  red-line or no-parking ROI evidence is present
stationary       track motion is below the configured speed threshold
ack_event        software has written the event/evidence package
```

The Verilog modules are intentionally small:

- `bbox_overlap_counter.v` counts red pixels inside the vehicle contact band.
- `dwell_fsm.v` turns stable restricted-zone occupancy into one candidate event.

This keeps YOLO/VLM in software while making the rule filter easy to reason
about with digital-IC style vectors.

## Current Verification

The repository does not require SystemC or Verilator to run the default tests.
Instead, pytest checks the JSON vectors against the Python golden model first:

```powershell
python -m pytest tests\test_hardware_sim_vectors.py
```

If Icarus Verilog is installed, the self-checking testbenches can be run with:

```powershell
iverilog -g2012 -o hardware_sim\build\dwell_fsm_tb.vvp hardware_sim\rtl\dwell_fsm.v hardware_sim\testbench\dwell_fsm_tb.v
vvp hardware_sim\build\dwell_fsm_tb.vvp

iverilog -g2012 -o hardware_sim\build\bbox_overlap_counter_tb.vvp hardware_sim\rtl\bbox_overlap_counter.v hardware_sim\testbench\bbox_overlap_counter_tb.v
vvp hardware_sim\build\bbox_overlap_counter_tb.vvp
```

SystemC remains the higher-level transaction simulation path. It mirrors the
same evidence signals and helps explain how an edge device would schedule the
rule filter before forwarding candidate events to software.

## Relationship To The Main Detector Experiments

The hardware-aware branch is not a separate product. It supports the main
research question:

```text
detector output is imperfect
  -> deterministic event filter checks dwell time and restricted-zone evidence
  -> only stable candidates reach VLM/human review
  -> fewer false alerts and lower review cost
```

This is important because the first YOLOv8n FishEye8K baseline is weak on some
classes. The system should not trust a single detector frame. It should use:

- detector confidence and class
- tracker stability
- restricted-zone overlap
- dwell-time persistence
- event acknowledgment after evidence is written

The Verilog modules model the low-level rule logic. The SystemC model represents
the transaction-level scheduling question: how many candidate events per second
can the edge device filter before expensive AI review becomes the bottleneck?

## Next SystemC Task

The next SystemC slice should add a queue/throughput simulation:

| Input | Meaning |
| --- | --- |
| camera_fps | Effective camera read FPS from live CCTV validation. |
| detections_per_frame | Detector load from YOLO/RT-DETR/D-FINE. |
| filter_cycles | Deterministic event-filter cost. |
| review_latency_ms | VLM or human-review latency per candidate. |

Expected output:

- candidate events per minute
- dropped/queued candidates
- VLM review load reduction
- minimum edge filter throughput required for deployment

This makes the branch useful even without physical NPU hardware. It simulates
the deployment pressure that a Snapdragon/NPU implementation would need to
handle later.
