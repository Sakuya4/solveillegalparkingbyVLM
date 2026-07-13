# Hardware Simulation Notes

This folder implements the hardware-aware part of the edge traffic-event
project. The goal is not to move every detector or VLM into RTL. The goal is to show which
parts of the edge pipeline are simple, deterministic, and suitable for
hardware-style simulation.

## Scope

The hardware side focuses on event filtering before expensive AI review:

- red-pixel and bbox-band overlap counting
- dwell-time accumulation
- event candidate FSM
- accident motion-spike triggering
- shared NPU and review queue capacity simulation

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
    motion_trigger.v
  testbench/
    dwell_fsm_tb.v
    bbox_overlap_counter_tb.v
    motion_trigger_tb.v
  systemc/
    event_pipeline_sim.cpp
    npu_queue_sim.cpp
  python_golden/
    dwell_fsm_model.py
    overlap_counter_model.py
    motion_trigger_model.py
    npu_queue_model.py
  test_vectors/
    dwell_fsm_vectors.json
    overlap_counter_vectors.json
    motion_trigger_vectors.json
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
- `motion_trigger.v` compares accident-ROI motion with global camera motion and
  emits one duplicate-suppressed incident pulse.
- `npu_queue_model.py` estimates multi-camera frame drops, latency, utilization,
  and review backpressure before target hardware is available.

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

The Python queue model is the executable transaction-level golden model.
`npu_queue_sim.cpp` implements the same camera, NPU, and review queues as a
timed SystemC simulation.

After installing the official Accellera SystemC package, build with:

```powershell
cmake -S hardware_sim\systemc -B hardware_sim\build-systemc -DCMAKE_PREFIX_PATH=<systemc-install>
cmake --build hardware_sim\build-systemc --config Release
```

The CMake target follows the official `SystemC::systemc` package interface.

Run a measured-latency profile with:

```powershell
python scripts\run_edge_queue_simulation.py --camera-count 4 --camera-fps 15 --detector-latency-ms 29 --temporal-latency-ms 3 --sweep-max-cameras 6
```

Using the current RT-DETR measurement (29 ms) plus a 3 ms temporal stage, the
simulation supports two 15 FPS cameras without frame loss. Three cameras drop
about 30.3% of frames and four drop about 47.8%. These are scheduling-model
results, not measurements from a physical Snapdragon device.

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

## SystemC Scope And Next Extension

The current SystemC slice models camera arrivals, a bounded shared-NPU FIFO,
detector plus temporal service time, candidate generation, and a bounded review
FIFO. It reports processed, pending, and dropped work at the simulation
boundary, so frame and candidate accounting can be checked.

Current configurable inputs are:

| Input | Meaning |
| --- | --- |
| camera_count | Number of camera streams sharing the NPU. |
| camera_fps | Effective camera read FPS from live CCTV validation. |
| detector_latency_ms | Measured detector service time per frame. |
| temporal_latency_ms | Incident temporal-stage service time. |
| npu_queue_capacity | Waiting-frame capacity before dropping. |
| candidate_stride | Deterministic candidate frequency for the smoke model. |
| review_latency_ms | VLM or human-review latency per candidate. |
| review_queue_capacity | Waiting-candidate capacity before dropping. |

Current output includes:

- candidate events per minute
- dropped/queued candidates
- VLM review load reduction
- minimum edge filter throughput required for deployment

This makes the branch useful even without physical NPU hardware. It simulates
the deployment pressure that a Snapdragon/NPU implementation would need to
handle later. The next extension is to replace constant service times and
candidate stride with per-frame traces from the real detector/incident models,
then add calibrated CPU/VLM stages and power proxies.
