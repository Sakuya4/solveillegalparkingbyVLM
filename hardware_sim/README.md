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
  systemc/
    event_pipeline_sim.cpp
  python_golden/
    dwell_fsm_model.py
  test_vectors/
    dwell_fsm_vectors.json
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

## Current Verification

The repository does not require SystemC or Verilator to run the default tests.
Instead, pytest checks the JSON vectors against the Python golden model first:

```powershell
python -m pytest tests\test_hardware_sim_vectors.py
```

Later, the same vectors can be reused by a Verilog testbench, Verilator, or
cocotb.
