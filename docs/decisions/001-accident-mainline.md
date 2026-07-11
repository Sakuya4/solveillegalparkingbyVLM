# ADR-001: Use CCTV Accident Detection As The Next Mainline

## Status

Accepted

## Date

2026-07-12

## Context

The red-line parking workflow is complete, but another parking-only dataset
would repeat the same contribution. The continuation needs public training data,
temporal AI, edge deployment, embedded logic, and hardware simulation.

## Decision

Use ACCIDENT as the primary training benchmark and build an edge-first traffic
incident detector. Use Taiwan government A1/A2, violation, VD, and CCTV data for
deployment priority and external operational validation.

Keep TUMTraf-A/V2X as a later trajectory and multi-sensor validation source.

## Why

- ACCIDENT matches the fixed surveillance-camera deployment assumption.
- Its labels support temporal localization, spatial localization, collision
  classification, and geographic OOD evaluation.
- The pipeline can reuse the existing event writer, SAM evidence, VLM review,
  privacy handling, and edge input adapters.
- Track-derived motion features provide a useful boundary between Python AI,
  Verilog trigger logic, and SystemC NPU scheduling.

## Alternatives

- FishEye8K/FE-DETRAC: useful for vehicle detection, but insufficient as the new
  project claim because they mainly measure objects rather than incidents.
- DoTA: large and trainable, but dominated by dashcam viewpoints rather than
  fixed government CCTV.
- TUMTraf-A as the only dataset: excellent roadside labels, but too few real
  accident sequences for the first main training benchmark.

## Consequences

- The main metric changes from detector mAP to event quality and alert delay.
- Full ACCIDENT redistribution is not allowed by this repository; local data
  stays ignored and upstream license terms remain binding.
- A1/A2 reduction remains a future field-study outcome, not an immediate model
  result.
