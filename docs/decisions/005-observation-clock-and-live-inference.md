# ADR 005: Preserve Observation Time Through Camera Ingestion

## Decision

Use persistent MJPEG connections for compatible public-camera endpoints and
retain an observation timestamp for every saved frame. Capture transport,
sampling, encoding, model processing, and event availability have distinct
time measurements. Mean-cadence playback must not replace the original event
clock in exported research records.

## Implemented Workflow

The native camera action provides a bounded observation, preview, cooperative
cancellation, and optional analysis after acquisition. Frames are spooled as
lossless PNG images on disk, keeping image memory independent of observation
duration. Scratch storage is confined to the job directory and cleaned on
normal completion or cooperative cancellation. Provenance strips URL
credentials and non-camera query values. Frame hashes and video hashes bind
timing metadata to the corresponding observation.

The incident worker verifies that binding, uses recorded timestamps for event
and evidence records, and stores output-frame indices for accurate event
navigation. Public-camera observations have a separate annotation status from
the normal-exposure evaluation cohort.

## Next Implementation Gate

Add continuous inference using the same proposal features, calibrated ONNX
model, persistence rule, and observation clock. Use a bounded frame/feature
ring, expose decisions as soon as their windows complete, and capture delayed
post-event evidence without blocking the source. Verify replay parity against
the existing video pipeline and record read time, processing time, queue
pressure, and decision availability separately.

The hardware comparison should additionally replay confirmed TCN event
requests from continuous incident evaluation. Retain raw proposal-level
workloads as stress tests rather than treating them as the final review demand.
