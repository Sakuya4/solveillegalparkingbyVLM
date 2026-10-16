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

## Continuous Extension

Continuous inference now uses the same proposal features, calibrated ONNX
model and persistence rule. A one-frame decoder mailbox and bounded temporal
ring expose decisions at window completion. Post-event evidence is deferred
until context arrives; it does not postpone event notification. Real-detector
CPU replay verifies numerical and decision parity with the existing pipeline.
The [streaming study](../streaming_inference.md) records measured processing
and decoder-arrival-to-decision latency separately from operator-display time.

The hardware comparison also replays confirmed TCN event requests from the
continuous incident evaluation, retaining annotation-matched event identities
through admission and service completion. Raw proposal workloads remain
separate stress tests. Subsequent work extends reviewed camera exposure,
train-only event calibration, and heterogeneous review-service timing.
