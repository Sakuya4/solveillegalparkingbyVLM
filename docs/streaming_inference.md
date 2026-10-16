# Causal Streaming Inference

## Architecture

The native camera workflow evaluates temporal windows while acquisition remains
active. A dedicated decoder drains the persistent stream into a one-frame
mailbox. Inference consumes the latest frame at a target cadence of 7.5 Hz;
overwritten mailbox entries are counted explicitly. This bounds acquisition
backlog instead of accumulating stale frames during a slow inference step.

The causal processor retains one previous feature image and 15 motion-feature
steps. A separate 16-frame privacy-treated context ring supplies evidence.
Each complete window produces an ONNX probability immediately; the selected
one- or two-window persistence gate determines a new review episode. The event
notification is published without waiting for post-event imagery or a VLM.
Post-context evidence is finalized after 15 further sampled frames. If the
observation ends first, the final available image is used and truncation is
recorded. Numerical logs and recordings are stored on disk for the bounded
observation interval.

The camera worker uses two CPU threads for model and image processing. The
decoder is an I/O thread; its mailbox contains at most one pending image. The
native window runs independently through QProcess and polls the current
preview and event manifest. CPU and explicit QNN execution follow the same
provider-selection contract as batch analysis.

## Replay Verification

On the included reference video, sequential CPU inference and batch inference
select the same 225 frames and 14 temporal windows. The maximum probability
difference is 1.79e-7. Persistence flags agree on every window and the detected
event refers to the same source frame. The event decision becomes available
before its three-frame evidence package is finalized.

[Verified replay artifact](results/streaming_reference_parity.json)

The regression test repeats real YOLO/ByteTrack and ONNX inference, rather
than substituting stored predictions. Additional unit tests verify warm-up,
window boundaries, persistence, monotonic source clocks, bounded feature
storage, and mailbox overwrites.

## Public-Camera Observation

A 30-second native session on Taichung camera C000129 decoded 296 frames and
processed 202 frames, yielding 6.74 sampled frames/s over the valid observation
span. The mailbox overwrote 27 pending entries. Median and 95th-percentile
decoder-arrival-to-window-decision times were 98 and 135 ms, respectively;
per-frame processing medians and 95th percentiles were 93 and 111 ms.

Two review episodes were emitted. Their event labels remain unassigned; the
session contributes transport and runtime measurements, separately from the
independently reviewed normal-exposure cohort. Timing begins at local decoder
observation, not at a remote camera sensor. Operator-display latency and
source-network transport require separate instrumentation. Public numeric
artifacts exclude the camera's image and video payloads.

[Camera session measurements](results/native_cctv_observations.json)

## Saved-Alert Multimodal Review

Both emitted episodes were subsequently reviewed from their saved three-frame
packages using pinned Qwen2.5-VL-3B, explicit generation contract version 2,
and a 256-token limit. No detector or temporal inference was rerun. Review
service took 9.778 and 7.758 s; the complete job, including model load, has its
own measured duration in the artifact. Both responses were schema-valid,
assigned self-reported confidence 0.85, declared evidence gaps, and requested
operator review. The complete-evidence gate therefore assigns both to
`incomplete_visual_evidence`.

These are triage outcomes, not adjudicated incident labels. In particular,
model descriptions of turning behavior or signal violations are not accepted
as verified scene facts. Independent event labels remain unassigned, and
reviewed normal exposure increases by zero camera-hours. The second event's
post-context truncation remains in its original event record. Public results
contain hashes, model responses, and timing, without camera imagery.

[Saved-alert review artifact](results/streaming_alert_vlm_review.json)

## Spatial Preprocessing

The preserved desktop reference uses 640-pixel feature frames. Its 500-clip
training extraction used a 320-pixel width limit without upscaling. The
streaming processor exposes the latter as an explicit controlled setting;
display width remains independent of feature width. A matched event-level
audit evaluates the resulting operating-distribution change without replacing
the original model, threshold, or reported reference protocol. Cadence-consistent
training remains a subsequent methodological step; train-only event calibration
is now published under its own fixed protocol. The [paired audit](spatial_preprocessing_study.md) reports resolution
and decision-hop sensitivity explicitly. Benchmark cards identify the original
IID hop-five protocol; desktop replay and streaming workers use 15-step hops.

## Reproduction

```powershell
python scripts/run_streaming_incident.py --input docs/assets/incident_input.mp4 --output-dir outputs/streaming/reference
python scripts/run_desktop.py --capture-page "https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129" --capture-duration 30 --continuous
python -m pytest tests/test_incident_streaming.py tests/test_streaming_integration.py tests/test_desktop_app.py
```

The camera dialog also supports capture followed by batch analysis. Streaming
exports clean evidence for later VLM service. The native saved-event review
action submits that package in a separate, GPU-guarded worker without rerunning
the detector. It writes a new report and retains the original source report;
review time does not replace immediate event-notification latency. Local-video
analysis also retains the integrated Qwen-review option.

New observations also save a full clean sampled-video track with its own hash.
The native observation-annotation dialog uses this unmarked track and records
reviewer identity, observation label and explicit review attestations. The
[normal-exposure protocol](observation_review_protocol.md) excludes acquisition
gaps, uncertain/model-informed/incomplete records, duplicate exposure and
conflicting labels before computing normal-camera rates.

Transient source failures use three bounded consecutive reconnect attempts and
interruptible backoff. Source-clock gaps above one second reset the tracker,
motion buffer, window-grid phase and persistence gate. Pending evidence is
truncated at the last pre-gap image rather than spanning the interruption.
Acquisition reports retain failure/retry/connection counters and context resets.
