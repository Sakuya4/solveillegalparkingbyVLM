# Continuous Presentation-Time Event Evaluation

## Objective

Determine whether the matched training policies transfer to continuous
incident recognition under a common physical-time input and alert budget.
All eight candidate/global, source-frame/fixed-time, IID/geographic TCNs
remain in the study regardless of their preceding window-level outcomes.
The original 500-clip desktop binaries and operating defaults are unchanged.

## Frozen Contracts

The parent study defines 94 IID and 83 geographic training-holdout clips,
and test intersections of 88 and 90 clips. Membership is fixed before this
replay, with no replacements. Each split calibrates only on its own training
holdout. The machine-readable event protocol binds source/video hashes,
annotation anchors, model hashes, code hashes, decoder/runtime versions,
and the parent paired-cohort specification before event inference starts.

All models use the same globally phased 7.5-Hz target grid, normalized to
the first presentation timestamp. Each 16-image context yields 15 motion
steps; successive contexts begin 15 target slots apart. Inputs retain a
320-pixel width limit, detector image size 320 and confidence 0.15. Tracking
resets within each context, matching the training extractor. This is a
rolling-context inference contract, distinct from persistent per-camera
tracking in the existing native reference pipeline.

Packet timestamps are obtained with the pinned FFprobe executable, using
its [structured packet/stream output](https://ffmpeg.org/ffprobe.html).
The selected full-clip grid is decoded sequentially from the source origin;
every selected frame must agree with frame identity and presentation time
within 0.1 ms. The supplied annotation frame maps to its previously verified
PTS. Annotation time is not provided to detector or feature extraction.

Duplicate-frame or excessive-lateness contexts produce no prediction.
Persistence resets across any missing hop. No frames are interpolated,
no invalid contexts are scored as normal, and no clip is replaced. A clip
without any valid context remains in the event denominator as a miss;
source integrity or decoder verification failures stop publication for
investigation rather than silently removing the clip.

## Calibration and Endpoints

Two consecutive valid positive contexts emit an episode at the second
context's actual last-frame PTS. Sustained positives are one episode;
a negative context or missing hop terminates its persistence state.
Events match episode starts within [-1, +3] s of the supplied annotation
frame's PTS. Delay is signed trigger PTS minus annotation PTS and is
summarized among detected clips, with misses reported separately.

The threshold grid and selection rule are inherited from the parent
pre-fitting plan: maximize training-holdout event recall subject to at
most 0.5 off-target episodes per accident clip; ties favor fewer off-target
episodes, then a higher threshold. Infeasible settings remain unselected.
The existing window-calibrated threshold is also included in the grid.
All eight selections must be recorded and hash-bound before test replay.
At publication, they must reproduce exactly from their calibration inputs.

Report all eight event recalls, delay distributions, rejected contexts,
off-target demand and test budget exceedance. Four fixed-minus-source-frame
recall contrasts use 2,000 paired source-clip bootstrap resamples, seed 42.
Intervals are pointwise and conditional on the fitted models and selected
thresholds; neither refitting nor selection uncertainty is resampled.
Off-target demand in accident clips is not a reviewed-normal camera-hour
false-alarm estimate. Source-time delay is distinct from compute latency.

## Execution

```powershell
python scripts/prepare_presentation_event_protocol.py
python scripts/run_overnight_research.py --phase pts_events --deadline 2026-10-17T04:00:00+00:00
python -m pytest tests/test_presentation_events.py
```

The separate queue replays calibration inputs, freezes operating thresholds,
replays test inputs, and publishes verified comparison JSON/CSV. It runs only
after the cadence training queue has completed, under the same exclusive
GPU lock and two-thread worker limit. Per-clip artifacts support resumption;
the runner terminates the complete worker process tree at its deadline.

CPU verification covers source-clock causality, missing-hop resets, invalid
and empty contexts, event-boundary matching, model-specific calibration
membership, cohort completeness and changed artifact hashes. A real-detector
CPU check on the training-only clip `xKT7khciy-c_2_00.mp4` yields one finite
15-by-14 feature tensor and a context ending at PTS 2.035367 s. This check
does not contribute to model fitting or reported test performance.

The frozen event protocol SHA256 is
`f8725a5e20f5d6ac04c55f15b37e88f8e2ae8555eb3599f8ed903768ab77b131`.
Its combined calibration/test source union contains 318 distinct clips.
A separate end-to-end CPU smoke replay loads all eight actual checkpoints
and records one valid context on a calibration clip; its smoke flag bars it
from complete-cohort publication. The complete 408-test regression suite
passes before starting the new GPU queue.

[Frozen model, source and implementation hashes](results/presentation_event_protocol.json).
