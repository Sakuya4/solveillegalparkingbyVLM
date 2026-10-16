# Spatial Preprocessing and Event-Decision Sensitivity

## Protocol

The 500-clip feature report records a 320-pixel width limit, frame stride two,
detector input size 320, and confidence 0.15. The preserved continuous-event
reference uses 640-pixel frames. Farneback flow retains pixel-scale information,
so spatial preprocessing is part of the model's operating contract.

The controlled audit processes 100 fixed IID and 100 fixed geographic test
clips with a 320-pixel source-downsampled feature view, without upscaling.
Display width remains 640. Model weights, calibrated thresholds, sampled source
frames, detector input size, event tolerance, and sampling seed are unchanged.
Tracking and motion both use the selected feature view; the intervention does
not isolate optical flow from detector-preprocessing effects.

The matched grid uses 15-step windows and 15-step hops. Original IID reports
have five-step hops, so their existing probabilities at starts 0, 15, 30, ...
are selected and persistence is recomputed. Geographic reports already use
15-step hops. The original IID hop-five recall values remain separate.

## Results

| Split | Persistence | Recall at 640 | Recall at 320 | Paired difference (95% interval) |
| --- | ---: | ---: | ---: | --- |
| IID | 1 | 0.48 | 0.50 | +0.02 (-0.09, +0.13) |
| IID | 2 | 0.20 | 0.08 | -0.12 (-0.19, -0.05) |
| Geographic | 1 | 0.63 | 0.47 | -0.16 (-0.26, -0.06) |
| Geographic | 2 | 0.30 | 0.20 | -0.10 (-0.19, 0.00) |

Intervals use 2,000 paired source-clip resamples, seed 42, with fixed models
and thresholds. The artifact also reports gains/losses, off-target episodes,
conditional delays, source hashes, and training preprocessing. Geographic
responsive replay gains six and loses 22 matched events at 320 pixels.

Resolution alignment alone does not establish improved event detection.
Its interaction with persistence is substantial. Changing only the IID
640-pixel hop from five to 15 reduces responsive recall from 0.75 to 0.48
and conservative recall from 0.44 to 0.20. These are different inference
protocols, not interchangeable estimates of one operating configuration.

## Implementation Decision

Preserve original model binaries, thresholds, and legacy results. Expose feature
resolution independently of display resolution and label decision hops in
research tables. Next, calibrate continuous-event operating points using
held-out training clips under a fixed spatial/cadence/hop contract before
evaluating the test set. Reviewed normal-camera exposure remains a separate
endpoint from off-target episodes in accident videos.

[Numerical artifact](results/event_resolution_audit.json) ·
[Recall, delay, and off-target table](results/event_resolution_audit.csv)

```powershell
python scripts/run_overnight_research.py --phase resolution --deadline 2026-10-17T04:00:00+00:00
python scripts/publish_resolution_audit.py
python -m pytest tests/test_incident_video_views.py tests/test_resolution_audit.py
```
