# Reference Media Attribution

`incident_input.mp4` preserves the complete 30-second
`real_videos/ig4k6LEEMy0_00.mp4` ACCIDENT clip for native desktop analysis.
Source: https://github.com/accidentbench/ACCIDENT. License: CC BY-NC-SA 4.0.
The complete temporal context supports reproducible persistence-gated inference.

`accident_reference.mp4` and `accident_reference.gif` are six-second derived
clips from `real_videos/ig4k6LEEMy0_00.mp4` in the ACCIDENT benchmark.

- Source project: https://github.com/accidentbench/ACCIDENT
- Event metadata: t-bone, Arkansas, daytime, fixed traffic camera
- Transformation: trimmed, resized, frame-rate reduced, and audio removed
- License: CC BY-NC-SA 4.0

The media files are included only as a non-commercial research and project
demonstration. They are not outputs predicted by this repository.

`candidate_incident_model_output.mp4` and
`candidate_incident_model_output.gif` use the same ACCIDENT source clip and
license. Unlike the reference files, these are repository model outputs:

- Inference: annotation-free YOLO/ByteTrack plus motion candidate ROI and the
  calibrated candidate-only causal TCN.
- Transformation: 4-16 second excerpt, candidate overlay, model score/status,
  audio removal, resize, and heuristic lower-vehicle privacy blur.
- Ground-truth accident boxes are not model inputs or overlays.

The current output media use availability-aligned window scores: a temporal
decision first appears after its input window completes.

`desktop_workbench.png`, `desktop_evidence.png`, and `desktop_walkthrough.mp4`
capture the native PySide6 application with a completed reference analysis.
The walkthrough includes actual output-video playback, event evidence, model
tables, government coverage, and runtime diagnostics. It is reproduced with
`scripts/record_desktop_demo.py` and H.264 transcoding; no additional inference
is performed during recording. Source ACCIDENT media retain CC BY-NC-SA 4.0.

`desktop_model_comparison.png` and `desktop_event_retention.png` render native
result tables from checked-in numerical studies. `full_cohort_model_curves.png`
and `.pdf` are generated from verified full-cohort test predictions using
`scripts/plot_full_model_comparison.py`. Camera observation payloads are retained
locally; public camera-session artifacts contain numerical measurements only.

## Continuous Reference Demonstration (16 October 2026)

`incident_demo.mp4` is the complete 30-second reference clip with causal
streaming output, vehicle privacy blur, motion proposals and temporal scores.
`incident_demo_preview.jpg` is a frame from that output; the corresponding GIF
shows source seconds 6--14 for inline README playback. `desktop_walkthrough.mp4`
records the corresponding native incident, evidence, corrected-decoding model,
government-allocation and event-retention views. These recordings are
derivatives, not additional independent benchmark samples.

- Dataset and creator: [ACCIDENT / accidentbench](https://github.com/accidentbench/ACCIDENT).
- Distribution: [ACCIDENT on Kaggle](https://www.kaggle.com/datasets/picekl/accident).
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- License verification: the official [Kaggle dataset metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/picekl/accident)
  returned `licenseName: CC BY-NC-SA 4.0` on 16 October 2026.
- Source credit: Arkansas Department of Transportation, as visible in the clip.
- Transformations: sampled/blurred frames, model annotations, silent H.264
  transcoding, and native application capture. Source credit remains visible.
- `incident_demo.json` preserves measured endpoints, source/model/media hashes,
  pinned VLM revision and generation settings, and the actual disagreement.
- These media retain the non-commercial share-alike license independently of
  the repository's software license.
