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
