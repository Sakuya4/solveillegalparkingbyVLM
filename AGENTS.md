# Project Agent Guide

This repository extends red-line parking detection into an edge-first traffic
incident detection system. Agents must preserve deployable metrics, reproducible
experiments, and privacy-safe evidence.

## Active Skill Set

- Coding and architecture: use stable module contracts and keep edge fallbacks
  independent from heavyweight optional models.
- Data and notebooks: keep raw data, generated features, weights, and reports out
  of Git; record deterministic seeds and dataset split contracts.
- Testing and debugging: use RED-GREEN-REFACTOR for behavior changes and run the
  full test suite before push.
- Documentation and knowledge: document measured results, limitations, data
  licenses, and the distinction between detection and anticipation.
- GitHub and collaboration: make atomic commits on the current task branch and
  push only after review and verification.

## Research Integrity

1. Runtime features must not use ACCIDENT annotations such as accident bounding
   boxes, accident type, quality, region, or timestamps beyond window labeling.
2. Annotation-derived ROI results are oracle upper bounds. Training requires an
   explicit `--allow-oracle-roi` opt-in and reports must label the result oracle.
3. Candidate ROI generation may use detector boxes, tracker state, frame motion,
   camera calibration, and optional SAM2 masks available at inference time.
4. Ground-truth accident boxes may evaluate proposal IoU or recall, but those
   diagnostics must not be written into model feature columns.
5. Centered incident windows include post-event frames. Call current results
   event detection or confirmation, never pre-crash prediction.
6. Report IID and geographic OOD results separately. Do not tune on test data.
7. Do not claim false alarms per camera-hour until normal-only CCTV footage has
   been evaluated for a measured duration.

## Edge Contract

- The default path must run without SAM2, a VLM, cloud access, or accident
  annotations.
- SAM2 is an optional proposal/mask refiner with a deterministic tracker-motion
  fallback.
- Missing detections or masks must produce explicit availability features and a
  conservative fallback, not crash a stream.
- Keep evidence images privacy-safe by masking plates and location/time overlays
  before publication.

## Delivery Loop

1. State assumptions and define the smallest public contract.
2. Add a failing test that demonstrates the intended behavior.
3. Implement the minimum deployable slice and make the test pass.
4. Run focused tests, then `python -m pytest -q` and Python compilation.
5. Review correctness, readability, architecture, security, and performance.
6. Commit the verified slice with a descriptive conventional commit message.
7. Update `docs/todo.md`, `docs/project_status.md`, and measured-result documents.

## Standard Commands

```powershell
& 'C:\Program Files\Python311\python.exe' -m pytest -q
& 'C:\Program Files\Python311\python.exe' -m compileall -q Application/DebugVersion/src scripts tests
git diff --check
git status -sb
```

`spec.md`, `data/raw/`, `data/processed/`, `models/`, and `outputs/` remain local
and untracked unless the project owner explicitly changes that policy.
