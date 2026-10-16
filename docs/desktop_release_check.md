# Native Desktop Installation Verification

## Environment and Scope

A fresh Git clone and isolated Python 3.11.7 virtual environment were tested
on Windows build 26200. The documented application and desktop requirements
were installed without access to the developer environment's site packages.
The detector was downloaded from the official Ultralytics asset distribution;
the temporal ONNX model and reference clip were taken from the checkout.
`pip check` reported no broken requirements.

The resolved environment used Torch 2.14.1 (CPU), Ultralytics 8.4.178,
OpenCV 5.0.0.93, PySide6 6.12.0, ONNX Runtime 1.31.0, NumPy 2.4.6 and
lap 0.5.13. This installation check uses a newer dependency resolution than the
research runs; it does not replace their recorded environments or benchmarks.

## Verification Results

All five native workspaces and the eighth model-diagnostic tab rendered.
Reference analysis completed with 225 sampled frames, 14 temporal windows
and one event at 10.041667 s. The CPU processing run took 37.891 s; this is
a single functional installation check, not a throughput estimate.

The post-fix portable regression suite passed **40 tests**, with one expected
skip because the developer-only Torch checkpoint is not distributed in the
portable clone. The skip concerns trained Torch/ONNX comparison; native ONNX
analysis, Windows GUI launch, live local-camera acquisition and reconnection
were exercised. A separate confirmation run returned process exit code zero.

Two installation issues were identified and addressed:

- A deeply nested test virtual environment exceeded Windows path limits in
  Qt wheel resources. Installation succeeded in a shorter checkout; the native
  installation instructions now recommend a short path.
- ByteTrack's assignment backend is now an explicit dependency. Lazy detector
  initialization also exhausted the three-second streaming fixture. Camera
  workers now warm both inference backends before observation, reset synthetic
  tracks, and report initialization time separately. File replay is unchanged.

The original failing regression record is retained alongside the passing
confirmation. No research model, threshold, cohort or published measurement
was replaced. Optional local VLM service and physical QNN execution remain
separate validation paths.

## Provenance

Reference analysis and workspace captures used `c62044e`; the passing portable
suite used `11d40fd`. Raw installation reports, JUnit records and screenshots
are retained under `outputs/release_verification/`.

| Artifact | SHA256 |
| --- | --- |
| ONNX model | `eb4b36d6976e887c6fdd7262e445c74af8fec759d0cfb402bd7a0cf8f7dff2c3` |
| Reference video | `1f2a29ef53659e8a1aa349a3d5bd02ccf93f1ffedc58d7e5e656d19d586c1633` |
| Detector | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Installation report | `99f72f6a9eb2e6b5cfe5be3930fafe97049483070831f6d243280aca84501a97` |
| Tracker installation report | `ec280b8fd2971106b1af5a74972a8f873e90bb3044c7b86ee68a5b1541fc1596` |
| Initial failing JUnit record | `7b2cf5aa9cc9d7c25f4ef49f297a77dfc0a2aac1b5b1a63534831888a4ca7524` |
| Passing confirmation JUnit record | `ee2c877dbbf1565d60167e448e4685ba4d8f305e5152d734dd994aff40e153de` |
| Reference analysis report | `f86fabc00d3fb14034b6e3a7432c5ae0307e501a23543740de8475f43b6bcdd7` |

## Main Publication Gate

On 17 October 2026, the complete developer regression passed **509 tests**
with process exit code zero, including the frozen-training inputs and new
public-camera trace verification. The JUnit record is retained at
`outputs/research_runs/main_release_20261017_tests.xml`. The separate
[120-second public-camera experiment](live_cctv_runtime.md) also completed,
with native playback verified. Research reference weights and operating
defaults remain unchanged.
