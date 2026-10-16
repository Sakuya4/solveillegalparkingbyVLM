# Research and Desktop Implementation Plan

Execution window: 16 October 2026 to 17 October 2026, 12:00 Asia/Taipei.

## Research Objective

Develop a native desktop system that connects temporal traffic-incident
recognition, evidence review, constrained edge inference, and public-sector
camera prioritization. The primary research question is whether historical
crash information can improve monitoring coverage under a fixed camera budget
while retaining a reproducible event-level perception evaluation.

The original red-line parking case remains the project's provenance and first
rule-based event application. ACCIDENT supports temporal model development;
Taiwan government crash and camera records support deployment research.

## Phase 1: Reproducible Research and Native Application

- [x] Audit repository, model artifacts, dependencies, and government sources.
- [x] Establish bounded hourly continuation through the execution deadline.
- [x] Record primary literature and connect each method to an implementation.
- [x] Implement risk-weighted maximum coverage with temporal holdout and
  matched-budget random, count-based, and independent-ranking baselines.
- [x] Record source hashes, split dates, weight/radius sensitivity, and results.
- [x] Build a PySide6 laptop application for video playback, incident analysis,
  evidence inspection, model comparisons, government risk, and job control.
- [x] Support explicit ONNX execution-provider selection and record the provider
  actually used, including CPU and QNN when available.
- [x] Verify desktop interactions, cancellation, artifacts, and inference parity.

Acceptance: a single Python launcher opens a native window; a local video can
be analyzed without blocking the UI; exported event results retain provenance;
government coverage experiments reproduce from downloaded source records.

## Phase 2: Evaluation, Hardware Integration, and Scholarly Presentation

- [x] Complete geographic event-level evaluation using the geographic model.
- [x] Expand clean-blind VLM evaluation with annotation-defined paired event labels.
- [x] Compare review-routing strategies under a fixed latency/queue budget.
- [x] Connect measured traces and desktop runtime diagnostics to the existing
  Verilog/SystemC workflow and Qualcomm deployment preparation.
- [x] Present bootstrap confidence intervals and operating-point tradeoffs.
- [x] Rewrite the main README and current research report in academic English;
  preserve team attribution, original images, licensing, and historical context.
- [x] Produce desktop screenshots, a reproducible demonstration, and a compact
  manuscript outline with research questions and evidence-backed contributions.
- [x] Run appropriate regression tests and review changes before checkpointing.

Acceptance: every reported number links to an experiment artifact and protocol;
model comparison separates event/window endpoints; the desktop demonstration
shows input, model output, evidence, and deployment implications end to end.

## Methodological Decisions

1. Use temporal holdout for government deployment experiments: select cameras
   using January--September 2024 and evaluate October--December 2024. Keep the
   camera inventory fixed and document its publication timestamp separately.
2. Interpret proximity as planning coverage. Camera orientation and field of
   view are separate measurements required for operational coverage.
3. Treat A1 severity weighting and coverage radius as explicit experimental
   factors. Report sensitivity instead of selecting the best test setting.
4. Preserve measured performance and describe study scope in Methods and
   Discussion. State proposed contributions as hypotheses until evaluated.
5. Prefer peer-reviewed journal methods and primary implementation sources.
   Quartiles require a verified index, year, and subject category.
6. Run at most one GPU workload at a time. Bound CPU thread counts and keep
   resumable experiment outputs. Never overwrite unrelated user changes.

## Continuation Order

Resume unfinished Phase 1 tasks first. Inspect active jobs before launching new
work. After each substantive experiment or feature, update this checklist and
`docs/todo.md`, record actual validation, and create a focused checkpoint.
At the deadline, report delivered artifacts and remaining external requirements.

## Current Research Queue

1. Completed: 100-clip geographic event batch and both persistence profiles.
2. Completed: matched-request Qwen/LLaVA comparison and eight-frame Qwen
   evidence-density ablation on 20 paired sequences. A runtime audit identifies
   that requested sampling parameters were not forwarded to generation.
   Completed separate `decoding` replications with explicit `generate_kwargs`
   and pinned model revisions; all legacy results remain unchanged.
3. Audit temporal sampling: source FPS spans 3.92--49.70, while the existing
   frame-stride training rate has median 7.49 steps/s and the deployment target
   is 7.5. Test cadence-consistent extraction as a controlled ablation.
4. Completed: full-cohort candidate/global TCN fits and prediction-verified
   publication. All 2,027 clips were processed; 2,021 meet window eligibility.
   The 500-clip desktop model remains the reference version.
5. Completed: five-policy routing comparison with exact SystemC/Python parity.
   Completed extension: TCN-gated event retention on IID/geographic 100-clip
   batches with 24 Python/SystemC parity checks.
6. Completed: native camera preview, bounded persistent observations, cooperative
   cancellation, and automatic subsequent analysis with source-clock provenance.
7. Completed: continuous native streaming inference with a one-frame mailbox,
   bounded frame/feature buffers, window-end decisions and real-detector replay
   parity. Taichung session measures 98/135 ms median/p95 core decision latency.
8. Completed: TCN-gated demand and annotation-matched event retention through
   measured-service hardware replay; proposal workloads remain stress tests.
9. Completed: a matched 100-clip-per-split spatial preprocessing audit.
   Training uses a 320-pixel width limit without upscaling; legacy continuous
   event replay uses 640-pixel frames. Farneback flow retains pixel units.
   Paired recall/delay/off-target changes use a common 15-step hop. IID legacy
   hop-five probabilities are subset and re-gated, not compared directly to
   hop-15 inference. No deployment default is selected by test performance.
   `--phase resolution` records progress
   in `outputs/research_runs/resolution_status.json`; inspect all status files.
10. Completed: train-only continuous-event calibration with fixed spatial/cadence/hop
    contracts and all 24 operating settings. Five of twelve IID and nine of
    twelve geographic test settings exceed their calibration alert budget.
    The reference deployment remains unchanged.
    The [frozen calibration protocol](continuous_event_calibration.md) uses
    101 IID and 90 geographic model-fitting holdout clips. The separate
    `--phase calibration` queue records `calibration_status.json`; selection
    must complete before each corresponding 100-clip test replay starts.
11. Completed: evidence-aware VLM disposition, including all declared evidence
    gaps and review requests, and 176 variable-service scenarios with 32
    exact SystemC/Python counter comparisons.
12. Completed: saved-event native VLM review without detector reruns, exclusive
    GPU OS locks, and actual research-child PID checks. Both saved Taichung
    alerts receive incomplete-evidence dispositions with explicit generation
    forwarding; independent labels remain unassigned.
13. Completed: `decoding_status.json` replications and verified paired reporting.
    Qwen three-frame correctness is 0.60 versus LLaVA 0.30; their paired
    difference is +0.30 [0.10, 0.50]. Qwen eight-frame correctness is 0.55.
    Native comparison tables preserve the same endpoints and source-clip units.
    Next extend independently reviewed normal exposure. Do not restart any
    completed queue.
    `scripts/publish_vlm_forwarded_comparison.py` verifies every new input/result
    hash and recomputes metrics before paired source-clip bootstrap reporting.
    Publisher verification and paired source-clip tests are included in the
    full regression suite.
14. Implemented: clean full-observation streaming recording, native reviewer
    annotation and hash-bound normal-exposure evaluation. The one-second
    acquisition-gap contract is fixed before collecting additional observations.
    Labels require reviewer attribution; uncertain/incomplete/model-informed
    records and conflicting labels are excluded. No additional normal camera-
    hours are claimed from unassigned observations.
15. Transport validation: an initial 120-second request ended after 30.303 s
    with a source-read failure. Preserve this record. Checkpoint `ecd3b46`
    adds bounded reconnects and source-gap temporal resets before a separately
    recorded 120-second follow-up. Publish actual completion and gap metrics;
    do not substitute requested duration for observed exposure.
    Completed: the follow-up reaches duration limit, processes 861 frames,
    and records four connections/retries and four unassigned alert episodes.
    The 2.216-second sampled-frame gap is excluded from acquisition coverage
    and resets temporal context. No additional normal labels are inferred.
16. Next: preserve the review-ready camera packets for independent labels and
    develop a predeclared cadence-consistent training ablation. Hold physical
    sampling duration, fit/calibration splits and test contracts explicit;
    preserve current model binaries and do not tune defaults on test scores.

The sequential runner is `scripts/run_overnight_research.py`. Its active
process and child PID, current step, log path, and completed checkpoints are
recorded atomically in `outputs/research_runs/overnight_status.json`. Inspect
that file before starting another GPU workload. The queue starts with controlled
requested-decoding VLM replications, then a matched eight-frame evidence ablation,
full-cohort candidate extraction, and four fixed-protocol temporal fits.
Current 500-clip desktop model bundles remain the reference deployment version.
The second queue uses `--phase representations` and records its live state in
`outputs/research_runs/representations_status.json`. It extracts full-cohort
VideoMAE features, fits frozen heads, and fine-tunes the last encoder block for
both splits. Inspect both state files and their actual child processes before
any GPU job. The runner rejects a live child belonging to the other phase.

Completed checkpoints: `af20fb2` (government temporal allocation), `a2dfcf0`
(native desktop and portable runtime). Current full regression: 330 tests pass,
including the locally built SystemC replay and Windows desktop integration.
Further checkpoints include `3b757f7` (QNN quantization), `a6f79cc`
(geographic evaluation and uncertainty), `93083ab` (controlled VLM study),
and `18f63b3` (native evidence and demonstration).
Subsequent checkpoints are `369f5ef` (measured-service SystemC routing),
`573c8d9` (full-cohort TCN publication), and `d7ae6e3` (VideoMAE queue).
Both initial and representation queues are complete. Full-cohort VideoMAE
results have been verified against predictions and paired source-clip bootstrap
estimates have been published. Three Icarus testbenches were rerun
successfully. SystemC 3.0.2 was built locally with MSVC 19.44 and verified on
all five measured-service routing workloads.
The spatial-resolution queue is also complete. Its publisher verifies common
window grids and prevents confounding decision hops with feature resolution.
Native continuous inference and 24 TCN-gated SystemC replays are published;
train-only event calibration and native post-stream review are complete.
Corrected decoding replications are complete; reviewed stream exposure remains next.
