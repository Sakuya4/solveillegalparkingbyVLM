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

- [ ] Complete geographic event-level evaluation using the geographic model.
- [ ] Expand clean-blind VLM evaluation with independent event labels.
- [ ] Compare review-routing strategies under a fixed latency/queue budget.
- [ ] Connect measured traces and desktop runtime diagnostics to the existing
  Verilog/SystemC workflow and Qualcomm deployment preparation.
- [ ] Present bootstrap confidence intervals and operating-point tradeoffs.
- [x] Rewrite the main README and current research report in academic English;
  preserve team attribution, original images, licensing, and historical context.
- [ ] Produce desktop screenshots, a reproducible demonstration, and a compact
  manuscript outline with research questions and evidence-backed contributions.
- [ ] Run appropriate regression tests and review changes before checkpointing.

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
