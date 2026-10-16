# Risk-Aware Edge Intelligence for Traffic Incident Detection and Camera Allocation

## Abstract

Roadside video intelligence requires timely incident recognition and effective
allocation of limited monitoring resources. This study presents a native
desktop framework integrating causal temporal inference, multimodal evidence,
hardware-aware execution, and government-informed camera allocation. A public
fixed-CCTV benchmark supports event evaluation and matched-cohort comparisons.
Taiwan government records support a temporal allocation experiment that selects
camera subsets from historical demand and evaluates subsequent geographic
coverage. On 100 IID incident clips, the responsive profile achieves 0.750
event recall with a median trigger delay of 0.394 s. Ten-camera coverage-aware
selection captures 15.49% of held-out New Taipei crash risk, compared with
13.69% for independent ranking and 11.38% for randomized allocation.
Training-calibrated A16W8 quantization preserves float32 threshold decisions
over all 450 test windows. The framework connects incident evidence, allocation
efficiency, and resource accounting through reproducible research artifacts.
An extended 2,021-clip cohort identifies domain-dependent tradeoffs between
candidate and global motion representations. Measured-service routing separates
proposal coalescing, admission decisions, and downstream queue loss.

## 1. Introduction

The original red-line parking application addressed a rule-based event.
Continuous roadside monitoring adds temporal ambiguity, geographic shift,
operator-review load, and heterogeneous compute constraints. Public crash
records provide geographic safety priorities that extend model-only evaluation.

Research questions:

- RQ1: How do compact temporal features and pretrained video representations
  trade sensitivity against false-positive burden under common calibration?
- RQ2: Can historical crash demand improve monitoring coverage under a fixed
  camera budget on a subsequent temporal partition?
- RQ3: How do quantization, persistence gating, and review capacity affect the
  executable incident-to-evidence workflow?

The contribution connects temporal perception and public-sector allocation
through availability-aligned evidence, temporal government validation, and an
explicit execution/resource contract. Compare this combination with the closest
systems identified in [the literature matrix](literature_basis.md).

## 2. Related Work

Organize prior work by accident detection/anticipation, edge video analytics,
road-safety network screening, and coverage optimization. Discuss assumptions
about time, geography, and resource availability. Use primary papers for
VideoMAE, SAM2, Qwen-VL, NAFNet, and SM3Det; use journal literature for the
system formulation and evaluation design.

## 3. Methods

For window x[t-L+1:t], the temporal model produces p[t] after the final sample
is available. Persistence generates a review event after the required positive
run. Event logs and display share this timing convention. Clean temporal
evidence is exported independently of model-score overlays.

For demand weights w[i], camera subset S, and proximity indicator a[i,j]:

```text
maximize F(S) = sum_i w[i] * 1{exists j in S: a[i,j] = 1}
subject to |S| <= k.
```

Greedy selection maximizes uncovered marginal demand. Its approximation
guarantee concerns historical coverage; later-period coverage is measured.

The native UI uses a separate inference worker. Feature standardization is
embedded in ONNX. QNN preparation uses fixed shape and training-only A16W8
calibration. Every run retains providers, model hashes, and processing times.
Verilog tests deterministic triggers; SystemC replays camera/review queues
using candidate traces and explicit latency assumptions.

## 4. Experimental Protocol

Use common incident cohorts and splits. Select thresholds on clip-grouped
training holdout. Report event recall/delay separately from window F1/FPR,
alongside off-target episodes and normal-stream exposure.

For allocation, select on January--September 2024 and evaluate
October--December. Preserve all severity/radius/budget settings and matched
inventories. Use district blocks for paired allocation uncertainty and
clip-level Wilson intervals for recall. Record random seeds and source hashes.

## 5. Results

Primary tables and figures cover event timing; calibrated temporal models;
camera-budget curves and district uncertainty; SAM2/VLM evidence quality;
ONNX/QDQ agreement; queue capacity; and native event inspection. Numeric
sources are linked from the [evaluation report](final_evaluation_report.md).
Include geographic event recall of 0.630 and the controlled 20-sequence VLM
comparison. The three/eight-frame ablation and measured-service queue replay
test the relationship between evidence cost and downstream review capacity.
The full-cohort temporal study and five-policy SystemC-verified routing results
are retained separately from the 500-clip reference comparison. Report explicit
budget denials alongside completed reviews and queue drops.
The full VideoMAE comparison includes eight model/split results, AUROC/AP,
source-clip bootstrap intervals, and test-prevalence baselines. Persistent
public-camera observations report source cadence and analysis time separately.
Continuous inference verifies batch parity and reports decoder-arrival latency
with a bounded mailbox. The 24-condition TCN-gated hardware replay measures
annotation-matched event retention, demonstrating why queue-drop reduction
alone is not a sufficient endpoint for evidence routing.
Evidence-aware VLM routing distinguishes schema compliance, classification,
and declared evidence sufficiency. Retain the zero-supported-case result under
the complete-evidence policy, with undefined support precision and explicit
unresolved coverage, alongside unchanged raw model classification results.
Document the installed generation-argument audit: legacy runs record requested
sampling settings; version-2 runs explicitly forward them to generation.
Keep the corrected replications and source hashes separate from legacy outcomes.
Version-2 Qwen/LLaVA all-case correctness is 0.60/0.30; the paired source-clip
difference is +0.30 [0.10, 0.50]. Eight-frame Qwen correctness is 0.55 with a
three-frame contrast of -0.05 [-0.25, 0.15]. Preserve both negative evidence-
density findings and the ten-clip resampling unit.
The exposure workflow records reviewer-attributed clean-video observations,
source-clock gap exclusion, duplicate/conflict handling and exact one-sided
Poisson bounds. Acquisition inventories and VLM responses do not supply
independent normal labels. This keeps model scores, review decisions and
exposure denominators as separate experimental objects.
The fixed-contract event-calibration study reports all 24 settings selected
on held-out training clips. Distinguish improvements in two-window event recall
from test budget exceedance and changes in one-window recall. The 176-scenario
service study adds empirical review-time variability and 32 SystemC counter
checks; assignment ranges are sensitivity summaries, not confidence intervals.

## 6. Discussion

Explain camera overlap and the gap between independent and marginal-gain
selection. Discuss when severity and count strategies coincide. Interpret
district variation, temporal shift, proximity versus visibility, and review
bottlenecks. Design subsequent intervention evaluation with traffic exposure
and regression-to-the-mean controls.

## 7. Reproducibility

Provide source manifests, model/archive hashes, split definitions, seeds,
calibration scope, complete sweeps, launch commands, tests, figure scripts,
software versions, and target specifications for hardware measurements.
Distinguish presentation timestamps from frame-index/container-FPS clocks.
The corpus clock audit hashes 2,027 sources and decodes seven train-only
traces before the cadence/context ablation. Preserve annotation-frame support
and specify physical-time cohorts before fitting additional models.
The frozen presentation study retains 1,861 clips and 2,985 paired windows,
with matched legacy-context and bounded-jitter 7.5-Hz policies, fixed grouped
holdouts and eight candidate/global by IID/geographic fits. Its four source-
clip bootstrap contrasts are pointwise; continuous event evaluation remains
separate and uses the fixed 88/90-clip intersections.
All eight fits now complete. Candidate fixed-minus-source-frame window-F1
changes are +3.64 pp [0.35, 6.64] for IID and +1.09 pp [-1.94, 4.39] for
geographic testing; global changes are -6.68 pp [-9.72, -3.64] and -8.24 pp
[-10.82, -5.53]. Candidate gains accompany increased FPR rather than improved
AUROC. Retain both directions and distinguish ranking from threshold effects.
The separate event follow-up uses common PTS inputs for both fitted arms,
two-window persistence, missing-hop resets and training-only episode budgets.
Include the source-hashed reference-case video and native walkthrough as
qualitative workflow material. The reference execution preserves temporal/VLM
disagreement and separates per-frame processing, source-time triggering and
VLM service duration; it does not enter aggregate performance estimates.
