# Evidence-Aware Multimodal Review Routing

## Operational Role

The temporal model emits an event before optional VLM service. The VLM inspects
ordered privacy-treated imagery and returns visual reasons, evidence gaps,
confidence, and an operator-review request. Its role is evidence triage, not
replacement of the original event record or independent verification of a
collision label. Streaming exports its post-context package while immediate
notification remains independent of review-service capacity.

## Disposition Contract

Policy version 2 distinguishes invalid responses, temporal/visual disagreement,
incomplete evidence, requested operator review, low confidence, missing visual
rationale, and consistent visual support. Positive support requires schema
validity, agreement with the candidate, finite confidence at least 0.7,
no declared evidence gap, no operator-review request, and a visual rationale.
The confidence value is model-reported, not a calibrated correctness probability.

The desktop displays evidence gaps and requested review. Saved reports retain
their original model response; the UI derives the current policy outcome
instead of trusting an obsolete confirmation label. New analysis records the
policy version alongside the disposition. No VLM outcome removes the temporal
event from the report.

## Controlled Cohort Reanalysis

The preserved Qwen cohort contains 20 annotation-defined sequences from ten
IID test clips. Raw structured classification remains 14/20 correct. A
confidence-only positive-support rule accepts eight sequences: six annotated
positives and two annotated negatives, giving 0.75 precision among supported
cases and 0.60 coverage of annotated positives.

All eight positive responses also declare evidence gaps. Policy version 2
therefore routes eight sequences to incomplete-evidence review and twelve to
temporal/visual disagreement. No sequence receives fully supported status in
this cohort. Supported-case precision is undefined, not 1.0; all 20 remain
unresolved by the selective support gate. The original classification metrics
and model outputs are unchanged.

This result distinguishes JSON validity, raw classification, and operational
evidence sufficiency. It motivates targeted temporal context and explicit
review outcomes rather than interpreting high self-reported confidence as
confirmation. The paired-sequence cohort is not a measured TCN-triggered field
cohort, and a missing-evidence declaration is evaluated as a routing signal,
not as an independently established scene fact.

The [generation-contract audit](vlm_generation_contract.md) documents the
legacy requested-decoding metadata and separately versioned corrected runs.

[Case-level routing and source hashes](results/vlm_review_dispositions.json)

```powershell
python scripts/evaluate_vlm_review_dispositions.py
python -m pytest tests/test_incident_vlm_evidence.py tests/test_vlm_review_dispositions.py tests/test_desktop_app.py
```
