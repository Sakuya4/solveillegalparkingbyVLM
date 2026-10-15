# ADR 004: Controlled Temporal Evidence Density

Status: Accepted for the next evaluation stage, 16 October 2026.

## Research Question

Does a denser temporal representation improve collision grounding when contact
may fall between sparse evidence frames?

## Protocol

Use the same ten IID test clips and paired annotation-defined collision and
pre-event intervals. Keep source selection, privacy processing, image width,
collision-specific prompt, 192-token limit, model revision, and BF16 execution
constant. First obtain greedy-decoding baselines for both models. Compare Qwen
with three reference frames and eight evenly spaced frames over each interval.
Report schema validity, all-case structured correctness, confusion counts,
and service time. Preserve earlier upstream-generation runs as separate records.

## Interpretation

The hypothesis concerns evidence density, with source clip as the paired unit.
It does not require new model training or accident bounding boxes. All selected
cases are evaluated. Temporal evidence availability and model processing cost
are recorded separately for deployment analysis.
