# ADR 003: Train-Only Calibration And Persistent Event Gate

Status: Accepted

## Context

A fixed 0.5 classifier threshold produced high window false-positive rates.
Selecting a threshold on test data would leak evaluation information, while a
single positive window creates excessive review load on live CCTV.

## Decision

Split training clips by path into fit and calibration groups. Train on the fit
partition, then choose the lowest threshold that meets the requested false
positive budget on the calibration partition. Never use test predictions for
threshold selection.

At runtime, preserve every raw window score but create a review event only after
two consecutive positive windows. Report raw positives and persistence-gated
events separately.

## Consequences

- IID test FPR decreases, with a measured recall/F1 cost.
- Geographic FPR may exceed the calibration budget and remains an OOD result.
- The event gate suppresses isolated live-stream spikes without hiding them.
- Camera-hour claims still require longer independently reviewed normal footage.
