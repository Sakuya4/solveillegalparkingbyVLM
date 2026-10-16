# Temporal Sampling and Presentation-Clock Audit

## Motivation

The reference extractor samples a 32-source-frame window at stride two,
producing 16 images and 15 optical-flow steps. The deployed temporal model
instead receives a nominal 7.5-Hz stream. Equal tensor dimensions therefore
do not imply equal temporal context. The controlled extension first audits
frame support and clock definitions before changing sampling or fitting.

## Corpus and Source Support

The audit reads all 2,027 original video containers and hashes every file.
Container counts exceed metadata counts by one in 2,021 clips; two counts
agree and four containers have fewer frames than metadata. This difference
is recorded rather than silently changing annotation indices. Nominal paired
window construction tolerates one additional terminal container frame but
retains the original metadata boundary in both arms. Other count mismatches,
insufficient context, and events after the final sampled frame are explicit
exclusions.

This diagnostic pairing retains the legacy effective event-frame anchor.
The subsequent presentation-time protocol must distinguish the supplied
annotation frame from any prior boundary clamp and verify terminal annotations
against decoded frame support before assigning physical event time.

The nominal pairing audit yields 1,973 clips and 3,174 windows. Its mutually
exclusive exclusions comprise 44 source rates below 7.5 Hz, three count
disagreements, five insufficient-context cases and two unobservable event
anchors. These are candidate sampling units, not a frozen training cohort.
Pre-event windows from accident clips retain their original interpretation;
they are not independent normal-camera exposure.

## Nominal Duration

| Sampling policy | Minimum (s) | Median (s) | 95th percentile (s) | Maximum (s) |
| --- | ---: | ---: | ---: | ---: |
| Source-frame stride two | 0.604 | 1.998 | 3.633 | 4.000 |
| First frame on nominal 7.5-Hz grid | 1.879 | 2.002 | 2.035 | 2.119 |

Durations in this table use source-index differences divided by the reported
container FPS. The second policy demonstrates reduced nominal context
variation, not measured classification improvement or exact presentation-time
normalization. Source-frame quantization remains visible.

## Presentation-Time Verification

Seven clips whose IID and geographic memberships are both training are
selected by source-FPS ranks at quantiles 0, 0.1, 0.25, 0.5, 0.75, 0.9 and 1.
Every frame of these clips is decoded with two CPU threads. Ordered packet
presentation timestamps are compared with decoded frame timestamps, with
separate origin normalization and exact frame-count verification.
[FFprobe's primary documentation](https://ffmpeg.org/ffprobe.html#Main-options)
defines packet, frame and timestamp inspection.

The seven decoded traces agree with packet presentation times within the
audit's 0.1-ms alignment tolerance. In contrast, the maximum difference
between the nominal frame-index/FPS clock and presentation time reaches
**1.36 s** in this training-only sample. The sample diagnoses a clock issue;
it does not estimate its prevalence across the remaining corpus.

The existing event studies retain their original frame-index clock, model,
cohort, and published endpoints. Their source-clock correction distinguishes
individual source indices from averaged playback timestamps; it does not
constitute presentation-timestamp decoding. Native camera observations use
decoder-arrival elapsed time and retain a separate timing contract.

## Predeclared Development Sequence

1. Verify presentation timestamps and annotation-frame support before freezing
   the new sampling manifest. Map the preserved annotation frame to that clock.
2. Freeze paired eligibility, physical duration, sampling-grid phase, grouped
   fit/calibration/test membership, spatial resolution and source hashes.
3. Fit matched legacy-context and time-normalized candidate/global TCNs with
   unchanged optimization settings. Interpret the intervention as joint
   sampling/context control, not an isolated architecture effect.
4. Calibrate thresholds using held-out training clips. Report paired source-
   clip uncertainty, test alert demand, event recall and trigger delay under
   matching presentation-time inference contracts.
5. Keep the current 500-clip reference binaries and desktop defaults unchanged.
   Publish new results separately and retain negative findings.

No additional temporal model is fitted as part of this source audit.
[Corpus hashes, counts, clock traces and implementation hashes](results/cadence_source_audit.json).

```powershell
python scripts/audit_incident_cadence.py --ffprobe C:/ffmpeg/bin/ffprobe.exe --output outputs/accident/NEW_CADENCE_AUDIT.json
python -m pytest tests/test_incident_cadence.py
```
