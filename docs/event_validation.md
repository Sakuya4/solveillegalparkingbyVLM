# Event Validation Workflow

## Purpose

The project needs event-level validation, not only single-image demos. The
validation set should measure whether the event system and VLM reviewers can
correctly classify traffic-event candidates as:

- `violation`
- `no_violation`
- `uncertain`

Red-line parking remains the first complete case, but the manifest can also
hold non-red-line events so the system does not overfit to "vehicle plus curb"
as a violation.

## Real Versus Simulated Data

Use real events for project claims:

- self-collected Taiwan red-line parking photos or clips
- privacy-redacted user-provided images
- approved CCTV captures where redistribution rules are clear
- official web images only when licensing allows reuse

Use simulated events for development:

- CLI and report workflow testing
- VLM schema-failure handling
- edge/hardware throughput experiments
- negative cases when real data is not yet enough

Simulated data must not be used as final accuracy evidence.

## Manifest Format

Example file:

```text
data/validation/events_manifest.csv.example
```

Columns:

| Column | Meaning |
| --- | --- |
| `event_id` | Stable event identifier. |
| `image_path` | Local privacy-redacted image or frame path. Empty is allowed for simulated rows. |
| `label` | `violation`, `no_violation`, or `uncertain`. |
| `source_type` | `self_collected`, `official_web`, `cctv_capture`, `web_reference`, or `simulated`. |
| `is_simulated` | `true` when the row is generated for workflow testing. |
| `has_red_line` | Whether a red curb line is visible or intended in the event. |
| `red_line_visible` | `clear`, `faded`, `occluded`, or `none`. |
| `plate_redacted` | Whether license plates are redacted. |
| `timestamp_redacted` | Whether timestamp/address overlays are redacted. |
| `redistribution_ok` | Whether the media can be redistributed in an open repo. |
| `notes` | Free-form validation notes. |

## Simulated Workflow

Create a small simulated validation set:

```powershell
python scripts\create_simulated_validation_set.py `
  --output-dir outputs\validation\simulated_demo
```

Evaluate simulated provider results:

```powershell
python scripts\evaluate_vlm_validation.py `
  --manifest outputs\validation\simulated_demo\events_manifest.csv `
  --results-root outputs\validation\simulated_demo\review_results `
  --provider offline_evidence_reviewer `
  --provider qwen2_5_vl_3b `
  --provider small_vlm `
  --output outputs\validation\simulated_demo\vlm_validation_report.json
```

The simulated report should be treated as a workflow smoke test. It verifies the
metrics table shape and failure handling before real event data is available.

## Real Event Workflow

1. Copy `data/validation/events_manifest.csv.example` to a local untracked
   manifest file, for example `data/raw/validation/events_manifest.csv`.
2. Add privacy-redacted images or frame paths.
3. Label each event as `violation`, `no_violation`, or `uncertain`.
4. Generate or copy VLM review results into:

```text
outputs/validation/<run_name>/review_results/<event_id>/<provider>.json
```

5. Run `scripts/evaluate_vlm_validation.py` with the provider names.

The first real validation target should be 20 to 50 events:

| Type | Suggested count |
| --- | ---: |
| Taiwan red-line violation positives | 15 to 20 |
| Taiwan red-line negatives or uncertain cases | 8 to 10 |
| Taiwan no-red-line vehicle negatives | 8 to 10 |
| Optional public/web reference examples | 5 to 10 |

## Metrics

The report includes:

- precision
- recall
- false-positive rate
- missing result count
- schema success rate
- human review rate
- skipped uncertain event count

These metrics let the project compare deterministic evidence review,
Qwen2.5-VL-3B, small local VLMs, and optional cloud VLMs on the same event set.
