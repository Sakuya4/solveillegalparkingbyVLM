# Project Direction

## Selected Direction

Use the application-facing direction:

```text
Smart-city traffic violation hot-spot monitoring, event review, and edge deployment system
```

The project should not be presented as only a red-line parking detector anymore.
Red-line parking remains the first complete event case, but the larger value is
the reusable event evidence and review workflow.

## Why This Direction Fits

- Government open data does not provide a ready-to-train red-line parking image
  dataset, so a narrow red-line-only topic would make validation fragile.
- The current repo already has more than red-line detection: edge simulation,
  detector comparison, SAM evidence, VLM review, hot-spot analysis, A1/A2 risk
  ranking, and Verilog event filtering.
- An application-facing title makes the project useful even when the first rule
  is replaced by another traffic event type.

## Final Positioning

The project contribution is:

```text
camera/live feed
  -> detector and tracker
  -> event evidence generation
  -> SAM-assisted mask/overlap evidence
  -> VLM or human review
  -> hot-spot and accident-risk validation
  -> edge/hardware-aware filtering
```

## Role Of Red-Line Parking

Red-line parking is the first complete demonstration case:

- It has clear road-rule evidence.
- It benefits from two-sided red-line contact-band logic.
- It benefits from SAM footprint/mask evidence.
- It can be reviewed by Qwen2.5-VL-3B and compared with a deterministic
  evidence baseline.

It should be described as the first event type, not the whole system boundary.

## Next Work

1. Build a small event validation manifest for 20 to 50 images or clips.
2. Include positives, negatives, uncertain cases, red-line cases, and non-red-line
   traffic-event cases.
3. Run offline evidence review, Qwen2.5-VL-3B, and optional cloud VLM review.
4. Report event accuracy, schema-following rate, human-review rate, trigger
   delay, and relation to violation/A1/A2 hot spots.
