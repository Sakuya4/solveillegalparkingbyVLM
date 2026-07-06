# VLM Review Results

## Event: Privacy-Safe User Red-Line Parking Demo

Input package:

- Request: `outputs/user_redline_sam_demo/vlm_review_request.json`
- Evidence: SAM-prompted vehicle mask, bottom-footprint overlap, privacy-redacted
  image artifacts
- Rule evidence: `footprint_overlap_pixels = 5882`,
  `footprint_overlap_ratio = 0.1628`, red-line margin `24px`

This is a single-event smoke comparison, not a full accuracy benchmark. Its
purpose is to prove the VLM review interface, schema-following behavior, and
provider comparison workflow.

## Comparison

| Provider | Model / method | Decision | Confidence | Human review | Schema status | Baseline agreement |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `offline_evidence_reviewer` | deterministic evidence baseline | violation | 0.8591 | no | normalized | yes |
| `llava_interleave_qwen_0_5b` | `llava-hf/llava-interleave-qwen-0.5b-hf` | untrusted fallback | 0.0000 | yes | failed normalization | no |
| `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | violation | 0.9500 | no | normalized | yes |

## Interpretation

The 0.5B LLaVA/Qwen model can run locally, but it did not produce a valid review
schema. It generated partial JSON-like text and used unsupported values such as
`"stop"` for a boolean field. The pipeline correctly converted that into a
human-review-needed fallback. This is useful evidence that small VLMs should be
measured for schema-following stability, not only visual reasoning.

Qwen2.5-VL-3B produced a valid review object wrapped in a Markdown JSON fence.
The normalization layer parsed it successfully, agreed with the deterministic
baseline, and returned higher confidence. This makes Qwen2.5-VL-3B the first
strong local VLM candidate for the project.

## Current Claim

For the privacy-safe red-line demo, the strongest current result is:

```json
{
  "provider": "qwen2_5_vl_3b",
  "likely_violation": true,
  "confidence": 0.95,
  "human_review_needed": false,
  "agrees_with_baseline": true
}
```

This does not prove final deployment accuracy yet. The next step is to repeat
the same comparison over an annotated 20 to 50 event validation set and report
accuracy, schema-following rate, latency, and human-review rate per provider.
