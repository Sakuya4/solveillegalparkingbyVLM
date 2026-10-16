# Multimodal Generation-Parameter Contract

## Runtime Audit

The installed Transformers 5.13.0 image-text-to-text pipeline separates
preprocessing arguments from `generate_kwargs`. Its `_sanitize_parameters`
routes a top-level `do_sample` argument to the processor, whereas nested
`generate_kwargs["do_sample"]` reaches model generation. The previous wrapper
requested `do_sample=False` at the top level. Its legacy result metadata records
the requested setting, not a verified effective decoding override.

The cached Qwen revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
specifies `do_sample=true` and temperature 0.000001 in `generation_config.json`.
This very low temperature does not establish an enforced greedy contract.
The LLaVA revision has a separate default configuration. Therefore the earlier
paired comparison must not be interpreted as a verified common greedy decoder.
Measured responses, schema outcomes, and elapsed times remain unchanged.

## Corrected Execution

[Installed sanitizer, wrapper and cached-configuration hashes](results/vlm_generation_contract.json)

Generation contract version 2 forwards both `do_sample=False` and the token
budget through `generate_kwargs`. A regression test invokes the installed
pipeline's actual sanitizer and verifies that decoding is absent from the
processor arguments and present in the generation arguments. Unit tests also
verify the wrapper call and saved-event provenance.

The corrected three/eight-frame experiments reuse every original request and
image. Model revisions are pinned to the previous measured revisions. Separate
`*_forwarded_v2_result.json` case files and `*_forwarded_v2_batch.json` summaries
preserve legacy results. Cache signatures include the wrapper hash, generation
contract version, Transformers version, request, images, model revision, and
token budget. Initial, representation, resolution, and calibration queues are
not restarted; a separate `decoding` queue contains only these replications.

The earlier queue and variable-service studies retain their observed service
samples. They describe those measured workloads rather than the service
distribution of the corrected decoder. Likewise, evidence-aware policy
reanalysis remains tied to its original hashed responses until the separately
versioned replication is available.

```powershell
python -m pytest tests/test_vlm_transformers.py tests/test_incident_review_service.py
python scripts/audit_vlm_generation_contract.py
python scripts/run_overnight_research.py --phase decoding --deadline 2026-10-17T04:00:00+00:00
```

Implementation reference: the installed `transformers/pipelines/image_text_to_text.py`
and [primary pipeline source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_text_to_text.py).
