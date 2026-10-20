import copy
import pytest

from scripts.review_daytime_camera import QWEN_REVISION, validate_review


def test_daytime_reviews_bind_all_evidence_and_forwarded_decoding():
    definition = {"request_sha256": "request", "image_sha256": ["a", "b", "c"], "prompt_sha256": "prompt"}
    review = {**definition, "model_revision": QWEN_REVISION, "generation_contract_version": 2,
              "do_sample": False, "max_new_tokens": 256, "seed": 42}
    validate_review(definition, review)
    for key in review:
        changed = copy.deepcopy(review)
        changed[key] = "changed"
        with pytest.raises(ValueError):
            validate_review(definition, changed)
