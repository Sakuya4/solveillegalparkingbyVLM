from illegal_parking.vlm_review import VlmReviewRequest
from illegal_parking.vlm_transformers import (
    TransformersVlmConfig,
    build_transformers_messages,
    extract_generated_text,
    run_transformers_image_text_to_text,
)
from scripts.run_transformers_vlm_review import _parse_device


def test_build_transformers_messages_uses_image_text_chat_content(tmp_path):
    image_path = tmp_path / "overlay.jpg"
    image_path.write_bytes(b"not-a-real-image")
    request = VlmReviewRequest(
        task="redline_parking_review",
        prompt="Return JSON.",
        image_paths=[str(image_path)],
        evidence={},
    )

    messages = build_transformers_messages(request)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][0]["url"].endswith("overlay.jpg")
    assert messages[0]["content"][1]["type"] == "text"
    assert "Return only valid JSON" in messages[0]["content"][1]["text"]


def test_run_transformers_image_text_to_text_uses_pipeline_contract():
    calls = {}

    def fake_pipeline(**kwargs):
        calls["pipeline_kwargs"] = kwargs

        def fake_pipe(**pipe_kwargs):
            calls["pipe_kwargs"] = pipe_kwargs
            return [{"generated_text": '{"likely_violation": true, "confidence": 0.7, "visual_reasons": [], "missing_evidence": [], "human_review_needed": false}'}]

        return fake_pipe

    request = VlmReviewRequest(
        task="redline_parking_review",
        prompt="Return JSON.",
        image_paths=["overlay.jpg"],
        evidence={},
    )
    config = TransformersVlmConfig(model_id="test/model", provider_name="test_vlm", device=0, max_new_tokens=32)

    text = run_transformers_image_text_to_text(request, config, pipeline_factory=fake_pipeline)

    assert calls["pipeline_kwargs"] == {"task": "image-text-to-text", "model": "test/model", "device": 0}
    assert calls["pipe_kwargs"]["max_new_tokens"] == 32
    assert calls["pipe_kwargs"]["return_full_text"] is False
    assert '"likely_violation": true' in text


def test_extract_generated_text_handles_chat_style_generated_text():
    output = [
        {
            "generated_text": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": [{"type": "text", "text": '{"likely_violation": false}'}]},
            ]
        }
    ]

    assert extract_generated_text(output) == '{"likely_violation": false}'


def test_parse_device_accepts_int_or_string():
    assert _parse_device("0") == 0
    assert _parse_device("cpu") == "cpu"
    assert _parse_device(None) is None
