from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .vlm_review import VlmReviewRequest


@dataclass(frozen=True)
class TransformersVlmConfig:
    model_id: str
    provider_name: str
    device: int | str | None = None
    max_new_tokens: int = 256
    return_full_text: bool = False


def build_transformers_messages(request: VlmReviewRequest) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    for path in request.image_paths:
        content.append({"type": "image", "url": _as_image_reference(path)})
    content.append({"type": "text", "text": _strict_json_prompt(request.prompt)})
    return [{"role": "user", "content": content}]


def run_transformers_image_text_to_text(
    request: VlmReviewRequest,
    config: TransformersVlmConfig,
    pipeline_factory: Callable[..., Any] | None = None,
) -> str:
    if pipeline_factory is None:
        from transformers import pipeline

        pipeline_factory = pipeline

    pipe_kwargs: dict[str, Any] = {
        "task": "image-text-to-text",
        "model": config.model_id,
    }
    if config.device is not None:
        pipe_kwargs["device"] = config.device

    pipe = pipeline_factory(**pipe_kwargs)
    output = pipe(
        text=build_transformers_messages(request),
        max_new_tokens=config.max_new_tokens,
        return_full_text=config.return_full_text,
    )
    return extract_generated_text(output)


def extract_generated_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict) and "generated_text" in first:
            return _stringify_generated_text(first["generated_text"])
        if isinstance(first, list) and first:
            return extract_generated_text(first)
    if isinstance(output, dict) and "generated_text" in output:
        return _stringify_generated_text(output["generated_text"])
    raise ValueError("Transformers VLM output does not contain generated_text.")


def _stringify_generated_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict) and "content" in last:
            content = last["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    return str(value)


def _strict_json_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Return only valid JSON. Do not wrap the JSON in Markdown. "
        "Use boolean values for likely_violation and human_review_needed."
    )


def _as_image_reference(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return str(Path(path).resolve())
