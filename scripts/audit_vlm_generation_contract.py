import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))


def main():
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline
    from illegal_parking.incident_review_service import QWEN_REVISION
    pipe = object.__new__(ImageTextToTextPipeline)
    old_pre, old_forward, _ = pipe._sanitize_parameters(do_sample=False, max_new_tokens=192, return_full_text=False)
    new_pre, new_forward, _ = pipe._sanitize_parameters(generate_kwargs={"do_sample": False, "max_new_tokens": 192}, return_full_text=False)
    if "do_sample" in new_pre or new_forward["generate_kwargs"]["do_sample"] is not False:
        raise ValueError("Installed pipeline does not satisfy the explicit generation contract")
    source = Path(transformers.__file__).parent / "pipelines/image_text_to_text.py"
    config_path = Path(hf_hub_download("Qwen/Qwen2.5-VL-3B-Instruct", "generation_config.json", revision=QWEN_REVISION, local_files_only=True))
    result = {
        "transformers_version": transformers.__version__,
        "pipeline_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256((ROOT / "Application/DebugVersion/src/illegal_parking/vlm_transformers.py").read_bytes()).hexdigest(),
        "legacy_requested": {"do_sample": False, "max_new_tokens": 192},
        "legacy_preprocess_parameters": old_pre, "legacy_generation_parameters": old_forward,
        "version_2_preprocess_parameters": new_pre, "version_2_generation_parameters": new_forward,
        "qwen_model_revision": QWEN_REVISION,
        "qwen_generation_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "qwen_generation_defaults": json.loads(config_path.read_text(encoding="utf-8")),
        "scope": "Actual installed pipeline argument routing and cached model defaults; no model weights are loaded or performance inferred",
    }
    path = ROOT / "docs/results/vlm_generation_contract.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
