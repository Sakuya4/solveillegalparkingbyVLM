import importlib.util
from pathlib import Path


def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_context_score_rtl.py"
    spec = importlib.util.spec_from_file_location("context_rtl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantization_audit_detects_near_threshold_policy_changes():
    _, summary, _, _ = verifier().audit_stream([0, 15], [.5, .5], .5000001)
    assert summary["positive_changes"] == 2
    assert summary["review_changes"] == 1 and summary["episode_changes"] == 1


def test_hardware_stream_preserves_empty_source_reset_and_gap_boundaries():
    module = verifier()
    tokens, summary, _, _ = module.audit_stream([], [], .5)
    assert len(tokens) == 1 and tokens[0][1] == 1 and summary["decisions"] == 0
    tokens, summary, _, _ = module.audit_stream([0, 30, 45], [1, 1, 1], .5)
    assert sum(token[1] for token in tokens) == 2
    assert summary["episode_changes"] == 0 and summary["maximum_probability_error"] == 0
