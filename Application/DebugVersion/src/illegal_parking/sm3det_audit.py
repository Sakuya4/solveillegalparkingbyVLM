from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def audit_sm3det_release(repository: Path) -> dict[str, Any]:
    readme = (repository / "README.md").read_text(encoding="utf-8")
    config_path = repository / "configs" / "SM3Det" / "SM3Det_convnext_t.py"
    config_text = config_path.read_text(encoding="utf-8")

    source_ratio = _literal_assignment(config_text, "source_ratio")
    num_classes = _literal_assignment(config_text, "num_classes")
    num_experts = _integer_keyword(config_text, "num_experts")
    top_k = _integer_keyword(config_text, "top_k")
    official_metrics = _official_table_row(readme)

    return {
        "release_verified": True,
        "config": config_path.relative_to(repository).as_posix(),
        "modalities": ["RGB", "SAR", "infrared"],
        "source_ratio": source_ratio,
        "num_classes": num_classes,
        "mixture_of_experts": {
            "num_experts": num_experts,
            "top_k": top_k,
        },
        "official_release_metrics": official_metrics,
        "official_environment": {
            "python": "3.10",
            "pytorch": "1.12.0+cu113",
            "mmcv_full": "1.6.1",
        },
        "license": "CC BY-NC 4.0",
        "cctv_transfer_assessment": {
            "direct_cctv_metric_available": False,
            "reason": (
                "The official benchmark is multi-modal remote-sensing object detection, "
                "not temporal roadside-CCTV incident recognition."
            ),
            "recommended_role": (
                "Optional cloud-side hard-frame routing or a separately fine-tuned "
                "multi-sensor research branch; exclude it from the real-time edge path."
            ),
        },
        "sources": [
            "https://github.com/zcablii/SM3Det",
            "https://arxiv.org/abs/2412.20665",
        ],
    }


def _literal_assignment(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing assignment: {name}")


def _integer_keyword(source: str, name: str) -> int:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)", source)
    if not match:
        raise ValueError(f"Missing keyword: {name}")
    return int(match.group(1))


def _official_table_row(readme: str) -> dict[str, float | int | str]:
    pattern = re.compile(
        r"<td>SM3Det</td>\s*<td>(\d+)G</td>\s*<td>(\d+)M</td>\s*"
        r"<td>Overall</td>\s*<td>([\d.]+)</td>\s*<td>([\d.]+)</td>\s*"
        r"<td>([\d.]+)</td>",
        re.IGNORECASE,
    )
    match = pattern.search(readme)
    if not match:
        raise ValueError("Could not find the official SM3Det result row")
    return {
        "benchmark": "SOI-Det remote sensing",
        "flops_g": int(match.group(1)),
        "parameters_m": int(match.group(2)),
        "rgb_map": float(match.group(3)),
        "sar_map": float(match.group(4)),
        "infrared_map": float(match.group(5)),
    }
