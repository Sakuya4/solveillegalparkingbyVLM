from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_continuous_context_study import PROTOCOL, DIRECTORY, load_study, load_models, sha, dump, now


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    load_study()
    comparison_path = ROOT / "docs/results/presentation_comparison.json"
    previous = json.loads(comparison_path.read_text())
    registry = load_models()
    results = []
    for definition in previous["models"]:
        if definition["arm"] != "fixed":
            continue
        feature, split = definition["feature"], definition["split"]
        key = f"anchored_{feature}_{split}"
        old_path = ROOT / "models" / f"presentation_fixed_{feature}_{split}.pt"
        if sha(old_path) != definition["model_sha256"]:
            raise ValueError("Historical anchored-control weights changed")
        old = torch.load(old_path, map_location="cpu", weights_only=True)
        current = torch.load(ROOT / registry[key]["path"], map_location="cpu", weights_only=True)
        if old["feature_names"] != current["feature_names"] or set(old["state_dict"]) != set(current["state_dict"]):
            raise ValueError("Anchored control has a different architecture or feature order")
        differences = [float((old["state_dict"][name] - value).abs().max()) for name, value in current["state_dict"].items()]
        scalar_differences = {field: float(np.max(np.abs(np.asarray(old["standardizer"][field]) - np.asarray(current["standardizer"][field]))))
                              for field in ("feature_mean", "feature_scale")}
        results.append({"model": key, "previous_model_sha256": definition["model_sha256"], "new_model_sha256": registry[key]["sha256"],
                        "parameters": sum(value.numel() for value in current["state_dict"].values()),
                        "maximum_parameter_difference": max(differences), "bitwise_state_equal": all(value == 0 for value in differences),
                        "standardizer_differences": scalar_differences})
    output = ROOT / "docs/results/continuous_context_anchor_replication.json"
    if output.exists():
        raise FileExistsError("Do not overwrite replication audit")
    dump(output, {"protocol_sha256": sha(PROTOCOL), "prior_comparison_sha256": sha(comparison_path),
                  "models_sha256": sha(DIRECTORY / "models.json"), "auditor_sha256": sha(Path(__file__)),
                  "completed_at_utc": now(), "results": results,
                  "scope": "Numerical historical-control audit only. Compare final-epoch parameters and fit-only standardizers, excluding changed threshold calibration. No test scores or deployment defaults are selected."})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
