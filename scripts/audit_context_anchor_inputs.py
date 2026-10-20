from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_continuous_context_study import DIRECTORY, load_study, load_fit, load_models, sha, dump
from illegal_parking.incident_context_study import fit_indices
from illegal_parking.incident_tcn import fit_sequence_standardizer
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
import numpy as np


def main():
    protocol, execution, rows = load_study()
    features, _ = load_fit(rows)
    parent = ROOT / execution["anchored_cache"]["directory"] / "sequences.npz"
    # Preserve the historical channel-before-source indexing and its reduction layout.
    with np.load(parent, allow_pickle=False) as data:
        old_features = data["features"].copy()
        old_names = data["path"].astype(str)
        old_indices = data["frame_indices"].copy()
        old_targets = data["target"].copy()
        old_parts = {split: data[split + "_partition"].astype(str) for split in protocol["splits"]}
    registry = load_models()
    records = []
    for feature in protocol["features"]:
        columns = [i for i, name in enumerate(CANDIDATE_MOTION_FEATURE_NAMES) if name.startswith(feature)]
        historical_channels = np.asarray(old_features, dtype=np.float32)[:, :, columns]
        for split in protocol["splits"]:
            selected = fit_indices(rows, protocol, "anchored", split)
            old = np.flatnonzero(old_parts[split] == "fit")
            new_input = features[selected][:, :, columns]
            old_input = historical_channels[old]
            identities = [(rows[i]["path"], tuple(rows[i]["indices"])) for i in selected]
            old_identities = [(old_names[i], tuple(old_indices[i])) for i in old]
            if identities != old_identities or not np.array_equal(old_targets[old], [rows[i]["target"] for i in selected]):
                raise ValueError("Historical fit rows or labels differ")
            if not np.array_equal(new_input, old_input):
                raise ValueError("Historical input tensor values differ")
            current_scaler = fit_sequence_standardizer(new_input)
            historical_scaler = fit_sequence_standardizer(old_input)
            contiguous = fit_sequence_standardizer(np.ascontiguousarray(new_input))
            record = {"model": f"anchored_{feature.rstrip('_')}_{split}", "contexts": len(selected),
                      "row_label_and_tensor_values_equal": True, "historical_strides": list(old_input.strides),
                      "new_strides": list(new_input.strides),
                      "mean_difference": float(np.max(np.abs(current_scaler.feature_mean - historical_scaler.feature_mean))),
                      "scale_difference": float(np.max(np.abs(current_scaler.feature_scale - historical_scaler.feature_scale))),
                      "contiguous_minus_historical_mean": float(np.max(np.abs(contiguous.feature_mean - historical_scaler.feature_mean))),
                      "contiguous_minus_historical_scale": float(np.max(np.abs(contiguous.feature_scale - historical_scaler.feature_scale)))}
            records.append(record)
    output = ROOT / "docs/results/continuous_context_anchor_inputs.json"
    if output.exists():
        raise FileExistsError("Do not overwrite historical input audit")
    dump(output, {"auditor_sha256": sha(Path(__file__)), "historical_features_sha256": sha(parent),
                  "new_features_sha256": sha(DIRECTORY / "fit_sequences.npz"), "models_sha256": sha(DIRECTORY / "models.json"),
                  "results": records,
                  "scope": "FIT-only tensor/order/layout audit. Equal tensor values do not guarantee bitwise-equivalent float32 reductions or GPU optimization. No refitting, threshold change, test-score inspection or default selection."})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
