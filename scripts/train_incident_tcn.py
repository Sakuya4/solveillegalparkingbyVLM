from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_baseline import binary_classification_metrics, filter_feature_names
from illegal_parking.incident_tcn import TemporalConvClassifier, fit_sequence_standardizer


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Train a compact causal TCN on ACCIDENT motion sequences.")
    parser.add_argument("--sequences", default="data/processed/accident/motion_sequences.npz")
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=None,
        help="Use sequence channels beginning with this prefix. Defaults to global_ to avoid oracle ROI leakage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-output", default="models/accident_tcn.pt")
    parser.add_argument("--report-output", default="outputs/accident/tcn_report.json")
    parser.add_argument("--predictions-output", default="outputs/accident/tcn_predictions.csv")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch size, and learning rate must be positive")
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    payload = np.load(_resolve(args.sequences), allow_pickle=False)
    all_feature_names = tuple(payload["feature_names"].astype(str).tolist())
    include_prefixes = tuple(args.include_prefix or ("global_",))
    feature_names = filter_feature_names(all_feature_names, include_prefixes)
    feature_indices = [all_feature_names.index(name) for name in feature_names]
    features = np.asarray(payload["features"], dtype=np.float32)[:, :, feature_indices]
    targets = np.asarray(payload["target"], dtype=np.int64)
    split_field = "iid_split" if args.split_scheme == "iid" else "geographic_split"
    splits = np.asarray(payload[split_field]).astype(str)
    train_indices = np.flatnonzero(splits == "train")
    test_indices = np.flatnonzero(splits == "test")
    _validate_split(targets, train_indices, test_indices, split_field)

    standardizer = fit_sequence_standardizer(features[train_indices])
    train_features = standardizer.transform(features[train_indices])
    test_features = standardizer.transform(features[test_indices])
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]

    model = TemporalConvClassifier(
        input_features=features.shape[2],
        channels=args.channels,
        dropout=args.dropout,
    ).to(device)
    positive_count = int(np.sum(train_targets == 1))
    negative_count = int(np.sum(train_targets == 0))
    positive_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_targets.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_targets)
            total_samples += len(batch_targets)
        losses.append(total_loss / total_samples)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} loss={losses[-1]:.6f}")

    train_probabilities = _predict(model, train_features, args.batch_size, device)
    test_probabilities = _predict(model, test_features, args.batch_size, device)
    report = {
        "model": "causal_temporal_convolutional_network",
        "device": str(device),
        "split_scheme": args.split_scheme,
        "seed": args.seed,
        "epochs": args.epochs,
        "channels": args.channels,
        "dropout": args.dropout,
        "sequence_shape": list(features.shape[1:]),
        "feature_names": list(feature_names),
        "include_prefixes": list(include_prefixes),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train": binary_classification_metrics(train_targets, train_probabilities, args.threshold),
        "test": binary_classification_metrics(test_targets, test_probabilities, args.threshold),
        "test_by_collision_type": _group_metrics(
            payload["collision_type"][test_indices].astype(str),
            test_targets,
            test_probabilities,
            args.threshold,
        ),
        "test_by_quality": _group_metrics(
            payload["quality"][test_indices].astype(str),
            test_targets,
            test_probabilities,
            args.threshold,
        ),
        "limitations": [
            "Normal windows are pre-incident segments from accident clips, not independent normal-only CCTV videos.",
            "This compact TCN consumes optical-flow/frame-difference sequences and is not a VideoMAE comparison.",
            "False-positive rate is window-level and must not be reported as false alarms per camera-hour.",
        ],
        "sources": [
            "https://arxiv.org/abs/1803.01271",
            "https://github.com/accidentbench/ACCIDENT",
        ],
    }

    model_output = _resolve(args.model_output)
    report_output = _resolve(args.report_output)
    predictions_output = _resolve(args.predictions_output)
    for output in (model_output, report_output, predictions_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": features.shape[2],
            "channels": args.channels,
            "dropout": args.dropout,
            "feature_names": list(feature_names),
            "standardizer": standardizer.to_dict(),
            "threshold": args.threshold,
        },
        model_output,
    )
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_predictions(predictions_output, payload, test_indices, test_probabilities, args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _predict(
    model: TemporalConvClassifier,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def _group_metrics(
    values: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    return {
        value: binary_classification_metrics(
            targets[values == value],
            probabilities[values == value],
            threshold,
        )
        for value in sorted(set(values.tolist()))
    }


def _write_predictions(
    output_path: Path,
    payload,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    fieldnames = ["path", "label", "target", "collision_type", "region", "quality", "probability", "prediction"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in zip(indices, probabilities):
            writer.writerow(
                {
                    "path": str(payload["path"][index]),
                    "label": str(payload["label"][index]),
                    "target": int(payload["target"][index]),
                    "collision_type": str(payload["collision_type"][index]),
                    "region": str(payload["region"][index]),
                    "quality": str(payload["quality"][index]),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                }
            )


def _validate_split(
    targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split_field: str,
) -> None:
    if not len(train_indices) or not len(test_indices):
        raise SystemExit(f"Sequence data must contain train and test rows for {split_field}")
    if set(np.unique(targets[train_indices])) != {0, 1}:
        raise SystemExit("Training split must contain normal and incident windows")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
