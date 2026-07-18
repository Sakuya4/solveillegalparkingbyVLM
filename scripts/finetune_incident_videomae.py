from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import (
    build_temporal_windows,
    load_accident_manifest,
    stratified_sample_clips,
)
from illegal_parking.incident_baseline import (
    binary_classification_metrics,
    grouped_calibration_indices,
    select_threshold_for_target_fpr,
)
from illegal_parking.incident_videomae import (
    configure_videomae_finetuning,
    load_compatible_videomae_classifier,
    read_uniform_video_window,
    videomae_optimizer_groups,
)


class VideoWindowDataset(Dataset):
    def __init__(
        self,
        records: list[tuple],
        indices: np.ndarray,
        dataset_root: Path,
        processor,
        num_frames: int,
        cache_dir: Path | None,
    ) -> None:
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.dataset_root = dataset_root
        self.processor = processor
        self.num_frames = num_frames
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> tuple[torch.Tensor, int, int]:
        record_index = int(self.indices[local_index])
        _clip, window = self.records[record_index]
        cache_path = self._cache_path(window)
        if cache_path is not None and cache_path.is_file():
            pixel_values = np.load(cache_path, allow_pickle=False).astype(np.float32)
        else:
            frames = read_uniform_video_window(
                self.dataset_root / window.relative_path,
                window.start_frame,
                window.end_frame,
                num_frames=self.num_frames,
            )
            pixel_values = self.processor([frames], return_tensors="np")["pixel_values"][0]
            if cache_path is not None:
                np.save(cache_path, pixel_values.astype(np.float16), allow_pickle=False)
        return torch.from_numpy(pixel_values), int(window.label == "incident"), record_index

    def _cache_path(self, window) -> Path | None:
        if self.cache_dir is None:
            return None
        key = (
            f"{window.relative_path.as_posix()}:{window.start_frame}:"
            f"{window.end_frame}:{self.num_frames}"
        )
        return self.cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.npy"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Fine-tune the last VideoMAE encoder block on ACCIDENT temporal windows."
    )
    parser.add_argument("--metadata", default="data/raw/accident/full/extracted/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident/full/extracted")
    parser.add_argument("--model-id", default="MCG-NJU/videomae-small-finetuned-kinetics")
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--max-clips", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--trainable-encoder-blocks", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--negative-gap-frames", type=int, default=8)
    parser.add_argument("--target-train-fpr", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default="data/processed/accident/videomae_pixel_cache_500",
    )
    parser.add_argument("--model-output", default="models/videomae_finetuned_iid.pt")
    parser.add_argument(
        "--report-output", default="outputs/accident/videomae_finetuned_iid_report.json"
    )
    parser.add_argument(
        "--predictions-output",
        default="outputs/accident/videomae_finetuned_iid_predictions.csv",
    )
    args = parser.parse_args()
    _validate_args(args)
    _set_seed(args.seed)

    from transformers import VideoMAEImageProcessor

    device = torch.device(args.device)
    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    cache_dir = None if args.no_cache else _resolve(args.cache_dir)
    clips = load_accident_manifest(metadata_path)
    clips = stratified_sample_clips(clips, args.max_clips, seed=args.sampling_seed)
    records = [
        (clip, window)
        for clip in clips
        for window in build_temporal_windows(
            clip,
            window_frames=args.window_frames,
            negative_gap_frames=args.negative_gap_frames,
        )
    ]
    split_field = "iid_split" if args.split_scheme == "iid" else "geographic_split"
    splits = np.asarray([getattr(clip, split_field) for clip, _window in records])
    targets = np.asarray([int(window.label == "incident") for _clip, window in records])
    paths = np.asarray([window.relative_path.as_posix() for _clip, window in records])
    train_indices = np.flatnonzero(splits == "train")
    test_indices = np.flatnonzero(splits == "test")
    fit_local, calibration_local = grouped_calibration_indices(
        paths[train_indices],
        fraction=args.calibration_fraction,
        seed=args.seed,
    )
    fit_indices = train_indices[fit_local]
    calibration_indices = train_indices[calibration_local]
    _validate_partitions(targets, fit_indices, calibration_indices, test_indices)

    processor = VideoMAEImageProcessor.from_pretrained(args.model_id)
    model, revision = load_compatible_videomae_classifier(args.model_id, args.device)
    parameter_summary = configure_videomae_finetuning(
        model,
        num_labels=2,
        trainable_encoder_blocks=args.trainable_encoder_blocks,
    )
    model.to(device)
    datasets = {
        "fit": VideoWindowDataset(
            records, fit_indices, dataset_root, processor, args.num_frames, cache_dir
        ),
        "calibration": VideoWindowDataset(
            records, calibration_indices, dataset_root, processor, args.num_frames, cache_dir
        ),
        "test": VideoWindowDataset(
            records, test_indices, dataset_root, processor, args.num_frames, cache_dir
        ),
    }
    loader_generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        datasets["fit"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    class_counts = np.bincount(targets[fit_indices], minlength=2)
    class_weights = torch.tensor(
        [len(fit_indices) / (2 * count) for count in class_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        videomae_optimizer_groups(
            model,
            encoder_learning_rate=args.learning_rate,
            head_learning_rate=args.head_learning_rate,
        ),
        weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses: list[float] = []
    calibration_losses: list[float] = []
    best_calibration_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for pixel_values, labels, _indices in fit_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            total_samples += len(labels)
        losses.append(total_loss / total_samples)
        calibration_probabilities = _predict(
            model, datasets["calibration"], args.batch_size, device, amp_enabled
        )
        calibration_loss = _binary_log_loss(
            targets[calibration_indices], calibration_probabilities
        )
        calibration_losses.append(calibration_loss)
        if calibration_loss < best_calibration_loss - 1e-5:
            best_calibration_loss = calibration_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch}/{args.epochs} loss={losses[-1]:.6f} "
            f"calibration_loss={calibration_loss:.6f}",
            flush=True,
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stop={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)

    fit_probabilities = _predict(model, datasets["fit"], args.batch_size, device, amp_enabled)
    calibration_probabilities = _predict(
        model, datasets["calibration"], args.batch_size, device, amp_enabled
    )
    test_probabilities = _predict(model, datasets["test"], args.batch_size, device, amp_enabled)
    threshold = select_threshold_for_target_fpr(
        targets[calibration_indices],
        calibration_probabilities,
        args.target_train_fpr,
    )
    elapsed_sec = time.perf_counter() - started
    report = {
        "model": "videomae_small_partial_finetune",
        "model_id": args.model_id,
        "model_revision": revision,
        "split_scheme": args.split_scheme,
        "clip_count": len(clips),
        "window_count": len(records),
        "fit_windows": len(fit_indices),
        "calibration_windows": len(calibration_indices),
        "test_windows": len(test_indices),
        "sampling_seed": args.sampling_seed,
        "training_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "num_frames": args.num_frames,
        "window_frames": args.window_frames,
        "negative_gap_frames": args.negative_gap_frames,
        "parameters": parameter_summary,
        "train_loss": losses,
        "calibration_loss": calibration_losses,
        "best_epoch": best_epoch,
        "epochs_completed": len(losses),
        "early_stopping_patience": args.early_stopping_patience,
        "decision_threshold": {
            "source": "grouped_train_holdout_fpr_calibration",
            "effective": threshold,
            "target_train_fpr": args.target_train_fpr,
            "calibration_fraction": args.calibration_fraction,
        },
        "fit": binary_classification_metrics(
            targets[fit_indices], fit_probabilities, threshold
        ),
        "calibration": binary_classification_metrics(
            targets[calibration_indices], calibration_probabilities, threshold
        ),
        "test": binary_classification_metrics(
            targets[test_indices], test_probabilities, threshold
        ),
        "test_by_collision_type": _group_metrics(
            records, test_indices, targets[test_indices], test_probabilities, threshold,
            field="collision_type",
        ),
        "test_by_quality": _group_metrics(
            records, test_indices, targets[test_indices], test_probabilities, threshold,
            field="quality",
        ),
        "device": str(device),
        "amp": amp_enabled,
        "elapsed_sec": elapsed_sec,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "annotation_contract": (
            "Accident timing defines temporal-window labels only; bbox, type, region, and quality "
            "are excluded from model inputs. Test windows are never used for threshold selection."
        ),
        "sources": [
            "https://github.com/MCG-NJU/VideoMAE",
            "https://huggingface.co/docs/transformers/model_doc/videomae",
            "https://huggingface.co/docs/transformers/tasks/video_classification",
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
            "model_id": args.model_id,
            "num_labels": 2,
            "threshold": threshold,
            "num_frames": args.num_frames,
            "window_frames": args.window_frames,
            "trainable_encoder_blocks": args.trainable_encoder_blocks,
            "head_learning_rate": args.head_learning_rate,
            "encoder_learning_rate": args.learning_rate,
            "best_epoch": best_epoch,
        },
        model_output,
    )
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_predictions(
        predictions_output,
        records,
        test_indices,
        test_probabilities,
        threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _predict(
    model,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    indexed_probabilities: list[tuple[int, float]] = []
    with torch.inference_mode():
        for pixel_values, _labels, indices in loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                logits = model(pixel_values=pixel_values).logits
                probabilities = torch.softmax(logits, dim=1)[:, 1]
            indexed_probabilities.extend(
                (int(index), float(probability))
                for index, probability in zip(indices, probabilities.cpu())
            )
    by_index = dict(indexed_probabilities)
    return np.asarray([by_index[int(index)] for index in dataset.indices], dtype=np.float64)


def _binary_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    labels = np.asarray(targets, dtype=np.float64)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _group_metrics(
    records: list[tuple],
    indices: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    field: str,
) -> dict[str, dict]:
    values = np.asarray([getattr(records[int(index)][0], field) for index in indices])
    return {
        value: binary_classification_metrics(
            targets[values == value], probabilities[values == value], threshold
        )
        for value in sorted(set(values.tolist()))
    }


def _write_predictions(
    output_path: Path,
    records: list[tuple],
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    fieldnames = [
        "path", "label", "target", "collision_type", "region", "quality",
        "start_frame", "end_frame", "probability", "prediction",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in zip(indices, probabilities):
            clip, window = records[int(index)]
            writer.writerow({
                "path": window.relative_path.as_posix(),
                "label": window.label,
                "target": int(window.label == "incident"),
                "collision_type": clip.collision_type,
                "region": clip.region,
                "quality": clip.quality,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "probability": float(probability),
                "prediction": int(probability >= threshold),
            })


def _validate_args(args) -> None:
    for name in (
        "max_clips", "epochs", "batch_size", "num_frames", "window_frames",
        "early_stopping_patience",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.num_frames > args.window_frames:
        raise SystemExit("--num-frames cannot exceed --window-frames")
    if not 0.0 <= args.target_train_fpr <= 1.0:
        raise SystemExit("--target-train-fpr must be between zero and one")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be between zero and one")


def _validate_partitions(
    targets: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    for name, indices in (
        ("fit", fit_indices),
        ("calibration", calibration_indices),
        ("test", test_indices),
    ):
        if not len(indices) or set(np.unique(targets[indices])) != {0, 1}:
            raise SystemExit(f"{name} partition must contain normal and incident windows")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
