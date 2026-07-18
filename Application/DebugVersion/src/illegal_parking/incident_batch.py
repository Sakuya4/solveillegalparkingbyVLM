from __future__ import annotations

from pathlib import Path

from .accident_dataset import AccidentClip, stratified_sample_clips


def select_evaluation_clips(
    clips: list[AccidentClip],
    split_scheme: str,
    max_clips: int | None,
    seed: int,
) -> list[AccidentClip]:
    if split_scheme not in {"iid", "geographic"}:
        raise ValueError("split_scheme must be 'iid' or 'geographic'")
    split_field = "iid_split" if split_scheme == "iid" else "geographic_split"
    test_clips = [clip for clip in clips if getattr(clip, split_field) == "test"]
    if not test_clips:
        raise ValueError(f"No test clips found for {split_scheme} split")
    if max_clips is not None and max_clips < len(test_clips):
        test_clips = stratified_sample_clips(test_clips, max_clips, seed=seed)
    return sorted(test_clips, key=lambda clip: clip.relative_path.as_posix())


def report_filename(relative_path: Path) -> str:
    safe_parts = [part.replace(" ", "_") for part in relative_path.with_suffix("").parts]
    return "__".join(safe_parts) + ".json"
