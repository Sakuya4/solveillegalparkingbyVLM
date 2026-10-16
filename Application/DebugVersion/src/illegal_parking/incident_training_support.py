from __future__ import annotations

from collections import Counter
import math

from .incident_presentation import normalized_presentation_times
from .incident_presentation_events import continuous_presentation_windows


def fit_source_records(rows, media):
    """Select source identities from fit membership, never from model scores."""
    sources = {item["path"]: item for item in media}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["path"], []).append(row)
    selected = []
    for name, windows in sorted(grouped.items()):
        memberships = {}
        for split in ("iid", "geographic"):
            partitions = {row[split + "_partition"] for row in windows}
            if len(partitions) != 1:
                raise ValueError("A source cannot cross partitions within a split")
            if partitions == {"fit"}:
                memberships[split] = "fit"
        if not memberships:
            continue
        anchors = {(row["supplied_event_frame"], row["event_time_sec"]) for row in windows}
        if len(anchors) != 1:
            raise ValueError("Source annotation anchors must agree")
        frame, event = anchors.pop()
        if name not in sources:
            raise ValueError("Fit source lacks audited media identity")
        selected.append({**sources[name], "fit_splits": sorted(memberships),
                         "supplied_event_frame": frame, "event_time_sec": event,
                         "anchored_windows": [{"label": row["label"], "indices": row["fixed_indices"],
                                               "end_time_sec": row["fixed_times_sec"][-1]} for row in windows]})
    return selected


def endpoint_label(end_time, event_time):
    if not math.isfinite(end_time) or not math.isfinite(event_time) or min(end_time, event_time) < 0:
        raise ValueError("Endpoint and event times must be finite and nonnegative")
    return "prefix" if end_time < event_time - 1 else "target" if end_time <= event_time + 3 else "post"


def audit_training_source(source, timestamps):
    times = normalized_presentation_times(timestamps)
    frame = source["supplied_event_frame"]
    if not isinstance(frame, int) or not 0 <= frame < len(times):
        raise ValueError("Supplied annotation must lie within timestamp support")
    event = float(times[frame])
    if abs(event - source["event_time_sec"]) > 1e-4:
        raise ValueError("Annotation presentation time changed")
    windows, rejected = continuous_presentation_windows(timestamps)
    contexts = [{"path": source["path"], "start": window["start"],
                 "end_time_sec": window["end_time_sec"],
                 "label": endpoint_label(window["end_time_sec"], event),
                 "indices": window["indices"]} for window in windows]
    counts = Counter(row["label"] for row in contexts)
    starts = {row["start"] for row in contexts}
    targets = [row for row in contexts if row["label"] == "target"]
    positive_starts = {row["start"] for row in targets}
    continuous_ids = {tuple(row["indices"]) for row in contexts}
    anchored = source["anchored_windows"]
    summary = {"path": source["path"], "valid_contexts": len(contexts), "invalid_contexts": len(rejected),
               "prefix": counts["prefix"], "target": counts["target"], "post": counts["post"],
               "both_classes": bool(counts["prefix"] and counts["target"]),
               "gate1_support": bool(targets),
               "gate2_structural_support": any(row["start"] - 15 in starts for row in targets),
               "two_target_contexts_adjacent": any(start - 15 in positive_starts for start in positive_starts),
               "anchored_windows": len(anchored),
               "anchored_exact_continuous_matches": sum(tuple(row["indices"]) in continuous_ids for row in anchored),
               "anchored_endpoint_label_disagreements": sum(endpoint_label(row["end_time_sec"], event) !=
                                                            ("target" if row["label"] == "incident" else "prefix") for row in anchored)}
    return summary, contexts
