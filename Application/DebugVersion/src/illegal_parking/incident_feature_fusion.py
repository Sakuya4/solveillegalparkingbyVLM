from __future__ import annotations

from illegal_parking.incident_baseline import NON_FEATURE_FIELDS, select_numeric_feature_names


WINDOW_KEY_FIELDS = ("path", "label", "start_frame", "end_frame")
CONSISTENCY_FIELDS = (
    "target",
    "collision_type",
    "region",
    "quality",
    "day_time",
    "iid_split",
    "geographic_split",
)


def merge_feature_rows(
    motion_rows: list[dict[str, str]],
    trajectory_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not motion_rows or not trajectory_rows:
        raise ValueError("Both feature sets must contain rows")
    motion_features = select_numeric_feature_names(motion_rows[0])
    trajectory_features = select_numeric_feature_names(trajectory_rows[0])
    motion_index = _index_rows(motion_rows)
    trajectory_index = _index_rows(trajectory_rows)
    if set(motion_index) != set(trajectory_index):
        missing_motion = len(set(trajectory_index) - set(motion_index))
        missing_trajectory = len(set(motion_index) - set(trajectory_index))
        raise ValueError(
            "Window keys do not match: "
            f"missing_motion={missing_motion}, missing_trajectory={missing_trajectory}"
        )

    merged: list[dict[str, str]] = []
    for key in motion_index:
        motion = motion_index[key]
        trajectory = trajectory_index[key]
        for field in CONSISTENCY_FIELDS:
            if motion.get(field) != trajectory.get(field):
                raise ValueError(f"Feature rows disagree on {field} for {key}")
        row = {
            name: value
            for name, value in motion.items()
            if name in NON_FEATURE_FIELDS
        }
        row.update({f"motion_{name}": motion[name] for name in motion_features})
        row.update({f"trajectory_{name}": trajectory[name] for name in trajectory_features})
        merged.append(row)
    return merged


def _index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        try:
            key = tuple(row[field] for field in WINDOW_KEY_FIELDS)
        except KeyError as exc:
            raise ValueError(f"Missing window key field: {exc.args[0]}") from exc
        if key in index:
            raise ValueError(f"Duplicate window key: {key}")
        index[key] = row
    return index
