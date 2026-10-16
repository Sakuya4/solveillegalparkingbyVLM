from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Ranking labels and finite scores must be aligned")
    if np.any((labels != 0) & (labels != 1)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Binary labels and bounded probabilities are required")
    prevalence = float(labels.mean()) if len(labels) else None
    supported = len(np.unique(labels)) == 2
    ap = float(average_precision_score(labels, scores)) if supported else None
    return {"auroc": float(roc_auc_score(labels, scores)) if supported else None,
            "average_precision": ap, "target_prevalence": prevalence,
            "ap_minus_prevalence": ap - prevalence if supported else None}


def context_arrays(clip, contexts):
    event = float(clip["annotation_pts_sec"])
    starts = np.asarray([int(row["start_slot"]) for row in contexts], dtype=int)
    times = np.asarray([float(row["end_pts_sec"]) for row in contexts])
    scores = np.asarray([float(row["probability"]) for row in contexts])
    if (not math.isfinite(event) or event < 0 or np.any(starts < 0) or np.any(starts % 15) or np.any(np.diff(starts) <= 0)
            or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1))
            or len(contexts) != int(clip["valid_contexts"])):
        raise ValueError("Context clock, count, grid or scores violate the frozen diagnostic contract")
    prefix, target = times < event - 1, (times >= event - 1) & (times <= event + 3)
    return starts, times, scores, prefix, target


def policy_decomposition(starts, times, scores, target, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A bounded finite inherited threshold is required")
    run = 0
    episodes, target_pairs = [], 0
    attainable = False
    for index, (start, score) in enumerate(zip(starts, scores)):
        adjacent = index > 0 and start == starts[index - 1] + 15
        if not adjacent:
            run = 0
        if adjacent and target[index]:
            attainable = True
            target_pairs += int(score >= threshold and scores[index - 1] >= threshold)
        run = run + 1 if score >= threshold else 0
        if run == 2:
            episodes.append(index)
    matched = [index for index in episodes if target[index]]
    category = ("detected" if matched else "structurally_unreachable" if not attainable else
                "no_target_positive_pair" if target_pairs == 0 else "onset_suppression")
    return {"category": category, "detected": bool(matched), "target_positive_pairs": target_pairs,
            "episodes": len(episodes), "matched_episodes": len(matched),
            "trigger_pts_sec": float(times[matched[0]]) if matched else None}


def clip_diagnostic(clip, contexts, thresholds):
    starts, times, scores, prefix, target = context_arrays(clip, contexts)
    supported = bool(prefix.any() and target.any())
    reason = ("both" if supported else "no_prefix" if target.any() else "no_target" if prefix.any() else "neither")
    metrics = ranking_metrics(target[prefix | target], scores[prefix | target])
    metrics["max_target_minus_max_prefix"] = float(scores[target].max() - scores[prefix].max()) if supported else None
    policies = {name: policy_decomposition(starts, times, scores, target, threshold) for name, threshold in thresholds.items()}
    gate2 = bool(np.any(target[1:] & (np.diff(starts) == 15))) if len(starts) > 1 else False
    return {"path": clip["path"], "source_sha256": clip["source_sha256"], "annotation_pts_sec": float(clip["annotation_pts_sec"]),
            "contexts": {"valid": len(contexts), "prefix": int(prefix.sum()), "target": int(target.sum()), "post": int((~(prefix | target)).sum()), "rejected": int(clip["rejected_contexts"])},
            "support": reason, "within_clip": metrics if supported else None,
            "gate1_attainable": bool(target.any()), "gate2_attainable": gate2, "policies": policies}


def distribution(values):
    return {"count": len(values), "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "minimum": float(np.min(values)) if len(values) else None, "maximum": float(np.max(values)) if len(values) else None}


def model_diagnostic(clips, contexts, thresholds):
    outcomes = [clip_diagnostic(clip, contexts[clip["path"]], thresholds) for clip in clips]
    labels, probabilities = [], []
    phases = {name: [] for name in ("prefix", "target", "post")}
    for clip in clips:
        _, _, scores, prefix, target = context_arrays(clip, contexts[clip["path"]])
        labels.extend(target[prefix | target].astype(int).tolist())
        probabilities.extend(scores[prefix | target].tolist())
        for name, mask in (("prefix", prefix), ("target", target), ("post", ~(prefix | target))):
            phases[name].extend(scores[mask].tolist())
    supported = [row for row in outcomes if row["support"] == "both"]
    fields = ("auroc", "average_precision", "target_prevalence", "ap_minus_prevalence", "max_target_minus_max_prefix")
    summary = {"clip_count": len(clips), "empty_context_clips": sum(not row["contexts"]["valid"] for row in outcomes),
               "support": {name: sum(row["support"] == name for row in outcomes) for name in ("both", "no_prefix", "no_target", "neither")},
               "pooled": ranking_metrics(labels, probabilities),
               "constant_control": ranking_metrics(labels, [.5] * len(labels)),
               "within_clip_macro": {field: float(np.mean([row["within_clip"][field] for row in supported])) if supported else None for field in fields},
               "score_distributions": {name: distribution(values) for name, values in phases.items()},
               "gate1_attainable_clips": sum(row["gate1_attainable"] for row in outcomes),
               "gate2_attainable_clips": sum(row["gate2_attainable"] for row in outcomes),
               "policy_decomposition": {name: {category: sum(row["policies"][name]["category"] == category for row in outcomes)
                   for category in ("detected", "structurally_unreachable", "no_target_positive_pair", "onset_suppression")} for name in thresholds}}
    return summary, outcomes


def paired_ranking_interval(fixed, source, metric, *, repeats=2000, seed=42):
    def supported(rows):
        return {row["path"]: row["within_clip"][metric] for row in rows if row["support"] == "both"}
    a, b = supported(fixed), supported(source)
    if set(a) != set(b):
        raise ValueError("Paired rankings require identical supported clip identities")
    names = sorted(a)
    if len(names) < 2:
        return {"clips": len(names), "difference": None, "lower_95": None, "upper_95": None}
    values = np.asarray([a[name] - b[name] for name in names])
    generator = np.random.default_rng(seed)
    draws = values[generator.integers(0, len(names), (repeats, len(names)))].mean(axis=1)
    return {"clips": len(names), "difference": float(values.mean()),
            "lower_95": float(np.percentile(draws, 2.5)), "upper_95": float(np.percentile(draws, 97.5))}
