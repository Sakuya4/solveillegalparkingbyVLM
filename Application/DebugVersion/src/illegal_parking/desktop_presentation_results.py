from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def number(value, *, signed=False):
    return "NA" if value is None else format(value, "+.3f" if signed else ".3f")


def add_presentation_tabs(tabs, snapshot, table):
    page = QWidget()
    layout = QVBoxLayout(page)
    study = snapshot["presentation_windows"]
    rows = [["Fixed-time" if row["arm"] == "fixed" else "Source-frame", row["feature"], row["split"],
             row["test_clips"], number(row["test"]["f1"]), number(row["test"]["false_positive_rate"]),
             number(row["auroc"]), number(row["average_precision"])] for row in study["models"]]
    window_table = table(["Training policy", "Features", "Split", "Clips", "Window F1", "Window FPR", "AUROC", "AP"], rows)
    window_table.setObjectName("ptsWindowResults")
    layout.addWidget(window_table)
    layout.addWidget(table(["Features", "Split", "Fixed minus source F1", "Pointwise 95% interval", "Clips"], [
        [row["feature"], row["split"], number(row["difference"], signed=True),
         f'[{number(row["lower_95"], signed=True)}, {number(row["upper_95"], signed=True)}]', row["clusters"]]
        for row in study["paired_f1_contrasts"]]))
    layout.setStretch(0, 5)
    layout.setStretch(1, 3)
    tabs.addTab(page, "PTS windows")

    page = QWidget()
    layout = QVBoxLayout(page)
    study = snapshot["presentation_events"]
    rows = []
    if study:
        for row in study["models"]:
            outcome = row["test"]
            rows.append(["Fixed-time" if row["arm"] == "fixed" else "Source-frame", row["feature"], row["split"],
                         row["test_clips"], number(row["selected_threshold"]), number(outcome["event_recall"]) if outcome else "NA",
                         number(outcome["trigger_delay_sec"]["median"]) if outcome else "NA",
                         number(outcome["off_target_episodes_per_clip"]) if outcome else "NA",
                         "Yes" if row["test_budget_exceeded"] else "No" if outcome else "NA"])
    event_table = table(["Training policy", "Features", "Split", "Clips", "Threshold", "Event recall", "Median delay / s", "Off-target / clip", "Budget exceeded"], rows)
    event_table.setObjectName("ptsEventResults")
    layout.addWidget(event_table)
    if study:
        pairs = study["paired_recall_contrasts"]
        layout.addWidget(table(["Features", "Split", "Fixed minus source recall", "Pointwise 95% interval"], [
            [row["feature"], row["split"], number(row["paired_interval"]["difference"], signed=True) if row["paired_interval"] else "NA",
             f'[{number(row["paired_interval"]["lower_95"], signed=True)}, {number(row["paired_interval"]["upper_95"], signed=True)}]' if row["paired_interval"] else "NA"] for row in pairs]))
        layout.setStretch(0, 5)
        layout.setStretch(1, 3)
    else:
        layout.addWidget(QLabel("Pending"))
    tabs.addTab(page, "PTS events")

    page = QWidget()
    layout = QVBoxLayout(page)
    study = snapshot["presentation_controls"]
    rows = []
    if study:
        for split, cohorts in study["controls"].items():
            for name, outcome in cohorts["test"].items():
                rows.append([split, "Clock-only positive" if name == "clock_only_positive" else "Always negative",
                             outcome["clip_count"], number(outcome["event_recall"]), number(outcome["trigger_delay_sec"]["median"]),
                             number(outcome["off_target_episodes_per_clip"]), outcome["rejected_windows"]])
    control_table = table(["Split", "Nonvisual control", "Clips", "Event recall", "Median delay / s", "Off-target / clip", "Rejected contexts"], rows)
    control_table.setObjectName("ptsControlResults")
    layout.addWidget(control_table)
    if study:
        layout.addWidget(table(["Model / training / split", "Threshold", "Positive windows", "Recall minus clock", "Pointwise 95% interval"], [
            [row["model"], number(row["selected_threshold"]), number(row.get("positive_window_fraction")),
             number(row["paired_recall_vs_clock"]["difference"], signed=True) if row["paired_recall_vs_clock"] else "NA",
             f'[{number(row["paired_recall_vs_clock"]["lower_95"], signed=True)}, {number(row["paired_recall_vs_clock"]["upper_95"], signed=True)}]' if row["paired_recall_vs_clock"] else "NA"] for row in study["model_comparisons"]]))
        layout.setStretch(0, 3)
        layout.setStretch(1, 5)
    else:
        layout.addWidget(QLabel("Pending"))
    tabs.addTab(page, "Clock controls")
