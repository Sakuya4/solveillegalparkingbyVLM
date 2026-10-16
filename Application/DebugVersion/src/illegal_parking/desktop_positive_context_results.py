from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .desktop_presentation_results import number


def add_positive_context_tab(tabs, snapshot, table):
    page = QWidget()
    layout = QVBoxLayout(page)
    study = snapshot.get("positive_context_events")
    layout.addWidget(QLabel("New event cohort: 100 IID / 100 geographic clips | PTS 7.5 Hz | hop 15 / gate 2 | TRAIN caps: prefix 0.2, episodes 0.5"))
    rows = []
    if study:
        indexed = {(row["model"], row["policy"]): row for row in study["policies"]}
        for pair in study["paired_recall_contrasts"]:
            model = pair["model"]
            onset, capped = (indexed[(model, policy)] for policy in ("onset_only", "positive_context"))
            interval = pair["paired_interval"]
            rows.append([model.replace("geographic", "geo"), number(capped["threshold"]),
                         number(onset["test"]["event_recall"]), number(capped["test"]["event_recall"]),
                         number(capped["pre_event"]["macro_positive_fraction"]), number(capped["test"]["off_target_episodes_per_clip"]),
                         "Yes" if capped["episode_budget_exceeded"] or capped["pre_event_cap_exceeded"] else "No",
                         f'[{number(interval["lower_95"], signed=True)}, {number(interval["upper_95"], signed=True)}]'])
    result_table = table(["Model / training / split", "Threshold", "Onset recall", "Capped recall", "Prefix + fraction", "Off-target / clip", "Cap exceeded", "Capped-onset 95% CI"], rows)
    result_table.setObjectName("positiveContextResults")
    result_table.verticalHeader().setDefaultSectionSize(27)
    layout.addWidget(result_table, 5)
    controls = []
    if study:
        for split, definitions in study["controls"].items():
            for name, row in definitions.items():
                controls.append([split, name.replace("_", " "), row["test"]["clip_count"],
                                 row["pre_event"]["clips_with_support"], row["pre_event"]["clips_without_support"],
                                 number(row["test"]["event_recall"]), number(row["pre_event"]["macro_positive_fraction"]),
                                 number(row["test"]["off_target_episodes_per_clip"])])
    control_table = table(["Split", "Control", "Clips", "Prefix support", "No prefix", "Recall", "Prefix + fraction", "Off-target / clip"], controls)
    control_table.setObjectName("positiveContextControls")
    control_table.verticalHeader().setDefaultSectionSize(27)
    layout.addWidget(control_table, 3)
    if not study:
        layout.addWidget(QLabel("Pending"))
    tabs.addTab(page, "Positive-context caps")
