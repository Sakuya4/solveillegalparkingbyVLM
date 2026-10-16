from PySide6.QtWidgets import QComboBox, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .desktop_presentation_results import number


def add_event_diagnostic_tab(tabs, snapshot, table):
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel("TRAIN holdouts: 94 IID / 83 geographic clips | Within-clip ranking support: 59 / 40 | Existing weights and thresholds"))
    selector = QComboBox()
    selector.setObjectName("eventDiagnosticView")
    selector.addItems(["Score ranking", "Controller outcomes", "Paired ranking intervals"])
    layout.addWidget(selector)
    views = QStackedWidget()
    views.setObjectName("eventDiagnosticViews")
    study = snapshot.get("event_discrimination")
    rows, policies, pairs = [], [], []
    if study:
        for row in study["models"]:
            value = row["summary"]
            rows.append([row["model"].replace("geographic", "geo"), value["clip_count"], value["support"]["both"],
                         number(value["pooled"]["auroc"]), number(value["within_clip_macro"]["auroc"]),
                         number(value["within_clip_macro"]["average_precision"]), number(value["within_clip_macro"]["target_prevalence"]),
                         value["gate2_attainable_clips"]])
            for policy, outcome in value["policy_decomposition"].items():
                policies.append([row["model"].replace("geographic", "geo"), policy.replace("_", " "),
                                 *(outcome[key] for key in ("detected", "structurally_unreachable", "no_target_positive_pair", "onset_suppression"))])
        for row in study["paired_ranking_contrasts"]:
            interval = row["interval"]
            pairs.append([row["split"], row["feature"], row["metric"], interval["clips"], number(interval["difference"], signed=True),
                          f'[{number(interval["lower_95"], signed=True)}, {number(interval["upper_95"], signed=True)}]'])
    for name, headers, values in (
        ("eventDiagnosticRanking", ["Model / training / split", "Clips", "Support", "Pooled AUC", "Within-clip AUC", "Within-clip AP", "Macro target fraction", "Gate-2 attainable"], rows),
        ("eventDiagnosticController", ["Model / training / split", "Policy", "Detected", "Unreachable", "No positive pair", "Onset suppression"], policies),
        ("eventDiagnosticPairs", ["Split", "Features", "Metric", "Clips", "Fixed minus source", "Pointwise 95% interval"], pairs)):
        widget = table(headers, values)
        widget.setObjectName(name)
        views.addWidget(widget)
    selector.currentIndexChanged.connect(views.setCurrentIndex)
    layout.addWidget(views, 1)
    tabs.addTab(page, "TRAIN diagnostics")
