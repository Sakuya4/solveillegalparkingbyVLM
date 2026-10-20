from PySide6.QtWidgets import QComboBox, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .desktop_presentation_results import number


def comparison_rows(study, latency):
    models, pairs, costs = [], [], []
    if study:
        for row in study["models"]:
            arm, feature, split = row["model"].split("_")
            result = row["test"]
            summary, prefix = (result["summary"], result["pre_event"]) if result else ({}, {})
            exceeded = row["test_episode_cap_exceeded"] or row["test_prefix_cap_exceeded"]
            models.append([arm, feature, "geo" if split == "geographic" else split, number(row["selected_threshold"]),
                           number(summary.get("event_recall")), number(summary.get("off_target_episodes_per_clip")),
                           number(prefix.get("macro_positive_fraction")), prefix.get("clips_with_support", "NA"),
                           number(summary.get("trigger_delay_sec", {}).get("median")), "Unselected" if result is None else "Exceeded" if exceeded else "Met"])
        for row in study["contrasts"]:
            interval = row["paired_interval"]
            pairs.append([row["feature"], row["split"],
                          number(100 * interval["difference"], signed=True) if interval else "NA",
                          f'[{100 * interval["lower_95"]:+.2f}, {100 * interval["upper_95"]:+.2f}]' if interval else "NA",
                          interval["clusters"] if interval else "NA"])
    if latency:
        for row in latency["results"]:
            arm, feature, split = row["model"].split("_")
            costs.append([arm, feature, "geo" if split == "geographic" else split,
                          number(row["median_ms"]), number(row["p95_ms"]), row["calls"]])
    return models, pairs, costs


def add_context_training_tab(tabs, snapshot, table):
    page = QWidget()
    layout = QVBoxLayout(page)
    label = QLabel("100 IID / 100 geographic clips | PTS 7.5 Hz | hop 15 / gate 2 | TRAIN caps: prefix 0.2, episodes 0.5")
    label.setWordWrap(True)
    layout.addWidget(label)
    selector = QComboBox()
    selector.setObjectName("contextTrainingView")
    selector.addItems(["Operating points", "Paired recall intervals", "Temporal CPU cost"])
    layout.addWidget(selector)
    views = QStackedWidget()
    views.setObjectName("contextTrainingViews")
    models, pairs, costs = comparison_rows(snapshot.get("continuous_context_training"), snapshot.get("continuous_context_latency"))
    for name, headers, values in (
        ("contextTrainingModels", ["Context", "Features", "Split", "Threshold", "Event recall", "Off-target / clip", "Prefix fraction", "Prefix support", "Median delay (s)", "Test caps"], models),
        ("contextTrainingPairs", ["Features", "Split", "Recall change (pp)", "Pointwise 95% interval (pp)", "Clips"], pairs),
        ("contextTrainingLatency", ["Context", "Features", "Split", "CPU median (ms)", "CPU p95 (ms)", "Timed calls"], costs)):
        widget = table(headers, values)
        widget.setObjectName(name)
        views.addWidget(widget)
    selector.currentIndexChanged.connect(views.setCurrentIndex)
    layout.addWidget(views, 1)
    tabs.addTab(page, "Context training")
