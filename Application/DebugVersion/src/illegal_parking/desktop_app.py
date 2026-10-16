from __future__ import annotations

import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QFont, QFontDatabase, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QSpinBox,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSlider, QSizePolicy, QSplitter, QStackedWidget, QStyle, QTableWidget, QTableWidgetItem,
    QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from .onnx_runtime import runtime_capabilities
from .project_snapshot import build_project_snapshot
from .incident_vlm_evidence import REVIEW_POLICY_VERSION, review_disposition
from .cctv_capture import load_camera_observation, redacted_camera_url


def configure_application_fonts(application: QApplication) -> None:
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ("segoeui.ttf", "seguisb.ttf", "msjh.ttc"):
            QFontDatabase.addApplicationFont(str(fonts / filename))
        application.setFont(QFont("Segoe UI", 10))


def worker_python() -> str:
    executable = Path(sys.executable)
    console_python = executable.with_name("python.exe")
    if executable.stem.lower() == "pythonw" and console_python.is_file():
        return str(console_python)
    return str(executable)


class VideoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = None
        self.frame_count = 0
        self.frame_index = 0
        self.current_image = None
        self.path = None
        self.preview = QLabel("No video selected")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 220)
        self.preview.setObjectName("videoPreview")
        self.play = QPushButton()
        self.play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play.setToolTip("Play or pause")
        self.play.setFixedWidth(40)
        self.play.clicked.connect(self.toggle_playback)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.seek)
        self.position = QLabel("00:00 / 00:00")
        controls = QHBoxLayout()
        controls.addWidget(self.play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.position)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview, 1)
        layout.addLayout(controls)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)

    def load(self, path: Path):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video: {path.name}")
        self.release()
        self.capture = capture
        self.path = path
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = max(1.0, capture.get(cv2.CAP_PROP_FPS))
        self.timer.setInterval(max(1, round(1000 / self.fps)))
        self.slider.setRange(0, max(0, self.frame_count - 1))
        self.play.setEnabled(True)
        self.slider.setEnabled(True)
        self.seek(0)

    def seek(self, index: int):
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.advance()

    def advance(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.timer.stop()
            self.play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            return
        self.frame_index = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.display_frame(frame)
        self.slider.setValue(self.frame_index)
        current = int(self.frame_index / self.fps)
        total = int(self.frame_count / self.fps)
        self.position.setText(f"{current // 60:02}:{current % 60:02} / {total // 60:02}:{total % 60:02}")

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._render()

    def _render(self):
        if self.current_image is not None:
            pixmap = QPixmap.fromImage(self.current_image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def toggle_playback(self):
        if self.capture is None:
            return
        if self.timer.isActive():
            self.timer.stop()
        else:
            if self.frame_index >= self.frame_count - 1:
                self.seek(0)
            self.timer.start()
        icon = QStyle.StandardPixmap.SP_MediaPause if self.timer.isActive() else QStyle.StandardPixmap.SP_MediaPlay
        self.play.setIcon(self.style().standardIcon(icon))

    def release(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def make_table(headers: list[str], rows: list[list[object]]) -> QTableWidget:
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().hide()
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            table.setItem(row_index, column, QTableWidgetItem(str(value)))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return table


class TrafficResearchWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.snapshot = build_project_snapshot(root)
        self.input_path = None
        self.report = None
        self.report_path = None
        self.job = None
        self.cancelled = False
        self.capture_report_path = None
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_camera_preview)
        self.setWindowTitle("Traffic Incident Detection and Risk Assessment | Research Workbench")
        self.resize(1400, 900)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(STYLE)
        self._build_toolbar()
        self.navigation = QListWidget()
        self.navigation.addItems(["Incident workspace", "Model evaluation", "Visual evidence", "Government risk", "Edge deployment"])
        self.navigation.setFixedWidth(185)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._incident_page())
        self.pages.addWidget(self._model_page())
        self.pages.addWidget(self._evidence_page())
        self.pages.addWidget(self._government_page())
        self.pages.addWidget(self._deployment_page())
        self.event_profile.currentIndexChanged.connect(self._update_reference_metrics)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.addWidget(self.navigation)
        content.addWidget(self.pages, 1)
        container = QWidget()
        container.setLayout(content)
        self.setCentralWidget(container)
        self.statusBar().showMessage("Research artifacts loaded | CPU thread budget: 2")
        self.load_demo()

    def _build_toolbar(self):
        toolbar = QToolBar("Project actions")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        actions = [
            ("Open video", QStyle.StandardPixmap.SP_DialogOpenButton, self.open_video),
            ("Camera observation", QStyle.StandardPixmap.SP_DriveNetIcon, self.open_camera_dialog),
            ("Open results", QStyle.StandardPixmap.SP_FileDialogDetailedView, self.open_results),
            ("Reference clip", QStyle.StandardPixmap.SP_FileDialogContentsView, self.load_demo),
            ("Analyze", QStyle.StandardPixmap.SP_MediaPlay, self.analyze_video),
            ("Review selected event", QStyle.StandardPixmap.SP_FileDialogContentsView, self.review_selected_event),
            ("Cancel", QStyle.StandardPixmap.SP_MediaStop, self.cancel_analysis),
            ("Export results", QStyle.StandardPixmap.SP_DialogSaveButton, self.export_results),
        ]
        self.actions_by_name = {}
        for text, icon, callback in actions:
            action = QAction(self.style().standardIcon(icon), text, self)
            action.setToolTip(text)
            action.triggered.connect(callback)
            toolbar.addAction(action)
            self.actions_by_name[text] = action
        self.actions_by_name["Cancel"].setEnabled(False)
        self.actions_by_name["Export results"].setEnabled(False)
        self.actions_by_name["Review selected event"].setEnabled(False)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Temporal runtime "))
        self.provider = QComboBox()
        capabilities = runtime_capabilities()
        self.provider.addItem("ONNX CPU", "CPUExecutionProvider")
        if capabilities["qnn_available"]:
            self.provider.addItem("Qualcomm QNN HTP", "QNNExecutionProvider")
        toolbar.addWidget(self.provider)
        toolbar.addWidget(QLabel(" Event profile "))
        self.event_profile = QComboBox()
        self.event_profile.addItem("Responsive", 1)
        self.event_profile.addItem("Conservative", 2)
        self.event_profile.setCurrentIndex(1)
        toolbar.addWidget(self.event_profile)
        self.vlm_enabled = QCheckBox("Qwen evidence review")
        self.vlm_enabled.setToolTip("Review up to three candidate events with the local Qwen2.5-VL model")
        toolbar.addWidget(self.vlm_enabled)

    def _page(self, title: str, subtitle: str):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        text = QLabel(subtitle)
        text.setWordWrap(True)
        text.setObjectName("subtitle")
        layout.addWidget(text)
        return page, layout

    def _incident_page(self):
        page, layout = self._page("Traffic incident workspace", "Vehicle proposals · causal temporal inference · persistence-gated review")
        metrics = QHBoxLayout()
        self.metric_labels = []
        for title, value in self._reference_metric_values():
            label = QLabel(self._metric_html(title, value))
            label.setObjectName("metric")
            metrics.addWidget(label)
            self.metric_labels.append(label)
        layout.addLayout(metrics)
        self.video = VideoPanel()
        self.events = make_table(["Event", "Trigger time", "Probability"], [])
        self.events.itemSelectionChanged.connect(self.seek_selected_event)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(self.events)
        splitter.setSizes([780, 260])
        layout.addWidget(splitter, 1)
        self.clip_label = QLabel()
        self.clip_label.setObjectName("subtitle")
        layout.addWidget(self.clip_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMaximumHeight(135)
        layout.addWidget(self.log)
        return page

    @staticmethod
    def _metric_html(title, value):
        return f"<span style='color:#60746c;font-size:11px'>{title}</span><br><span style='font-size:23px;font-weight:600'>{value}</span>"

    def _reference_metric_values(self):
        name = self.event_profile.currentText().lower()
        profile = next(row for row in self.snapshot["event_profiles"] if row["profile"] == name)
        geographic = next(row for row in self.snapshot["model_comparison"] if row["model"] == "Candidate ROI TCN" and row["split"] == "geographic")
        return [(f"IID benchmark recall / {name} / hop 5", f'{profile["event_recall"]:.1%}'), (f"IID benchmark delay / {name} / hop 5", f'{profile["delay_median_sec"]:.3f} s'), ("Reference window F1 / geographic", f'{geographic["f1"]:.3f}')]

    def _update_reference_metrics(self):
        for label, (title, value) in zip(self.metric_labels, self._reference_metric_values()):
            label.setText(self._metric_html(title, value))

    def _model_page(self):
        page, layout = self._page("Temporal model evaluation", "500-clip reference and 2,021-clip extended cohorts; grouped calibration and IID / geographic evaluation")
        profiles = [("IID", row) for row in self.snapshot["event_profiles"]] + [("Geographic", row) for row in self.snapshot["geographic_event_profiles"]]
        layout.addWidget(make_table(["Split", "Profile", "Hop", "Clips", "Event recall", "Median delay", "p95 delay", "Early / late episodes"], [[split, p["profile"], 5 if split == "IID" else 15, p["clips"], f'{p["event_recall"]:.3f}', f'{p["delay_median_sec"]:.3f} s', f'{p["delay_p95_sec"]:.3f} s', f'{p["early_alert_episodes"]} / {p["late_alert_episodes"]}'] for split, p in profiles]))
        tabs = QTabWidget()
        tabs.addTab(make_table(["Model", "Split", "Precision", "Recall", "F1", "FPR"], [[r["model"], r["split"], *(f'{r[key]:.3f}' for key in ("precision", "recall", "f1", "false_positive_rate"))] for r in self.snapshot["model_comparison"]]), "500-clip reference")
        tabs.addTab(make_table(["Representation", "Split", "Fit", "Calibration", "Test clips", "F1", "FPR", "AUROC", "AP"], [[r["display_name"], r["split"], r["fit_clips"], r["calibration_clips"], r["test_clips"], f'{r["test"]["f1"]:.3f}', f'{r["test"]["false_positive_rate"]:.3f}', f'{r["ranking"]["auroc"]:.3f}', f'{r["ranking"]["average_precision"]:.3f}'] for r in self.snapshot["full_cohort_models"]]), "2,021-clip extension")
        layout.addWidget(tabs, 1)
        return page

    def _evidence_page(self):
        page, layout = self._page("Multimodal visual evidence", "Temporal context, image restoration, and model-specific evidence products")
        self.evidence_choice = QComboBox()
        self.evidence_choice.addItems(["Before / trigger / after", "NAFNet restoration", "SM3Det architecture"])
        self.frame_choice = QComboBox()
        self.frame_choice.addItems(["Temporal context", "Before", "Trigger", "After"])
        self.frame_choice.setCurrentIndex(2)
        controls = QHBoxLayout()
        controls.addWidget(self.evidence_choice, 1)
        controls.addWidget(self.frame_choice)
        layout.addLayout(controls)
        self.evidence_image = QLabel()
        self.evidence_image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.evidence_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evidence_image.setMinimumHeight(280)
        layout.addWidget(self.evidence_image, 1)
        self.evidence_text = QLabel()
        self.evidence_text.setTextFormat(Qt.TextFormat.PlainText)
        self.evidence_text.setWordWrap(True)
        layout.addWidget(self.evidence_text)
        self.evidence_choice.currentIndexChanged.connect(self.show_evidence)
        self.frame_choice.currentIndexChanged.connect(lambda: self.show_evidence(self.evidence_choice.currentIndex()))
        self.show_evidence(0)
        return page

    def show_evidence(self, index: int):
        assets = [
            ("incident_vlm_clean_evidence.jpg", "Qwen2.5-VL-3B: structured evidence review. Clean-blind evaluation identifies the importance of temporal localization and independent event labels."),
            ("nafnet_cctv_demo/comparison.jpg", "NAFNet restoration: PSNR 21.91 to 23.41 dB; SSIM 0.517 to 0.636. Official REDS checkpoint, privacy-treated reference, deterministic motion blur."),
            ("sm3det_architecture.png", "SM3Det: RGB / SAR / infrared multi-expert architecture. Official architecture reproduced for multi-sensor transfer analysis; 178M parameters and 487G FLOPs."),
        ]
        path, text = assets[index]
        image_path = self.root / "docs/assets" / path
        self.frame_choice.setVisible(index == 0)
        no_event = index == 0 and self.report is not None and not self.report["events"]
        self.frame_choice.setEnabled(not no_event)
        if no_event:
            self.evidence_image.clear()
            self.evidence_text.setText("No review event in this analysis")
            return
        frame_name = (None, "before", "trigger", "after")[self.frame_choice.currentIndex()]
        if index == 0 and self.report is not None and self.report["events"]:
            event = self.report["events"][max(0, self.events.currentRow())]
            image_path = Path(event["evidence"]["contact_sheet"])
            if frame_name is not None:
                image_path = image_path.with_name(f"{frame_name}.jpg")
            text = f'Current event at {event["trigger_time_sec"]:.3f} s. Ordered clean before / trigger / after evidence.'
            if "vlm_review" in event:
                review = event["vlm_review"]
                disposition = review_disposition(True, review).replace("_", " ")
                text += f' Disposition (policy {REVIEW_POLICY_VERSION}): {disposition}. Qwen2.5-VL confidence={review["confidence"]:.3f}. ' + " ".join(review["visual_reasons"])
                if review.get("missing_evidence"):
                    text += " Evidence gaps: " + "; ".join(review["missing_evidence"])
                if review.get("human_review_needed"):
                    text += " Operator review requested."
        pixmap = QPixmap(str(image_path))
        if index == 0 and self.report is None and frame_name is not None:
            offset = self.frame_choice.currentIndex() - 1
            panel_width = pixmap.width() // 3
            pixmap = pixmap.copy(offset * panel_width, 0, panel_width, pixmap.height())
        self.evidence_image.setPixmap(pixmap.scaled(950, 460, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.evidence_text.setText(text)

    def _government_page(self):
        page, layout = self._page("Government-informed camera allocation", "Taiwan A1/A2 crash demand and fixed public-camera inventory; temporal holdout evaluation")
        tabs = QTabWidget()
        comparison_path = self.root / "docs/results/government_camera_budget/comparison.csv"
        if comparison_path.exists():
            with comparison_path.open(encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if float(row["a1_weight"]) == 5 and int(row["radius_m"]) == 1000]
            tabs.addTab(make_table(["Camera budget", "Selection method", "Held-out risk coverage", "Randomization SD"], [[row["budget"], row["method"], f'{float(row["holdout_risk_coverage_mean"]):.2%}', f'{float(row["holdout_risk_coverage_sd"]):.2%}'] for row in rows]), "Camera budgets")
        matches = self.snapshot["government"]["nearest_camera_matches"]
        tabs.addTab(make_table(["Rank", "Risk location", "Risk score", "Nearest camera distance"], [[r["risk_rank"], r["representative_location"], r["risk_score"], f'{r["distance_m"]:.0f} m'] for r in matches]), "Site inventory")
        curve = QLabel()
        curve.setAlignment(Qt.AlignmentFlag.AlignCenter)
        curve.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        curve.setPixmap(QPixmap(str(self.root / "docs/assets/government_camera_budget.png")).scaled(1050, 620, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        tabs.addTab(curve, "Coverage curve")
        layout.addWidget(tabs, 1)
        return page

    def _deployment_page(self):
        page, layout = self._page("Edge runtime and hardware validation", "Explicit execution providers and trace-driven capacity measurements")
        caps = runtime_capabilities()
        quantization = self.snapshot["deployment"]["qnn_quantization"]
        measured_queue = self.snapshot["deployment"]["measured_review_queue"]
        layout.addWidget(make_table(["Component", "Measurement", "Evidence"], [
            ["ONNX Runtime", caps["onnxruntime_version"], ", ".join(caps["available_providers"])],
            ["TCN ONNX parity", "128 samples", "Maximum absolute error 1.79e-7"],
            ["QNN A16W8 preparation", f'{quantization["test_samples"]} test windows', f'{quantization["decision_agreement"]:.1%} float32 decision agreement'],
            ["Verilog", "3 / 3 testbenches", "Dwell / overlap / motion trigger"],
            ["SystemC 3.0.2", "49,320 processed frames", "Python trace parity"],
            ["4-camera frame queue", "0 dropped frames", "7.5 FPS per camera"],
            ["Review queue", "68.33% candidate drops", "300 ms review-stage assumption"],
            ["Measured Qwen review", f'{measured_queue["profile"]["review_latency_ms"] / 1000:.2f} s / review', f'{measured_queue["result"]["review_drop_rate"]:.2%} candidate drops; four replicated streams'],
            ["Qualcomm QNN HTP", "Available" if caps["qnn_available"] else "Install target runtime", "Explicit provider selection; CPU fallback disabled"],
        ]), 1)
        routing = self.snapshot["deployment"]["review_routing"]
        tabs = QTabWidget()
        tabs.addTab(make_table(["Review policy", "Merged", "Budget denied", "Completed", "Pending", "Queue drops"], [[r["label"], r["coalesced_proposals"], r["budget_denied"], r["queue"]["completed_reviews"], r["queue"]["pending_reviews"], r["queue"]["dropped_candidates"]] for r in routing["policies"]]), "Proposal workload")
        event_rows = []
        for split, study in self.snapshot["deployment"]["incident_review_routing"].items():
            for row in study["policies"]:
                retention = row["event_retention"]
                event_rows.append([split, row["min_positive_windows"], row["camera_count"], row["policy"], retention["detected_before_routing"], retention["retained_after_completed_service"], f'{retention["completed_review_latency_sec"]["p95"]:.2f} s'])
        tabs.addTab(make_table(["Split", "Gate", "Cameras", "Policy", "Detected", "Service retained", "Review p95"], event_rows), "TCN event retention")
        layout.addWidget(tabs, 1)
        return page

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select traffic video", str(self.root), "Videos (*.mp4 *.avi *.mov *.mkv)")
        if filename:
            self.load_input(Path(filename))

    def open_camera_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Camera observation")
        dialog.resize(600, 240)
        layout = QFormLayout(dialog)
        mode = QComboBox()
        for label, value in (("Public page / MJPEG", "page"), ("MJPEG URL", "mjpeg"), ("Snapshot URL", "snapshot")):
            mode.addItem(label, value)
        url = QLineEdit("https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129")
        url.setMaxLength(2048)
        duration = QSpinBox()
        duration.setRange(30, 120)
        duration.setSingleStep(30)
        duration.setValue(60)
        analyze = QCheckBox()
        analyze.setChecked(True)
        continuous = QCheckBox()
        continuous.setChecked(True)
        analyze.setEnabled(False)
        continuous.toggled.connect(lambda active: analyze.setEnabled(not active))
        layout.addRow("Source type", mode)
        layout.addRow("Source URL", url)
        layout.addRow("Duration (s)", duration)
        layout.addRow("Continuous inference", continuous)
        layout.addRow("Analyze when ready", analyze)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.capture_camera(url.text().strip(), mode.currentData(), duration.value(), analyze.isChecked(), continuous=continuous.isChecked())
            except ValueError as exc:
                QMessageBox.warning(self, "Camera source", str(exc))

    def capture_camera(self, url: str, mode: str, duration_sec: float, analyze_after: bool = True, *, continuous: bool = False):
        if self.job and self.job.state() != QProcess.ProcessState.NotRunning:
            return
        public_source = redacted_camera_url(url)
        if mode not in {"page", "mjpeg", "snapshot"} or not 0 < duration_sec <= 120:
            raise ValueError("Select a supported source and an observation of at most 120 seconds")
        self.input_path = self.report = self.capture_report_path = None
        self.report_path = None
        self.actions_by_name["Review selected event"].setEnabled(False)
        self.events.setRowCount(0)
        self.show_evidence(self.evidence_choice.currentIndex())
        self.actions_by_name["Export results"].setEnabled(False)
        self.output_dir = self.root / "outputs/desktop" / uuid.uuid4().hex[:12]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video.release()
        self.video.current_image = None
        self.video.preview.clear()
        self.video.preview.setText("Connecting camera")
        self.video.play.setEnabled(False)
        self.video.slider.setEnabled(False)
        self.clip_label.setText(public_source.split("/", 3)[2])
        self.navigation.setCurrentRow(0)
        self._preview_modified = None
        self._capture_started = time.perf_counter()
        if continuous:
            arguments = ["--input", url, "--source-mode", mode, "--duration-sec", str(duration_sec), "--output-dir", str(self.output_dir), "--stop-file", str(self.output_dir / "cancel_capture"), "--min-positive-windows", str(self.event_profile.currentData()), "--provider", self.provider.currentData()]
            self._start_job("scripts/run_streaming_incident.py", arguments, "stream")
            self.log.appendPlainText("Streaming review: clean evidence export; VLM service is deferred")
        else:
            arguments = ["--page-url" if mode == "page" else "--source-url", url, "--transport", "snapshot" if mode == "snapshot" else "mjpeg", "--duration-sec", str(duration_sec), "--output-video", str(self.output_dir / "capture.mp4"), "--output-report", str(self.output_dir / "capture.json"), "--preview", str(self.output_dir / "preview.jpg"), "--stop-file", str(self.output_dir / "cancel_capture")]
            self._start_job("scripts/capture_public_cctv.py", arguments, "capture")
        self.job.setProperty("analyze_after_capture", analyze_after)
        self.preview_timer.start(300)

    def _refresh_camera_preview(self):
        path = self.output_dir / "preview.jpg"
        try:
            modified = path.stat().st_mtime_ns
            if modified == self._preview_modified:
                return
            import numpy as np
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                self.video.display_frame(frame)
                self.video.position.setText(f"{time.perf_counter() - self._capture_started:.0f} s | CCTV")
                self._preview_modified = modified
                if self.job and self.job.property("kind") == "stream":
                    progress = json.loads((self.output_dir / "progress.json").read_text(encoding="utf-8"))
                    self.events.setRowCount(len(progress["events"]))
                    for index, event in enumerate(progress["events"]):
                        for column, value in enumerate((index + 1, f'{event["trigger_time_sec"]:.3f} s', f'{event["probability"]:.3f}')):
                            self.events.setItem(index, column, QTableWidgetItem(str(value)))
                    self.statusBar().showMessage(f'Streaming | {progress["sampled_frames"]} frames | {progress["window_count"]} causal decisions | {len(progress["events"])} events')
        except (OSError, ValueError):
            return

    def open_results(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open completed analysis", str(self.root / "outputs/desktop"), "JSON (*.json)")
        if filename:
            try:
                self.load_report(Path(filename))
            except (OSError, ValueError, KeyError) as exc:
                QMessageBox.warning(self, "Analysis results", str(exc))

    def load_input(self, path: Path):
        if self.job and self.job.state() != QProcess.ProcessState.NotRunning:
            return
        try:
            self.video.load(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Video input", str(exc))
            return
        self.input_path = path
        self.capture_report_path = None
        capture_path = path.with_name("capture.json")
        if capture_path.is_file():
            try:
                capture = load_camera_observation(capture_path, path, self.video.frame_count)
                self.capture_report_path = capture_path
            except (OSError, ValueError):
                self.log.appendPlainText("Capture metadata could not be associated with this video")
        self.report = None
        self.report_path = None
        self.actions_by_name["Review selected event"].setEnabled(False)
        self.events.setRowCount(0)
        self.show_evidence(self.evidence_choice.currentIndex())
        self.actions_by_name["Export results"].setEnabled(False)
        self.actions_by_name["Analyze"].setEnabled(True)
        self.clip_label.setText(path.name)
        if self.capture_report_path is not None:
            self.clip_label.setText(f'{path.name} | {capture["effective_fps"]:.2f} FPS | {capture["exact_duplicate_rate"]:.1%} repeated frames')
        self.navigation.setCurrentRow(0)

    def load_demo(self):
        self.load_input(self.root / "docs/assets/incident_input.mp4")

    def seek_selected_event(self):
        row = self.events.currentRow()
        if self.report is not None and 0 <= row < len(self.report["events"]):
            event = self.report["events"][row]
            frame_index = event.get("output_frame_index", round(event["trigger_time_sec"] * self.video.fps))
            self.video.seek(frame_index)
            self.show_evidence(0)

    def analyze_video(self):
        if self.input_path is None or (self.job and self.job.state() != QProcess.ProcessState.NotRunning):
            return
        self.cancelled = False
        self.report = None
        self.events.setRowCount(0)
        self.show_evidence(self.evidence_choice.currentIndex())
        self.actions_by_name["Export results"].setEnabled(False)
        self.output_dir = self.root / "outputs/desktop" / uuid.uuid4().hex[:12]
        arguments = ["--input", str(self.input_path), "--output-dir", str(self.output_dir), "--provider", self.provider.currentData(), "--min-positive-windows", str(self.event_profile.currentData())]
        if self.vlm_enabled.isChecked():
            arguments.append("--vlm-review")
        if self.capture_report_path is not None:
            arguments.extend(["--capture-report", str(self.capture_report_path)])
        self._start_job("scripts/analyze_desktop_video.py", arguments, "analysis")

    def review_selected_event(self):
        if self.report_path is None or not self.report or not self.report["events"] or (self.job and self.job.state() != QProcess.ProcessState.NotRunning):
            return
        row = max(0, self.events.currentRow())
        self.output_dir = self.root / "outputs/desktop" / uuid.uuid4().hex[:12]
        arguments = ["--report", str(self.report_path), "--event-index", str(row), "--output-dir", str(self.output_dir)]
        self._start_job("scripts/review_desktop_events.py", arguments, "review")

    def _start_job(self, script: str, arguments: list[str], kind: str):
        self.cancelled = False
        self.log.clear()
        job = QProcess(self)
        job.setProperty("kind", kind)
        self.job = job
        job.setWorkingDirectory(str(self.root))
        environment = QProcessEnvironment.systemEnvironment()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment.insert(name, "2")
        environment.insert("PYTHONIOENCODING", "utf-8")
        job.setProcessEnvironment(environment)
        job.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        job.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(bytes(job.readAllStandardOutput()).decode("utf-8", errors="replace").rstrip()))
        job.finished.connect(lambda code, status: self._job_finished(job, code))
        job.errorOccurred.connect(lambda error: self._job_error(job, error))
        self.actions_by_name["Analyze"].setEnabled(False)
        self.actions_by_name["Review selected event"].setEnabled(False)
        self.actions_by_name["Cancel"].setEnabled(True)
        self.provider.setEnabled(False)
        self.event_profile.setEnabled(False)
        self.vlm_enabled.setEnabled(False)
        self.actions_by_name["Open video"].setEnabled(False)
        self.actions_by_name["Reference clip"].setEnabled(False)
        self.actions_by_name["Open results"].setEnabled(False)
        self.actions_by_name["Camera observation"].setEnabled(False)
        self.statusBar().showMessage({"capture": "Observing camera", "stream": "Streaming inference | connecting camera", "review": "Reviewing saved event evidence | original detector result preserved"}.get(kind, "Analyzing video | CPU proposals + selected temporal runtime"))
        job.start(worker_python(), [str(self.root / script), *arguments])

    def _reset_job_controls(self):
        self.actions_by_name["Analyze"].setEnabled(self.input_path is not None)
        self.actions_by_name["Cancel"].setEnabled(False)
        self.provider.setEnabled(True)
        self.event_profile.setEnabled(True)
        self.vlm_enabled.setEnabled(True)
        self.actions_by_name["Open video"].setEnabled(True)
        self.actions_by_name["Reference clip"].setEnabled(True)
        self.actions_by_name["Open results"].setEnabled(True)
        self.actions_by_name["Camera observation"].setEnabled(True)
        self.actions_by_name["Review selected event"].setEnabled(bool(self.report_path and self.report and self.report["events"]))

    def _job_error(self, job: QProcess, error):
        if job is not self.job:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self.preview_timer.stop()
            self._reset_job_controls()
            self.log.appendPlainText(job.errorString())
            self.statusBar().showMessage("Analysis process could not start")

    def _job_finished(self, job: QProcess, code: int):
        if job is not self.job:
            return
        self.preview_timer.stop()
        self._reset_job_controls()
        if self.cancelled:
            self.statusBar().showMessage("Job cancelled")
            return
        if job.property("kind") == "capture":
            capture_path = self.output_dir / "capture.json"
            if code != 0 or not capture_path.is_file():
                self.statusBar().showMessage("Camera observation ended with an error; see process output")
                return
            self.load_input(self.output_dir / "capture.mp4")
            if job.property("analyze_after_capture"):
                self.analyze_video()
            else:
                self.statusBar().showMessage("Camera observation saved")
            return
        report_path = self.output_dir / "report.json"
        if code != 0 or not report_path.is_file():
            self.statusBar().showMessage("Analysis ended with an error; process output is available below")
            return
        self.load_report(report_path)

    def load_report(self, report_path: Path):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("events"), list) or not isinstance(payload.get("runtime"), dict):
            raise ValueError("Select a completed desktop analysis report")
        profile_index = self.event_profile.findData(payload.get("min_positive_windows", 2))
        if profile_index < 0:
            raise ValueError("Report event profile must require one or two positive windows")
        video_path = Path(payload["output_video"])
        if not video_path.is_absolute():
            video_path = report_path.parent / video_path
        self.video.load(video_path)
        self.report = payload
        self.report_path = report_path.resolve()
        self.event_profile.setCurrentIndex(profile_index)
        source = Path(payload.get("input", ""))
        if not source.is_absolute():
            source = self.root / source
        self.input_path = source if source.is_file() else None
        self.capture_report_path = source.with_name("capture.json") if self.input_path is not None and source.with_name("capture.json").is_file() else None
        self.actions_by_name["Analyze"].setEnabled(self.input_path is not None)
        self.clip_label.setText(f"{video_path.name} | {report_path.name}")
        if "capture" in self.report:
            self.clip_label.setText(f'{video_path.name} | source {self.report["capture"]["effective_fps"]:.2f} FPS | {report_path.name}')
        self.events.setRowCount(len(self.report["events"]))
        for index, event in enumerate(self.report["events"]):
            for column, value in enumerate((index + 1, f'{event["trigger_time_sec"]:.3f} s', f'{event["probability"]:.3f}')):
                self.events.setItem(index, column, QTableWidgetItem(str(value)))
        self.show_evidence(0)
        self.actions_by_name["Export results"].setEnabled(True)
        self.actions_by_name["Review selected event"].setEnabled(bool(payload["events"]))
        self.statusBar().showMessage(f'{len(self.report["events"])} review events | {self.report["runtime"]["requested_provider"]} | {self.report["elapsed_sec"]:.1f} s processing')

    def cancel_analysis(self):
        if self.job and self.job.state() != QProcess.ProcessState.NotRunning:
            self.cancelled = True
            job = self.job
            if job.property("kind") in {"capture", "stream"}:
                (self.output_dir / "cancel_capture").touch()
                timeout = 6500
            else:
                job.terminate()
                timeout = 1500
            QTimer.singleShot(timeout, job, lambda: job.kill() if job.state() != QProcess.ProcessState.NotRunning else None)

    def export_results(self):
        if self.report is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export analysis results", "incident_report.json", "JSON (*.json)")
        if filename:
            Path(filename).write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")

    def closeEvent(self, event: QCloseEvent):
        self.preview_timer.stop()
        self.video.release()
        if self.job and self.job.state() != QProcess.ProcessState.NotRunning:
            self.job.kill()
            self.job.waitForFinished(3000)
        event.accept()


STYLE = """
QMainWindow, QWidget { background: #f3f6f4; color: #182920; font-family: 'Segoe UI'; font-size: 12px; }
QToolBar { background: #ffffff; border-bottom: 1px solid #d4ded8; padding: 8px; spacing: 8px; }
QToolButton { padding: 7px; border-radius: 4px; }
QToolButton:hover { background: #e0efea; }
QListWidget { background: #eaf0ec; border: none; border-right: 1px solid #d4ded8; padding: 12px 6px; }
QListWidget::item { padding: 14px 8px; margin: 2px 0; }
QListWidget::item:selected { background: #d6ebe3; color: #075d4c; border-left: 3px solid #08765c; }
QLabel#pageTitle { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
QLabel#subtitle { color: #62756a; padding-bottom: 8px; }
QLabel#metric { background: #ffffff; border: 1px solid #d4ded8; border-radius: 5px; padding: 15px; margin: 8px 0 16px; }
QLabel#videoPreview { background: #111b15; color: #d5ded8; }
QTableWidget { background: #ffffff; alternate-background-color: #f5f8f6; border: 1px solid #d4ded8; gridline-color: #e4ebe6; selection-background-color: #dfefe8; selection-color: #182920; }
QHeaderView::section { background: #e7eeea; padding: 10px; border: none; border-bottom: 1px solid #d4ded8; font-weight: 600; }
QPlainTextEdit { background: #ffffff; border: 1px solid #d4ded8; font-family: Consolas; font-size: 11px; }
QComboBox, QPushButton { background: #ffffff; border: 1px solid #c7d5cc; border-radius: 4px; padding: 6px; }
QStatusBar { background: #e6eeE9; color: #536b5e; }
"""
