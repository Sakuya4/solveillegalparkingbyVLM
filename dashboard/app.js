const stateLabels = {
  verified: "已驗證",
  pilot: "試驗完成",
  external: "需外部條件",
};

const formatPercent = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;
const formatNumber = (value, digits = 0) => Number(value).toLocaleString("zh-TW", {
  minimumFractionDigits: digits,
  maximumFractionDigits: digits,
});
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[character]);

function metric(label, value, note) {
  return `<div class="metric"><span class="metric-label">${label}</span><strong>${value}</strong><small>${note}</small></div>`;
}

function statusLabel(state) {
  return `<span class="status-label ${state}">${stateLabels[state]}</span>`;
}

function renderOverview(data) {
  const responsive = data.event_profiles.find((row) => row.profile === "responsive");
  const cctv = data.deployment.normal_cctv;
  const coverage = data.government.coverage["1000"];
  document.querySelector("#overview-metrics").innerHTML = [
    metric("事故事件召回", formatPercent(responsive.event_recall), "100 clips responsive profile"),
    metric("觸發延遲中位數", `${responsive.delay_median_sec.toFixed(3)}s`, `p95 ${responsive.delay_p95_sec.toFixed(3)}s`),
    metric("正常 CCTV alerts", cctv.false_alert_episodes, `${cctv.observed_camera_hours.toFixed(4)} camera-hours pilot`),
    metric("1 km 風險覆蓋", formatPercent(coverage.risk_weighted_coverage_rate, 2), "17 個公開攝影機點"),
  ].join("");

  document.querySelector("#pipeline").innerHTML = data.pipeline.map((step) => `
    <div class="pipeline-step ${step.state}">
      <strong>${escapeHtml(step.name)}</strong><span>${escapeHtml(step.detail)}</span>
    </div>`).join("");

  document.querySelector("#overview-video").src = data.media.model_output_video;
  document.querySelector("#event-profiles").innerHTML = data.event_profiles.map((profile) => `
    <div class="profile-row">
      <div class="profile-title"><strong>${profile.profile}</strong><span>${profile.min_positive_windows} positive window${profile.min_positive_windows > 1 ? "s" : ""}</span></div>
      <div class="profile-metrics">
        <div><span>Event recall</span><strong>${formatPercent(profile.event_recall)}</strong></div>
        <div><span>Median delay</span><strong>${profile.delay_median_sec.toFixed(3)}s</strong></div>
        <div><span>Early / late</span><strong>${profile.early_alert_episodes} / ${profile.late_alert_episodes}</strong></div>
      </div>
    </div>`).join("");
}

function renderModels(data, split = "iid") {
  const rows = data.model_comparison.filter((row) => row.split === split);
  const bestF1 = Math.max(...rows.map((row) => row.f1));
  document.querySelector("#model-comparison").innerHTML = rows.map((row) => `
    <div class="model-row ${row.f1 === bestF1 ? "best" : ""}">
      <div class="model-name"><strong>${escapeHtml(row.model)}</strong><span>${escapeHtml(row.training.replaceAll("_", " "))}</span></div>
      ${barCell(row.f1)}
      ${barCell(row.recall)}
      ${barCell(row.false_positive_rate, true)}
    </div>`).join("");
}

function barCell(value, isFpr = false) {
  return `<div class="bar-cell"><span>${Number(value).toFixed(3)}</span><div class="bar-track"><div class="bar-fill ${isFpr ? "fpr" : ""}" style="width:${Math.min(Number(value) * 100, 100)}%"></div></div></div>`;
}

function evidenceConfig(data) {
  const restoration = data.restoration;
  return {
    output: {
      kind: "video", src: data.media.model_output_video, label: "MODEL EVENT OUTPUT",
      title: "時序事故觸發", body: "候選框來自 YOLO、ByteTrack 與 motion ROI，causal TCN 依連續視窗決定是否送審。",
      metrics: [["資料洩漏", "Inference 不讀取事故 bbox"], ["Demo trigger", "事故後 0.792s"], ["隱私", "車輛下半部模糊"]],
    },
    reference: {
      kind: "video", src: data.media.reference_video, label: "ACCIDENT REFERENCE",
      title: "公開事故資料", body: "ACCIDENT 固定式 CCTV 片段提供事故時間、類型與位置標註，用於推論後事件評估。",
      metrics: [["正式批次", "100 IID test clips"], ["訓練比較", "500 clips"], ["授權", "CC BY-NC-SA 4.0"]],
    },
    vlm: {
      kind: "image", src: data.media.vlm_evidence, label: "CLEAN BLIND REVIEW",
      title: "VLM 證據覆核", body: "移除模型 overlay 與分數後，Qwen2.5-VL-3B 漏掉遠距碰撞，證明 VLM 應負責摘要與佇列排序。",
      metrics: [["模型", "Qwen2.5-VL-3B"], ["Schema", "有效 JSON"], ["Clean-blind", "漏判可見碰撞"]],
    },
    nafnet: {
      kind: "image", src: data.media.nafnet_comparison, label: "DIFFICULT-FRAME RESTORATION",
      title: "NAFNet 困難畫面復原", body: "先完成車牌與文字隱私處理，再加入 deterministic motion blur 並以官方 REDS 權重復原。",
      metrics: [["PSNR", `${restoration.degraded_psnr_db.toFixed(2)} → ${restoration.restored_psnr_db.toFixed(2)} dB`], ["SSIM", `${restoration.degraded_ssim.toFixed(3)} → ${restoration.restored_ssim.toFixed(3)}`], ["RTX 3060", `${(restoration.latency_ms / 1000).toFixed(2)}s / image`]],
    },
    sm3det: {
      kind: "image", src: data.media.sm3det_architecture, label: "MULTI-SENSOR RESEARCH BRANCH",
      title: "SM3Det 適用性稽核", body: "官方模型針對 RGB、SAR、IR 遙測偵測。已驗證 release，但不把遙測 mAP 當作道路 CCTV 成績。",
      metrics: [["Modalities", data.sm3det.modalities.join(" / ")], ["Parameters", `${data.sm3det.parameters_m}M`], ["FLOPs", `${data.sm3det.flops_g}G`]],
    },
  };
}

function renderEvidence(data, key = "output") {
  const item = evidenceConfig(data)[key];
  const media = item.kind === "video"
    ? `<video controls muted playsinline preload="metadata" src="${item.src}"></video>`
    : `<img src="${item.src}" alt="${item.title}">`;
  document.querySelector("#evidence-stage").innerHTML = media;
  document.querySelector("#evidence-inspector").innerHTML = `
    <p class="eyebrow">${item.label}</p><h2>${item.title}</h2><p>${item.body}</p>
    ${item.metrics.map(([label, value]) => `<div class="inspector-metric"><span>${label}</span><strong>${value}</strong></div>`).join("")}`;
}

function renderAblations(data) {
  document.querySelector("#ablation-rows").innerHTML = data.ablations.map((row) => `
    <tr><td>${escapeHtml(row.stage)}</td><td>${escapeHtml(row.method)}</td><td>${escapeHtml(row.samples)}</td><td>${escapeHtml(String(row.decision_or_recall).replaceAll("_", " "))}</td><td>${Number(row.elapsed_sec).toFixed(3)}s</td><td>${escapeHtml(String(row.result).replaceAll("_", " "))}</td></tr>`).join("");
}

function renderDeployment(data) {
  const edge = data.deployment.edge_queue;
  const onnx = data.deployment.onnx_qnn;
  const verilog = data.deployment.verilog;
  const systemc = data.deployment.systemc;
  document.querySelector("#edge-metrics").innerHTML = [
    metric("多攝影機輸入", `${edge.camera_count} × ${edge.camera_fps} FPS`, `${formatNumber(edge.processed_frames)} frames`),
    metric("NPU frame drop", formatPercent(edge.frame_drop_rate), `p95 ${edge.p95_frame_latency_ms.toFixed(1)}ms`),
    metric("VLM review drop", formatPercent(edge.review_drop_rate), `${formatNumber(edge.completed_reviews)} completed reviews`),
    metric("ONNX parity error", onnx.max_absolute_error.toExponential(2), `${onnx.parity_samples} samples`),
  ].join("");

  const statuses = [
    ["Verilog trigger logic", `${verilog.passed_testbenches} / ${verilog.total_testbenches} testbenches`, "verified"],
    [`SystemC ${systemc.version}`, `${formatNumber(systemc.processed_frames)} frames, Python parity`, "verified"],
    ["TCN ONNX", `${onnx.parity_samples} samples checker + parity`, "verified"],
    ["Qualcomm QNN", "compile workflow ready, device latency pending", "external"],
  ];
  document.querySelector("#hardware-status").innerHTML = statuses.map(([name, detail, status]) => `
    <div class="status-row"><div><strong>${name}</strong><span>${detail}</span></div>${statusLabel(status)}</div>`).join("");

  document.querySelector("#coverage-bars").innerHTML = Object.entries(data.government.coverage).map(([radius, coverage]) => `
    <div class="coverage-row">
      <div class="coverage-label"><strong>${Number(radius) / 1000} km</strong><span>${coverage.covered_hotspots} / ${data.government.records.new_taipei_hotspots_joined} hotspots · ${formatPercent(coverage.risk_weighted_coverage_rate, 2)}</span></div>
      <div class="coverage-track"><div class="coverage-fill" style="width:${coverage.risk_weighted_coverage_rate * 100}%"></div></div>
    </div>`).join("");

  document.querySelector("#hotspot-rows").innerHTML = data.government.nearest_camera_matches.map((row) => `
    <tr><td>#${row.risk_rank}</td><td>${escapeHtml(row.representative_location)}</td><td>${formatNumber(row.risk_score)}</td><td>${escapeHtml(row.nearest_camera_location)}</td><td>${formatNumber(row.distance_m)} m</td></tr>`).join("");
}

function renderReadiness(data, filter = "all") {
  const readiness = data.readiness;
  document.querySelector("#readiness-summary").innerHTML = ["verified", "pilot", "external"].map((state) => `
    <div class="readiness-count ${state}"><span>${stateLabels[state]}</span><strong>${readiness[state]}</strong></div>`).join("");
  const items = readiness.items.filter((item) => filter === "all" || item.state === filter);
  document.querySelector("#readiness-list").innerHTML = items.map((item) => `
    <div class="readiness-item"><div><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.detail)}</span></div>${statusLabel(item.state)}</div>`).join("");
}

function setupInteractions(data) {
  document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item, .view").forEach((element) => element.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.view}-view`).classList.add("active");
    history.replaceState(null, "", `#${button.dataset.view}`);
    document.querySelector("#main-content").focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }));

  document.querySelectorAll(".segment").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderModels(data, button.dataset.split);
  }));

  document.querySelectorAll(".evidence-tab").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".evidence-tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderEvidence(data, button.dataset.evidence);
  }));

  document.querySelectorAll(".readiness-filter").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".readiness-filter").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderReadiness(data, button.dataset.state);
  }));

  const initialView = location.hash.replace("#", "");
  const initialButton = document.querySelector(`.nav-item[data-view="${initialView}"]`);
  if (initialButton) initialButton.click();
}

async function initialize() {
  try {
    const response = await fetch("data/project_snapshot.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    document.querySelector("#snapshot-status").textContent = `Snapshot v${data.schema_version} · ${data.readiness.items.length} validation checks`;
    renderOverview(data);
    renderModels(data);
    renderEvidence(data);
    renderAblations(data);
    renderDeployment(data);
    renderReadiness(data);
    setupInteractions(data);
  } catch (error) {
    document.querySelector("#load-error").hidden = false;
    document.querySelector("#snapshot-status").textContent = "成果快照載入失敗";
    console.error(error);
  }
}

initialize();
