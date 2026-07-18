from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> int | float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def _typed_rows(path: Path) -> list[dict[str, Any]]:
    return [{key: _number(value) for key, value in row.items()} for row in _read_csv(path)]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_project_snapshot(repo_root: Path) -> dict[str, Any]:
    """Build the dashboard contract from checked-in evaluation artifacts."""
    results = repo_root / "docs" / "results"
    assets = repo_root / "docs" / "assets"

    event_profiles = _typed_rows(results / "accident_event_metrics.csv")
    model_rows = _typed_rows(results / "model_fair_comparison.csv")
    ablations = _typed_rows(results / "sam2_vlm_ablation.csv")
    deployment = _load_json(results / "deployment_metrics.json")
    government = _load_json(assets / "government_risk_integration.json")
    nafnet = _load_json(assets / "nafnet_cctv_demo" / "report.json")
    sm3det = _load_json(assets / "sm3det_transfer_audit.json")

    deployment_models = [
        row for row in model_rows if row["operating_point"] == "train_holdout_fpr20"
    ]
    coverage = government["risk_camera_coverage"]["coverage_by_radius_m"]
    scenarios = government["scenario_only_not_measured_outcome"]
    readiness_items = [
        {"name": "紅線違停事件案例", "state": "verified", "detail": "規則、隱私處理與證據輸出已整合"},
        {"name": "事故候選與時序觸發", "state": "verified", "detail": "100 clips 正式事件評估"},
        {"name": "TCN / VideoMAE 比較", "state": "verified", "detail": "相同 500 clips 與 FPR 校準"},
        {"name": "SAM2 proposal evidence", "state": "pilot", "detail": "10 clips 同樣本消融"},
        {"name": "VLM evidence review", "state": "pilot", "detail": "clean-blind 限制已量測"},
        {"name": "政府正常 CCTV", "state": "pilot", "detail": "0.0867 camera-hours"},
        {"name": "政府 A1/A2 空間整合", "state": "verified", "detail": "393,854 records 與 17 camera sites"},
        {"name": "Verilog trigger modules", "state": "verified", "detail": "3 / 3 testbenches 通過"},
        {"name": "SystemC multi-camera queue", "state": "verified", "detail": "49,320 frames 與 Python parity"},
        {"name": "TCN ONNX 匯出", "state": "verified", "detail": "128 samples parity"},
        {"name": "NAFNet 困難畫面復原", "state": "verified", "detail": "真實 GPU restoration"},
        {"name": "SM3Det 多感測器適用性", "state": "verified", "detail": "官方 release 與 transfer audit"},
        {"name": "獨立正常 CCTV 長時驗證", "state": "external", "detail": "需累積更多 camera-hours"},
        {"name": "實體 Snapdragon profile", "state": "external", "detail": "需 QAI Hub credential 或裝置"},
        {"name": "部署前後成效驗證", "state": "external", "detail": "需政府重複月份資料"}
    ]
    readiness_counts = {
        state: sum(item["state"] == state for item in readiness_items)
        for state in ("verified", "pilot", "external")
    }

    return {
        "schema_version": 1,
        "project": {
            "name": "基於多模態時序感知與邊緣運算之道路交通事件偵測與風險評估系統",
            "positioning": "edge-first traffic incident detection and risk prioritization",
            "dataset": "ACCIDENT fixed-CCTV benchmark",
        },
        "readiness": {
            **readiness_counts,
            "items": readiness_items,
        },
        "pipeline": [
            {"name": "CCTV ingest", "state": "verified", "detail": "video / stream / public CCTV"},
            {"name": "Candidate proposal", "state": "verified", "detail": "YOLO + ByteTrack + motion"},
            {"name": "Temporal trigger", "state": "verified", "detail": "causal TCN on edge"},
            {"name": "Mask evidence", "state": "pilot", "detail": "SAM2 on difficult candidates"},
            {"name": "Evidence review", "state": "pilot", "detail": "VLM summary and triage"},
            {"name": "Risk deployment", "state": "verified", "detail": "A1/A2 hotspot prioritization"}
        ],
        "event_profiles": event_profiles,
        "model_comparison": deployment_models,
        "ablations": ablations,
        "deployment": deployment,
        "government": {
            "records": government["government_records"],
            "coverage": coverage,
            "nearest_camera_matches": government["risk_camera_coverage"]["nearest_camera_matches"][:6],
            "scenarios": scenarios,
        },
        "restoration": {
            "method": nafnet["method"],
            "latency_ms": nafnet["latency_ms"],
            "degraded_psnr_db": nafnet["degraded"]["psnr_db"],
            "restored_psnr_db": nafnet["restored"]["psnr_db"],
            "degraded_ssim": nafnet["degraded"]["ssim"],
            "restored_ssim": nafnet["restored"]["ssim"],
        },
        "sm3det": {
            "release_verified": sm3det["release_verified"],
            "modalities": sm3det["modalities"],
            "parameters_m": sm3det["official_release_metrics"]["parameters_m"],
            "flops_g": sm3det["official_release_metrics"]["flops_g"],
            "role": sm3det["cctv_transfer_assessment"]["recommended_role"],
        },
        "media": {
            "reference_video": "../docs/assets/accident_reference.mp4",
            "model_output_video": "../docs/assets/candidate_incident_model_output.mp4",
            "vlm_evidence": "../docs/assets/incident_vlm_clean_evidence.jpg",
            "nafnet_comparison": "../docs/assets/nafnet_cctv_demo/comparison.jpg",
            "sm3det_architecture": "../docs/assets/sm3det_architecture.png"
        },
        "reports": {
            "final": "../docs/final_evaluation_report.md",
            "status": "../docs/project_status.md",
            "spec": "../spec.md"
        }
    }


def write_project_snapshot(repo_root: Path, output: Path) -> dict[str, Any]:
    snapshot = build_project_snapshot(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot
