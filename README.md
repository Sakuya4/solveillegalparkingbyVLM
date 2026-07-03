# solveillegalparkingbyVLM
### 2025高通台灣AI黑客松比賽題目
YOLO做車輛檢測，經過NAFNet的模糊處理，最後提供給VLM去判斷是否違規。

如果畫面無法辨識(Bad case)，會由SM3Det介入協助影像處理。

本專案目的希望可以將SM3Det的無人機角度，改變成一般監視器之角度，以應用在交通違規處理。

---
### 團隊成員資訊

蔡霆鋒 janalexei88@gmail.com

辛語柔 yujouhsin@gmail.com

周佳欣 jassinchouxd@gmail.com


---
### 專案介紹

1. 攝影機及其輔助裝置優先採取邊緣運算的裝置設備，例如：ESP32, raspberry pi 等性能較低之設備

2. 架構上採混合雲，透過ESP32-CAM收集到的畫面，傳到雲端，經系統處理後，產生摘要，最後交由USER確定是否開立罰單

<img width="1146" height="643" alt="image" src="https://github.com/user-attachments/assets/9ed41312-32ab-4fab-a5f3-b95b308a0c0f" />
<img width="1146" height="644" alt="image" src="https://github.com/user-attachments/assets/9024c9d7-5e1e-459b-a2aa-6ae1dea65bc9" />

---
### Demo
<img width="1108" height="319" alt="image" src="https://github.com/user-attachments/assets/186939f9-38ed-4c36-b833-38ea244dd503" />

---
### 安裝說明

1. 建立虛擬環境
   
`python -m venv venv`

2. 啟動虛擬環境

PowerShell:

`.\venv\Scripts\Activate.ps1`

cmd:

`.\venv\Scripts\activate.bat`

3. 安裝需要的套件

`pip install -r Application\DebugVersion\requirements.txt`

---
### 執行與使用說明

1. 開啟終端機

2. 進入程式資料夾

`cd Application/DebugVersion/src`

3. 執行程式檔

`python main.py`

4. 在 GUI 中上傳照片

5. 開始分析

---
### 專題升級方向：Edge 即時違停事件偵測

本專案後續規劃從原本的單張圖片 YOLO + VLM demo，升級為可模擬政府機構監視器或 edge 裝置部署的「違規事件輔助偵測與蒐證系統」。

核心定位不是全自動開罰，而是：

1. 在 edge 端即時偵測車輛、紅線或禁停區。
2. 透過追蹤器計算車輛停留時間與靜止狀態。
3. 只在疑似違規事件發生時建立 evidence package。
4. 由 VLM 作為二次審查與說明產生器。
5. 最後交由人工審核確認是否成立。

這樣可以降低 VLM 成本、避免逐幀上雲，也讓每個判定都有可解釋的依據。

---
### 目前新增功能

目前已新增一組可測試的 edge 模擬與事件判定核心，放在：

`Application/DebugVersion/src/illegal_parking`

新增內容包含：

1. `event_models.py`
   - 定義 `BBox`, `TrackSnapshot`, `RuleEvidence`, `ViolationCandidate` 等事件資料結構。

2. `violation_engine.py`
   - 根據車輛類別、YOLO 信心分數、停留時間、紅線重疊比例、ROI 狀態判斷是否成為違規候選事件。

3. `frame_sources.py`
   - 支援模擬 edge input：
     - `SyntheticFrameSource`
     - `ImageFolderSource`
     - `VideoFileSource`
     - `WebcamSource`
   - 也提供 `EdgeProfile`，可模擬低 FPS 或低算力 edge 裝置。

4. `simulation.py`
   - 可讀取 frame source 並輸出處理幀數摘要。

5. `metrics.py`
   - 提供事件層級的 TP / FP / FN / TN、precision、recall、false positive rate。

---
### Edge 模擬執行方式

目前可以在沒有實體監視器或 edge 硬體的情況下，用資料夾或影片模擬輸入來源。

範例：使用圖片資料夾模擬監視器串流

```powershell
python scripts\run_edge_simulation.py --source image-folder --path data\samples\frames --source-fps 30 --target-fps 5
```

範例：使用影片檔模擬監視器串流

```powershell
python scripts\run_edge_simulation.py --source video --path data\samples\demo.mp4 --source-fps 30 --target-fps 10
```

輸出會是 JSON 摘要，例如：

```json
{
  "camera_id": "image_folder",
  "source_type": "image_folder",
  "profile_name": "jetson_or_mini_pc_sim",
  "total_frames": 100,
  "processed_frames": 17,
  "skipped_frames": 83
}
```

---
### 測試與目前結果統計

目前新增了 pytest 測試，主要驗證：

- 違規候選事件判定
- 停留時間與紅線/ROI 條件
- edge profile 的降 FPS 行為
- frame source 的讀取流程
- 模擬 runner 的統計摘要
- precision / recall / false positive rate 計算

執行測試：

```powershell
python -m pytest
```

目前測試結果：

```text
14 passed
```

---
### 後續功能展示規劃

後續展示畫面會以「事件審核」為主，而不是只顯示模型框線：

1. 原始監視器畫面
2. YOLO 車輛偵測框
3. 紅線或禁停區 mask
4. 車輛停留時間
5. 疑似違規事件列表
6. VLM 審查理由
7. 人工審核結果：確認、駁回、證據不足

---
### 後續模型比較規劃

模型不會一次全部放進主線，而是分階段比較：

1. Baseline
   - YOLO + 紅線 overlap + 停留時間

2. VLM Reviewer
   - BLIP-2 作為既有 baseline
   - Qwen3-VL 或 InternVL 作為 open-source VLM 比較
   - GPT / Gemini 類雲端 VLM 僅在資料政策允許時比較

3. Bad Case Enhancement
   - NAFNet：只用於模糊、低光、壓縮嚴重影像的輔助辨識
   - SM3Det：只用於高角度、小目標、遠距監視器場景 fallback

---
### 重要設計原則

- 原始影像永遠保留，影像增強結果只作為模型輔助。
- VLM 不直接作為唯一裁判，只負責二次審查與理由生成。
- 系統輸出疑似違規事件，最終仍由人工確認。
- 不逐幀呼叫 VLM，只在事件候選成立時呼叫。
- 所有判定都需要可追溯的 evidence package。
