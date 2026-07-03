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
### 未來展望

目前版本已經可以完成基本的影像上傳與違停辨識。後續希望把它整理成更接近真實場景的監視器端系統：攝影機持續輸入影像，系統先在本地端篩出疑似違規事件，再交給 VLM 或人工審核做後續判斷。

這個方向的重點不是把所有模型都堆進來，而是把「違停」從單張圖片判斷，整理成一個可以被測試、統計、解釋的事件流程。

---
### 目前完成的延伸工作

目前新增了一組比較乾淨的事件判定與模擬核心，放在：

`Application/DebugVersion/src/illegal_parking`

它可以先在沒有實體監視器或 edge 硬體的情況下，用圖片資料夾、影片或 webcam 模擬輸入來源，並測試事件判定邏輯。

另外也新增了：

- `scripts/run_edge_simulation.py`：用命令列跑模擬輸入來源。
- `hardware_sim/`：用 Verilog / SystemC 的方式描述未來可能硬體化的事件過濾邏輯。
- `tests/`：目前用 pytest 驗證 Python 事件引擎、edge 模擬與硬體模擬測試向量。

---
### Phase 1：軟體端事件模擬與統計

這個階段先把現有 YOLO / VLM demo 整理成可測試的軟體流程。

目前已完成：

1. 建立事件資料結構，例如車輛框、追蹤狀態、停留時間、紅線/ROI 判斷。
2. 建立違停候選事件判定邏輯。
3. 建立圖片資料夾、影片、webcam 的模擬輸入介面。
4. 建立初步統計指標，例如 precision、recall、false positive rate。
5. 保留原本 GUI demo，新增的核心邏輯先獨立測試。

模擬執行範例：

```powershell
python scripts\run_edge_simulation.py --source image-folder --path data\samples\frames --source-fps 30 --target-fps 5
```

測試：

```powershell
python -m pytest
```

目前測試結果：

```text
16 passed
```

---
### Phase 2：硬體感知模擬

這個階段先不急著把 AI 模型放進硬體，而是挑出比較適合硬體化的部分：紅線重疊統計、停留時間累積、事件狀態機。

目前先建立了：

- `hardware_sim/rtl/dwell_fsm.v`
- `hardware_sim/rtl/bbox_overlap_counter.v`
- `hardware_sim/systemc/event_pipeline_sim.cpp`
- `hardware_sim/python_golden/dwell_fsm_model.py`
- `hardware_sim/test_vectors/dwell_fsm_vectors.json`

這部分的目標是讓專題可以多一個角度：除了 Python / YOLO / VLM，也能說明哪些邏輯適合放在 edge accelerator 或 FPGA-like pipeline 裡先做過濾，減少後端 AI 模型的負擔。

現階段 pytest 會先用 Python golden model 檢查測試向量。之後如果有時間，可以再接 Verilator、Icarus Verilog、SystemC 或 cocotb 做更完整的 RTL 模擬。

---
### 使用到的語言與工具

- Python：主流程、事件判定、模擬、評估、測試。
- OpenCV / NumPy：影像與影片處理。
- YOLO：車輛偵測。
- VLM：後續做違規事件的二次審查與說明。
- PySide6：目前 GUI demo。
- pytest：測試事件邏輯與模擬流程。
- Verilog：描述可硬體化的事件過濾模組。
- SystemC / C++：描述較高階的硬體/系統模擬流程。
