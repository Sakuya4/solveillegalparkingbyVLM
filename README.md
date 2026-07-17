# solveillegalparkingbyVLM
### 2025高通台灣AI黑客松比賽題目
YOLO做車輛檢測，經過NAFNet的模糊處理，最後提供給VLM去判斷是否違規。

如果畫面無法辨識(Bad case)，會由SM3Det介入協助影像處理。

本專案目的希望可以將SM3Det的無人機角度，改變成一般監視器之角度，以應用在交通違規處理。

目前延續方向：面向智慧城市的交通違規熱區監測、事件審核與邊緣部署系統；紅線違停保留為第一個完整事件案例。

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
### 延伸任務參考影片

[![ACCIDENT 固定式道路監視器事故片段](docs/assets/accident_reference.gif)](docs/assets/accident_reference.mp4)

點擊預覽可播放 MP4。片段取自
[ACCIDENT 交通監視器事故資料集](https://github.com/accidentbench/ACCIDENT)，用於測試事件時間、位置與事故類型辨識；此衍生片段依原資料集的
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 條款提供。

---
### 線上事故候選區域

系統已能在不讀取事故標註框的情況下，以 YOLO/ByteTrack、畫面 motion 與可選的 SAM2 box prompt 產生候選區域。500 支 ACCIDENT 影片共產生 822 個視窗，零處理失敗。

| 500-clip 模型 | IID F1 | Geographic F1 |
| --- | ---: | ---: |
| Global motion logistic | 0.405 | 0.516 |
| Online candidate ROI logistic | 0.581 | 0.615 |
| Global causal TCN | 0.642 | 0.569 |
| Online candidate ROI TCN | 0.629 | 0.668 |

完整指標、FPR 與研究限制請見 [ACCIDENT Phase 1 results](docs/accident_phase1_results.md)。開發代理的研究誠信、測試與提交規範記錄於 [AGENTS.md](AGENTS.md)。

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
