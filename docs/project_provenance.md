# Project Provenance

## Origin

The project originated in the 2025 Qualcomm Taiwan AI Hackathon. Its first
application combined vehicle detection, image restoration, multimodal review,
and an ESP32-CAM acquisition concept for red-line parking evidence. The current
research extends that application to temporal incident detection, native laptop
operation, edge-runtime integration, and government-informed camera allocation.

## Original Team

| Contributor | Contact |
| --- | --- |
| 蔡霆鋒 | janalexei88@gmail.com |
| 辛語柔 | yujouhsin@gmail.com |
| 周佳欣 | jassinchouxd@gmail.com |

## Original System Description

The original deployment concept used an ESP32-CAM image source and a hybrid
edge/cloud architecture. A vehicle detector produced candidates; a multimodal
model summarized visual evidence for operator review. Red-line parking remains
the first rule-based event case in the extended system.

Original descriptions, retained for attribution:

> 攝影機及其輔助裝置優先採取邊緣運算的裝置設備，例如：ESP32, raspberry pi 等性能較低之設備

> 架構上採混合雲，透過ESP32-CAM收集到的畫面，傳到雲端，經系統處理後，產生摘要，最後交由USER確定是否開立罰單

## Original Demonstration Assets

![Original system interface](https://github.com/user-attachments/assets/9ed41312-32ab-4fab-a5f3-b95b308a0c0f)

![Original evidence interface](https://github.com/user-attachments/assets/9024c9d7-5e1e-459b-a2aa-6ae1dea65bc9)

![Original parking demonstration](https://github.com/user-attachments/assets/186939f9-38ed-4c36-b833-38ea244dd503)

The original application remains accessible through
`Application/DebugVersion/src/main.py`. The integrated native research
workbench is launched with `scripts/run_desktop.py`.
