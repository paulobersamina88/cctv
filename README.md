
# CCTV AI Dashboard PRO

A Streamlit-based local CCTV dashboard for Tapo C200, Xiaomi cameras where RTSP/ONVIF is available, and other RTSP-compatible cameras.

## Features

- Multi-camera dashboard
- RTSP live video feed
- YOLO person detection
- AI-assisted baby fall-risk heuristic
- Configurable danger zones
- Event logging with snapshot capture
- Optional Telegram alerts
- CSV export of alert history

## Important Safety Note

This app is only an AI-assisted monitoring system. It may miss events or produce false alarms. It is not a substitute for adult supervision, baby gates, proper furniture placement, or certified safety devices.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tapo C200 RTSP Example

```text
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
```

Low-resolution stream:

```text
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream2
```

## Xiaomi Note

Xiaomi camera support depends on exact model and firmware. If your Xiaomi model does not expose RTSP/ONVIF, consider Home Assistant or go2rtc as a bridge.

## Telegram Alert Setup

1. Open Telegram.
2. Search BotFather.
3. Create a bot and copy the token.
4. Get your chat ID.
5. Enter bot token and chat ID in the Streamlit sidebar.


## Streamlit Cloud Note

This fixed version uses `opencv-python-headless` to avoid the common `import cv2` error on Streamlit Cloud.

However, Streamlit Cloud usually cannot access CCTV cameras using local IP addresses such as `192.168.1.xxx`. For real CCTV viewing, run this app on a local PC/Raspberry Pi connected to the same WiFi as your cameras.

Recommended local command:

```bash
pip install -r requirements.txt
streamlit run app.py
```
