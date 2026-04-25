
# Home CCTV Dashboard - Streamlit Cloud Safe

This package is designed to install successfully on Streamlit Cloud.

## Requirements

```txt
streamlit
```

No OpenCV. No Ultralytics. No Torch.

## Important Limitation

Streamlit Cloud cannot normally access home CCTV RTSP cameras using local IP addresses like:

```text
192.168.1.xxx
```

For real CCTV viewing and AI detection, run the OpenCV version locally on the same WiFi as the cameras.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
