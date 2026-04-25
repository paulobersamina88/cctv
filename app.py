
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Home CCTV Dashboard - Cloud Safe", layout="wide")

st.title("🏠 Home CCTV Dashboard - Streamlit Cloud Safe")
st.caption("This version has no OpenCV dependency, so requirements install cleanly on Streamlit Cloud.")

st.warning(
    "Cloud limitation: Streamlit Cloud usually cannot access local CCTV RTSP cameras such as "
    "192.168.x.x. For real CCTV + AI detection, run the OpenCV version locally on the same WiFi."
)

st.sidebar.header("Camera Settings")

camera_name = st.sidebar.text_input("Camera name", "Living Room")
camera_url = st.sidebar.text_input(
    "Camera browser-accessible URL",
    "http://your-camera-or-nvr-url"
)

view_mode = st.sidebar.selectbox(
    "View mode",
    ["Browser iframe / HTTP camera page", "RTSP instruction only"]
)

st.subheader("📹 Camera Viewer")

if view_mode == "Browser iframe / HTTP camera page":
    st.info("Use this only if your camera/NVR has a browser-accessible HTTP/HTTPS page.")
    if camera_url and camera_url.startswith(("http://", "https://")):
        st.components.v1.iframe(camera_url, height=600, scrolling=True)
    else:
        st.error("Please enter an HTTP or HTTPS URL.")
else:
    st.info(
        "RTSP links like rtsp://username:password@192.168.1.100:554/stream1 "
        "cannot be displayed directly by Streamlit Cloud/browser."
    )
    st.code("rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1")

st.divider()

st.subheader("🚨 Manual Safety Log")

alert_type = st.selectbox(
    "Observation",
    [
        "Normal",
        "Baby near chair/edge",
        "Possible fall-risk situation",
        "Motion observed",
        "Other"
    ]
)

notes = st.text_area("Notes")

if st.button("Save observation"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success(f"Saved observation: {timestamp} - {camera_name} - {alert_type}")
    st.write(notes)

st.divider()

st.subheader("Recommended Deployment")

st.markdown("""
### Best setup for your Xiaomi / Tapo CCTV

For actual CCTV + AI detection:

1. Install Python on a laptop or mini PC at home.
2. Connect it to the same WiFi as your CCTV.
3. Use the local OpenCV version.
4. Access the dashboard inside your home network.

### Why this cloud-safe version exists

This package is made to avoid installation failure on Streamlit Cloud by using only:

```txt
streamlit
```

No `opencv-python`, no `opencv-python-headless`, no `ultralytics`, no Torch.
""")
