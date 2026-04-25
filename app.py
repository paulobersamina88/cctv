
import streamlit as st
import cv2
import time

st.set_page_config(page_title="CCTV AI Dashboard LITE", layout="wide")

st.title("🏠 CCTV AI Dashboard (Lite)")
st.caption("Lightweight version with person detection and fall-risk heuristic")

camera_url = st.sidebar.text_input(
    "Camera RTSP URL",
    "rtsp://username:password@192.168.1.100:554/stream1"
)

fall_ratio = st.sidebar.slider("Fall aspect ratio threshold", 1.0, 2.5, 1.2)
show_zone = st.sidebar.checkbox("Show danger zone", True)

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

frame_placeholder = st.empty()
status_placeholder = st.empty()

if st.button("Start Camera"):
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("Cannot open camera stream.")
    else:
        while True:
            ret, frame = cap.read()

            if not ret:
                st.warning("No frame received.")
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (960, 540))
            alert = "Normal"

            h, w = frame.shape[:2]

            # Danger zone (example)
            zx1, zy1, zx2, zy2 = int(0.6*w), int(0.45*h), int(0.95*w), int(0.95*h)

            if show_zone:
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0,0,255), 2)

            boxes, _ = hog.detectMultiScale(frame, winStride=(8,8))

            for (x, y, bw, bh) in boxes:
                cx = x + bw//2
                cy = y + bh//2

                cv2.rectangle(frame, (x,y), (x+bw, y+bh), (0,255,0), 2)

                if bw > bh * fall_ratio:
                    alert = "⚠️ Possible fall-risk posture"

                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    alert = "⚠️ Person in danger zone"

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            status_placeholder.info(alert)

            time.sleep(0.03)
