import streamlit as st
import cv2
import time
from ultralytics import YOLO

st.set_page_config(page_title="Home CCTV AI Dashboard", layout="wide")

st.title("🏠 Home CCTV AI Dashboard")
st.caption("Local CCTV viewing with AI-assisted fall-risk monitoring")

camera_url = st.sidebar.text_input(
    "Camera RTSP URL",
    "rtsp://username:password@192.168.1.100:554/stream1"
)

enable_ai = st.sidebar.checkbox("Enable AI detection", value=True)
confidence = st.sidebar.slider("AI confidence", 0.1, 0.9, 0.4)

danger_zone = st.sidebar.checkbox("Show danger-zone concept", value=True)

model = YOLO("yolov8n.pt")

frame_placeholder = st.empty()
status_placeholder = st.empty()

if st.button("Start Camera"):
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        st.error("Cannot open camera stream. Check RTSP URL, username, password, and local network.")
    else:
        while True:
            ret, frame = cap.read()

            if not ret:
                st.warning("No frame received from camera.")
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (960, 540))

            alert = "Normal monitoring"

            if enable_ai:
                results = model(frame, conf=confidence, verbose=False)

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = model.names[cls]

                        if label == "person":
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "Person detected", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            height = y2 - y1
                            width = x2 - x1

                            if width > height * 1.2:
                                alert = "⚠️ Possible fall-like posture detected"

            if danger_zone:
                cv2.rectangle(frame, (600, 250), (900, 520), (0, 0, 255), 2)
                cv2.putText(frame, "Danger Zone", (600, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            status_placeholder.info(alert)

            time.sleep(0.03)
