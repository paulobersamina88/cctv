
import os
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from ultralytics import YOLO

APP_DIR = Path(__file__).parent
CONFIG_PATH = APP_DIR / "camera_config.json"
EVENT_DIR = APP_DIR / "events"
SNAPSHOT_DIR = EVENT_DIR / "snapshots"
EVENT_LOG = EVENT_DIR / "event_log.csv"

EVENT_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="CCTV AI Dashboard PRO", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cameras": [], "telegram": {"enabled": False, "bot_token": "", "chat_id": ""}}

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def ensure_event_log():
    if not EVENT_LOG.exists():
        pd.DataFrame(columns=[
            "datetime", "camera", "alert_type", "message", "snapshot"
        ]).to_csv(EVENT_LOG, index=False)

def log_event(camera_name, alert_type, message, frame=None):
    ensure_event_log()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path = ""

    if frame is not None:
        snapshot_path = str(SNAPSHOT_DIR / f"{timestamp}_{camera_name.replace(' ', '_')}.jpg")
        cv2.imwrite(snapshot_path, frame)

    row = pd.DataFrame([{
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera_name,
        "alert_type": alert_type,
        "message": message,
        "snapshot": snapshot_path
    }])
    row.to_csv(EVENT_LOG, mode="a", header=False, index=False)
    return snapshot_path

def send_telegram(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False, "Missing Telegram bot token or chat ID."
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        return r.ok, r.text[:200]
    except Exception as e:
        return False, str(e)

def point_in_rect(x, y, rect):
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2

def draw_zone(frame, zone, label="Danger Zone"):
    h, w = frame.shape[:2]
    x1 = int(zone["x1"] * w)
    y1 = int(zone["y1"] * h)
    x2 = int(zone["x2"] * w)
    y2 = int(zone["y2"] * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return (x1, y1, x2, y2)

def analyze_frame(frame, model, camera_cfg, conf=0.4):
    annotated = frame.copy()
    alert_messages = []
    fall_risk = False
    danger_zone_alert = False

    zones = camera_cfg.get("danger_zones", [])
    rects = []
    for z in zones:
        rects.append(draw_zone(annotated, z, z.get("name", "Danger Zone")))

    results = model(annotated, conf=conf, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            score = float(box.conf[0])

            if label != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(
                annotated,
                f"person {score:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Heuristic fall-risk: person bbox unusually horizontal.
            if height > 0 and width / height > camera_cfg.get("fall_aspect_ratio", 1.20):
                fall_risk = True
                alert_messages.append("Possible fall-risk posture detected.")

            # Danger zone: person center or feet near manually defined zone.
            foot_x = cx
            foot_y = y2
            for rect in rects:
                if point_in_rect(cx, cy, rect) or point_in_rect(foot_x, foot_y, rect):
                    danger_zone_alert = True
                    alert_messages.append("Person/baby detected inside danger zone.")

    return annotated, fall_risk, danger_zone_alert, list(set(alert_messages))

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

# -----------------------------
# UI
# -----------------------------
cfg = load_config()
model = load_yolo()

st.title("🏠 CCTV AI Dashboard PRO")
st.caption("Multi-camera home monitoring with AI-assisted baby fall-risk and danger-zone alerts.")

with st.sidebar:
    st.header("System")
    conf = st.slider("AI confidence", 0.10, 0.90, 0.40, 0.05)
    refresh_sec = st.slider("Frame refresh interval, seconds", 0.05, 2.00, 0.25, 0.05)
    enable_ai = st.checkbox("Enable AI analysis", True)
    save_alert_snapshots = st.checkbox("Save alert snapshots", True)

    st.divider()
    st.header("Telegram Alerts")
    cfg["telegram"]["enabled"] = st.checkbox("Enable Telegram", cfg.get("telegram", {}).get("enabled", False))
    cfg["telegram"]["bot_token"] = st.text_input("Bot token", cfg.get("telegram", {}).get("bot_token", ""), type="password")
    cfg["telegram"]["chat_id"] = st.text_input("Chat ID", cfg.get("telegram", {}).get("chat_id", ""))

    if st.button("Save settings"):
        save_config(cfg)
        st.success("Settings saved.")

tab_live, tab_config, tab_events, tab_help = st.tabs([
    "📹 Live Dashboard", "⚙️ Camera Setup", "🚨 Event Log", "📘 Guide"
])

# -----------------------------
# Camera setup tab
# -----------------------------
with tab_config:
    st.subheader("Camera Setup")

    st.markdown("""
    **Typical Tapo C200 RTSP format**
    ```
    rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
    ```
    Use `stream2` for lower bandwidth.
    """)

    with st.form("add_camera"):
        name = st.text_input("Camera name", "Living Room")
        url = st.text_input("RTSP URL", "rtsp://username:password@192.168.1.100:554/stream1")
        fall_ratio = st.slider("Fall posture aspect ratio threshold", 1.0, 2.5, 1.2, 0.05)

        st.markdown("**Danger Zone Coordinates**")
        st.caption("Use normalized coordinates from 0 to 1. Example: x1=0.60, y1=0.45, x2=0.95, y2=0.95 for right-lower chair area.")
        zone_name = st.text_input("Zone name", "Chair / Edge Zone")
        x1 = st.number_input("x1", 0.0, 1.0, 0.60, 0.01)
        y1 = st.number_input("y1", 0.0, 1.0, 0.45, 0.01)
        x2 = st.number_input("x2", 0.0, 1.0, 0.95, 0.01)
        y2 = st.number_input("y2", 0.0, 1.0, 0.95, 0.01)

        submitted = st.form_submit_button("Add camera")
        if submitted:
            cfg["cameras"].append({
                "name": name,
                "url": url,
                "enabled": True,
                "fall_aspect_ratio": fall_ratio,
                "danger_zones": [{
                    "name": zone_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                }]
            })
            save_config(cfg)
            st.success(f"Added {name}. Reload the app if needed.")

    st.divider()
    st.subheader("Current Cameras")
    if cfg.get("cameras"):
        for i, cam in enumerate(cfg["cameras"]):
            with st.expander(f"{i+1}. {cam.get('name', 'Camera')}"):
                st.json(cam)
                if st.button(f"Remove {cam.get('name','Camera')}", key=f"remove_{i}"):
                    cfg["cameras"].pop(i)
                    save_config(cfg)
                    st.warning("Camera removed. Reload the app.")
    else:
        st.info("No camera added yet.")

# -----------------------------
# Live dashboard
# -----------------------------
with tab_live:
    cameras = [c for c in cfg.get("cameras", []) if c.get("enabled", True)]
    if not cameras:
        st.warning("No camera configured. Add one in Camera Setup.")
    else:
        cols = st.columns(min(2, len(cameras)))
        placeholders = []

        for idx, cam in enumerate(cameras):
            with cols[idx % len(cols)]:
                st.subheader(cam.get("name", f"Camera {idx+1}"))
                placeholders.append(st.empty())

        start = st.button("Start PRO Monitoring")

        if start:
            caps = []
            for cam in cameras:
                cap = cv2.VideoCapture(cam["url"])
                caps.append(cap)

            last_alert_time = {cam["name"]: 0 for cam in cameras}
            alert_cooldown = 15

            while True:
                for idx, (cam, cap) in enumerate(zip(cameras, caps)):
                    ret, frame = cap.read()

                    if not ret:
                        placeholders[idx].error(f"{cam.get('name')} - no frame received.")
                        continue

                    frame = cv2.resize(frame, (960, 540))

                    messages = []
                    if enable_ai:
                        frame, fall_risk, danger_alert, messages = analyze_frame(frame, model, cam, conf=conf)

                        if messages:
                            now = time.time()
                            if now - last_alert_time[cam["name"]] > alert_cooldown:
                                msg = f"⚠️ CCTV AI Alert: {cam['name']} - " + " ".join(messages)
                                snapshot = log_event(
                                    cam["name"],
                                    "AI_ALERT",
                                    msg,
                                    frame if save_alert_snapshots else None
                                )
                                last_alert_time[cam["name"]] = now

                                if cfg.get("telegram", {}).get("enabled", False):
                                    send_telegram(
                                        cfg["telegram"].get("bot_token", ""),
                                        cfg["telegram"].get("chat_id", ""),
                                        msg
                                    )

                    status_text = "✅ Normal"
                    if messages:
                        status_text = "⚠️ " + " | ".join(messages)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    placeholders[idx].image(frame_rgb, channels="RGB", caption=status_text)

                time.sleep(refresh_sec)

# -----------------------------
# Event log
# -----------------------------
with tab_events:
    st.subheader("AI Event Log")
    ensure_event_log()

    df = pd.read_csv(EVENT_LOG)
    if df.empty:
        st.info("No alerts recorded yet.")
    else:
        st.dataframe(df.sort_values("datetime", ascending=False), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download event log CSV",
            csv,
            "cctv_ai_event_log.csv",
            "text/csv"
        )

        st.subheader("Latest Snapshot")
        latest = df.dropna().tail(1)
        if len(latest) > 0:
            snap = latest.iloc[-1].get("snapshot", "")
            if isinstance(snap, str) and snap and os.path.exists(snap):
                st.image(snap)

# -----------------------------
# Guide
# -----------------------------
with tab_help:
    st.subheader("Setup Guide")

    st.markdown("""
    ### 1. Tapo C200
    In the Tapo app, enable the camera account / RTSP access, then use:
    ```
    rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1
    ```
    For low bandwidth:
    ```
    rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream2
    ```

    ### 2. Xiaomi Cameras
    Xiaomi support depends on exact model and firmware. If RTSP/ONVIF is not available, consider:
    - Home Assistant integration
    - go2rtc bridge
    - Xiaomi Mi Home integration
    - Replacing with ONVIF/RTSP-supported camera for reliable local CCTV dashboard

    ### 3. Baby Fall-Risk AI
    This app uses simple computer vision heuristics:
    - person detected near defined danger zone
    - horizontal person box suggesting fall-like posture
    - event snapshot logging

    This is **not a substitute for adult supervision**. Use it as an additional warning system only.

    ### 4. Telegram Alert
    Create a bot using BotFather, get your bot token, then get your chat ID.
    Enter both in the sidebar and enable Telegram.
    """)

