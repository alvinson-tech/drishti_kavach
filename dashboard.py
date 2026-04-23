import streamlit as st
import json
import time
import base64
import numpy as np
from PIL import Image
import io
import os

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Drishti Kavach — Control Room",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── DARK THEME CSS ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #050a0f !important;
        color: #c8d8e8 !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(rgba(0,255,200,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,200,0.02) 1px, transparent 1px),
            #050a0f;
        background-size: 40px 40px;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding: 1rem 2rem !important; max-width: 100% !important; }

    /* Header */
    .dk-header {
        background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        padding: 16px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 0 30px rgba(0, 180, 255, 0.08);
    }

    .dk-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        color: #00d4ff;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        margin: 0;
    }

    .dk-subtitle {
        font-size: 0.85rem;
        color: #4a7fa5;
        letter-spacing: 2px;
        margin: 2px 0 0 0;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Alert banner */
    .alert-banner-danger {
        background: linear-gradient(90deg, #3d0000, #1a0000);
        border: 1px solid #ff3333;
        border-left: 4px solid #ff3333;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 16px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        color: #ff6666;
        text-shadow: 0 0 10px rgba(255,50,50,0.6);
        animation: pulse-red 1.5s infinite;
        text-align: center;
        letter-spacing: 2px;
    }

    .alert-banner-safe {
        background: linear-gradient(90deg, #002a1a, #001a0f);
        border: 1px solid #00aa55;
        border-left: 4px solid #00cc66;
        border-radius: 6px;
        padding: 14px 20px;
        margin-bottom: 16px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        color: #00ff88;
        text-align: center;
        letter-spacing: 2px;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 10px rgba(255,50,50,0.3); }
        50% { box-shadow: 0 0 25px rgba(255,50,50,0.7); }
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0a1628, #0d1f3c);
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        height: 100%;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        transition: border-color 0.3s;
    }

    .metric-card:hover { border-color: #00d4ff44; }

    .metric-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: #4a7fa5;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.9rem;
        font-weight: bold;
        color: #00d4ff;
        text-shadow: 0 0 15px rgba(0,212,255,0.4);
        line-height: 1;
    }

    .metric-value-green { color: #00ff88; text-shadow: 0 0 15px rgba(0,255,136,0.4); }
    .metric-value-red   { color: #ff4444; text-shadow: 0 0 15px rgba(255,68,68,0.4); }
    .metric-value-yellow{ color: #ffcc00; text-shadow: 0 0 15px rgba(255,204,0,0.4); }

    .metric-unit {
        font-size: 0.7rem;
        color: #4a7fa5;
        margin-top: 2px;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Section headers */
    .section-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.7rem;
        color: #4a7fa5;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #1a3a5c;
        padding-bottom: 6px;
        margin-bottom: 12px;
        margin-top: 8px;
    }

    /* Feed container */
    .feed-container {
        background: #020609;
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    }

    .feed-label {
        background: rgba(5, 10, 15, 0.8);
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: #4a7fa5;
        letter-spacing: 2px;
        padding: 6px 12px;
        border-bottom: 1px solid #1a3a5c;
    }

    /* Detection log */
    .log-entry {
        background: #0a1628;
        border: 1px solid #1a3a5c;
        border-left: 3px solid #ff4444;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .log-time { color: #4a7fa5; }
    .log-class { color: #ff8888; font-weight: bold; }
    .log-conf { color: #ffcc00; }

    .log-empty {
        text-align: center;
        color: #2a4a6a;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        padding: 20px;
        letter-spacing: 2px;
    }

    /* Performance bars */
    .perf-bar-container {
        background: #0a1628;
        border: 1px solid #1a3a5c;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .perf-bar-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: #4a7fa5;
        letter-spacing: 1px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .perf-bar-track {
        background: #050a0f;
        border-radius: 3px;
        height: 6px;
        overflow: hidden;
    }

    .perf-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #00d4ff, #0088ff);
        box-shadow: 0 0 8px rgba(0,212,255,0.4);
    }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00ff88;
        box-shadow: 0 0 8px #00ff88;
        margin-right: 8px;
        animation: blink 1.5s infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Class breakdown */
    .class-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #1a3a5c;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
    }

    .class-name { color: #c8d8e8; }
    .class-count {
        background: #0d1f3c;
        color: #00d4ff;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.7rem;
    }

    /* Streamlit overrides */
    [data-testid="stImage"] img {
        border-radius: 0 !important;
        width: 100% !important;
    }

    div[data-testid="column"] { gap: 0 !important; }
    .stSpinner { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

STATE_FILE = "static/session_data.json"

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return None

def decode_frame(b64_str):
    try:
        if not b64_str:
            return None
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except:
        return None

def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def metric_card(label, value, unit="", color="default"):
    color_class = {
        "green": "metric-value-green",
        "red": "metric-value-red",
        "yellow": "metric-value-yellow",
        "default": "metric-value"
    }.get(color, "metric-value")

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{color_class}">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """

def perf_bar(label, value, max_val, unit):
    pct = min(100, (value / max_val) * 100) if max_val > 0 else 0
    color = "#ff4444" if pct > 80 else "#ffcc00" if pct > 50 else "#00d4ff"
    return f"""
    <div class="perf-bar-container">
        <div class="perf-bar-label">
            <span>{label}</span>
            <span style="color:{color}">{value} {unit}</span>
        </div>
        <div class="perf-bar-track">
            <div class="perf-bar-fill" style="width:{pct}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
        </div>
    </div>
    """

# ─── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="dk-header">
    <div>
        <p class="dk-title">🚆 DRISHTI KAVACH</p>
        <p class="dk-subtitle">AI-ENHANCED RAILWAY SAFETY SYSTEM — LIVE CONTROL ROOM</p>
    </div>
    <div style="text-align:right; font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#4a7fa5;">
        <div><span class="status-dot"></span>SYSTEM ONLINE</div>
        <div style="margin-top:4px;">BiSeNet + YOLO11</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LOOP ────────────────────────────────────────────
placeholder = st.empty()

while True:
    state = load_state()

    with placeholder.container():
        if state is None:
            st.markdown("""
            <div style="text-align:center; padding: 80px; font-family:'Share Tech Mono',monospace; color:#4a7fa5;">
                <div style="font-size:3rem; margin-bottom:20px;">⏳</div>
                <div style="font-size:1rem; letter-spacing:3px; color:#00d4ff;">WAITING FOR MAIN.PY TO START</div>
                <div style="font-size:0.7rem; margin-top:10px; letter-spacing:2px;">Run: python main.py in another terminal</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            continue

        # ── Alert Banner ──
        if state.get("alert"):
            st.markdown('<div class="alert-banner-danger">⚠️  KAVACH ALERT — OBSTACLE DETECTED ON TRACK  ⚠️</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-banner-safe">✅  ALL CLEAR — TRACK IS SAFE</div>', unsafe_allow_html=True)

        # ── Top Metrics Row ──
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(metric_card("LIVE FPS", state.get("avg_fps", 0), "frames/sec", "green"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("TOTAL DETECTIONS", state.get("total_detections", 0), "obstacles", "red" if state.get("total_detections", 0) > 0 else "default"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("ALERT RATE", f"{state.get('alert_rate', 0)}", "%", "yellow" if state.get("alert_rate", 0) > 10 else "default"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("AVG CONFIDENCE", f"{state.get('avg_confidence', 0)}", "%", "green"), unsafe_allow_html=True)
        with c5:
            st.markdown(metric_card("TRACK COVERAGE", f"{state.get('avg_track_coverage', 0)}", "%", "default"), unsafe_allow_html=True)
        with c6:
            st.markdown(metric_card("DURATION", format_duration(state.get("duration_seconds", 0)), "hh:mm:ss", "default"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

        # ── Main Content ──
        left, right = st.columns([3, 2])

        with left:
            # Live Feed
            st.markdown('<div class="section-header">📹 LIVE CAMERA FEED</div>', unsafe_allow_html=True)
            frame_img = decode_frame(state.get("frame_b64", ""))
            if frame_img:
                st.image(frame_img, use_container_width=True)
            else:
                st.markdown("""
                <div style="background:#020609; border:1px solid #1a3a5c; border-radius:8px;
                            height:300px; display:flex; align-items:center; justify-content:center;
                            font-family:'Share Tech Mono',monospace; color:#2a4a6a; letter-spacing:2px;">
                    NO FEED AVAILABLE
                </div>
                """, unsafe_allow_html=True)

            # Session Info Row
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            i1, i2, i3 = st.columns(3)
            with i1:
                st.markdown(metric_card("FRAMES CAPTURED", state.get("total_frames_captured", 0), "total"), unsafe_allow_html=True)
            with i2:
                st.markdown(metric_card("FRAMES PROCESSED", state.get("total_frames_processed", 0), "by AI"), unsafe_allow_html=True)
            with i3:
                st.markdown(metric_card("ALERT FRAMES", state.get("total_alert_frames", 0), "triggered", "red" if state.get("total_alert_frames", 0) > 0 else "default"), unsafe_allow_html=True)

        with right:
            # Performance
            st.markdown('<div class="section-header">⚡ MODEL PERFORMANCE</div>', unsafe_allow_html=True)
            st.markdown(perf_bar("BiSeNet Inference", state.get("avg_bisenet_ms", 0), 200, "ms"), unsafe_allow_html=True)
            st.markdown(perf_bar("YOLO11 Inference", state.get("avg_yolo_ms", 0), 200, "ms"), unsafe_allow_html=True)
            total_ms = state.get("avg_bisenet_ms", 0) + state.get("avg_yolo_ms", 0)
            st.markdown(perf_bar("Total Latency", round(total_ms, 1), 400, "ms"), unsafe_allow_html=True)

            # System Info
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#0a1628; border:1px solid #1a3a5c; border-radius:6px; padding:12px 16px;
                        font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#4a7fa5; line-height:1.8;">
                <div><span style="color:#c8d8e8">CAMERA</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {state.get('camera_mode','—')}</div>
                <div><span style="color:#c8d8e8">DEVICE</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {state.get('device','—').upper()}</div>
                <div><span style="color:#c8d8e8">SEGMODEL</span> &nbsp;&nbsp; BISENETV2</div>
                <div><span style="color:#c8d8e8">DETMODEL</span> &nbsp;&nbsp; YOLO11N</div>
                <div><span style="color:#c8d8e8">STARTED</span> &nbsp;&nbsp;&nbsp; {state.get('start_time','—')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Detection Classes
            detection_classes = state.get("detection_classes", {})
            if detection_classes:
                st.markdown('<div class="section-header" style="margin-top:12px">🎯 OBSTACLES BY CLASS</div>', unsafe_allow_html=True)
                sorted_classes = sorted(detection_classes.items(), key=lambda x: x[1], reverse=True)
                rows_html = ""
                for cls, count in sorted_classes:
                    rows_html += f"""
                    <div class="class-row">
                        <span class="class-name">▶ {cls.upper()}</span>
                        <span class="class-count">{count}x</span>
                    </div>
                    """
                st.markdown(f'<div style="background:#0a1628; border:1px solid #1a3a5c; border-radius:6px; padding:8px 16px;">{rows_html}</div>', unsafe_allow_html=True)

            # Detection Log
            st.markdown('<div class="section-header" style="margin-top:12px">📋 DETECTION LOG</div>', unsafe_allow_html=True)
            log = state.get("detection_log", [])
            if log:
                log_html = ""
                for entry in reversed(log[-10:]):
                    log_html += f"""
                    <div class="log-entry">
                        <span class="log-time">{entry.get('time','')}</span>
                        <span class="log-class">{entry.get('class','').upper()}</span>
                        <span class="log-conf">{entry.get('confidence',0)}%</span>
                    </div>
                    """
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.markdown('<div class="log-empty">— NO DETECTIONS YET —</div>', unsafe_allow_html=True)

    time.sleep(1)