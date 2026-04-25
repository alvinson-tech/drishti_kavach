# Drishti Kavach: Dashboard & Session Reporting

---

## 1. Overview

Drishti Kavach has two visual interfaces running simultaneously:

| Interface | File | Technology | Purpose |
|-----------|------|-----------|---------|
| **OpenCV Window** | `main.py` | OpenCV `imshow` | Real-time annotated camera feed for the local operator |
| **Streamlit Dashboard** | `dashboard.py` | Streamlit + HTML/CSS | Web-based control room for remote monitoring and analytics |

The two processes communicate exclusively through a shared JSON file:

```
main.py  ──writes──▶  static/session_data.json  ──reads──▶  dashboard.py
```

This file-based decoupling means:
- The dashboard can be run on a different machine on the same network
- Crashing the dashboard never affects the safety system
- Any other tool (scripts, web apps) can also consume the JSON

---

## 2. How to Run the Dashboard

In a **separate terminal** while `main.py` is running:

```bash
streamlit run dashboard.py
```

Streamlit opens automatically in the default browser at `http://localhost:8501`.

If `main.py` is not running yet, the dashboard shows a waiting screen:
```
⏳
WAITING FOR MAIN.PY TO START
Run: python main.py in another terminal
```

---

## 3. Communication Protocol: `session_data.json`

Every **0.5 seconds**, `main.py` writes a JSON snapshot to `static/session_data.json`. The dashboard reads this file every **1 second** and updates the UI.

### Full JSON Schema

```json
{
  "running"                : true,
  "timestamp"              : "2024-04-22 14:30:05",
  "start_time"             : "2024-04-22 14:20:01",
  "duration_seconds"       : 604.1,
  "camera_mode"            : "Kreo Owl Lite FHD 1080p",
  "alert"                  : false,
  "avg_fps"                : 12.4,
  "total_frames_captured"  : 18120,
  "total_frames_processed" : 6040,
  "total_detections"       : 3,
  "total_alert_frames"     : 2,
  "alert_rate"             : 0.03,
  "avg_confidence"         : 78.2,
  "avg_track_coverage"     : 18.5,
  "avg_bisenet_ms"         : 95.3,
  "avg_yolo_ms"            : 112.7,
  "detection_classes"      : { "person": 2, "dog": 1 },
  "detection_log"          : [
    {
      "time"       : "14:25:03",
      "class"      : "person",
      "confidence" : 87.2,
      "status"     : "ON TRACK"
    }
  ],
  "frame_b64"              : "...(base64 encoded JPEG)...",
  "device"                 : "mps"
}
```

### Frame Encoding

The current processed frame is encoded as a Base64 JPEG (quality=70) and embedded in the JSON:

```python
_, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
frame_b64 = base64.b64encode(buf).decode('utf-8')
```

Quality 70 is a balance — sufficient visual clarity for the dashboard feed without making the JSON file excessively large (typically ~50–100KB per write).

The dashboard decodes it:
```python
img_bytes = base64.b64decode(b64_str)
img = Image.open(io.BytesIO(img_bytes))
st.image(img, use_container_width=True)
```

---

## 4. Dashboard Design & Theme

The dashboard uses a **custom dark military/tech aesthetic** implemented entirely in CSS injected via `st.markdown(..., unsafe_allow_html=True)`.

### Design System

| Element | Style |
|---------|-------|
| Background | Near-black `#050a0f` with subtle cyan grid overlay |
| Font | `Rajdhani` (body) + `Share Tech Mono` (numbers/labels) — imported from Google Fonts |
| Accent colour | Cyan `#00d4ff` with glow effects (`text-shadow`) |
| Cards | Dark blue gradient `#0a1628 → #0d1f3c` with `#1a3a5c` borders |
| Alert state | Animated pulsing red glow (`@keyframes pulse-red`) |
| Safe state | Static green glow |
| Status dot | Blinking green dot (`@keyframes blink`) next to "SYSTEM ONLINE" |

### Hidden Streamlit UI Elements

Default Streamlit chrome (menu, footer, toolbar) is hidden to give a clean full-screen control room feel:

```css
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
```

---

## 5. Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🚆 DRISHTI KAVACH      AI-ENHANCED RAILWAY SAFETY SYSTEM       │
│  (header bar with glow effect)          ● SYSTEM ONLINE        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ⚠️ KAVACH ALERT — OBSTACLE DETECTED ON TRACK ⚠️  (red pulse)  │
│    OR                                                           │
│  ✅ ALL CLEAR — TRACK IS SAFE  (green)                          │
│                                                                 │
├──────┬──────┬──────┬──────┬──────┬──────────────────────────────┤
│ FPS  │ DET  │ALERT │CONF  │COVER │ DURATION                    │
│      │      │RATE  │      │AGE   │                             │
├──────┴──────┴──────┴──────┴──────┴──────────────────────────────┤
│                                                                 │
│  Left panel (60%)              Right panel (40%)               │
│  ─────────────────             ───────────────                 │
│  📹 LIVE CAMERA FEED           ⚡ MODEL PERFORMANCE            │
│  [frame image]                 [BiSeNet bar]                   │
│                                [YOLO bar]                      │
│                                [Total latency bar]             │
│  ─────────────────             ───────────────                 │
│  FRAMES    FRAMES    ALERT     [System Info box]               │
│  CAPTURED  PROCESSED FRAMES    Camera / Device / Models        │
│                                ───────────────                 │
│                                🎯 OBSTACLES BY CLASS           │
│                                [class breakdown]               │
│                                ───────────────                 │
│                                📋 DETECTION LOG                │
│                                [recent detections]             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Top Metrics Row — 6 KPI Cards

The top row shows six key performance indicators, each as a styled card:

| Card | Value Source | Colour Logic |
|------|-------------|-------------|
| **LIVE FPS** | `avg_fps` (last 30 samples) | Always green |
| **TOTAL DETECTIONS** | `total_detections` | Red if > 0, default otherwise |
| **ALERT RATE** | `alert_rate` (%) | Yellow if > 10%, default otherwise |
| **AVG CONFIDENCE** | `avg_confidence` (%) | Always green |
| **TRACK COVERAGE** | `avg_track_coverage` (%) | Always default cyan |
| **DURATION** | `duration_seconds` formatted as `HH:MM:SS` | Always default cyan |

Cards use the `metric-card` CSS class with hover effects (`border-color: #00d4ff44` on hover).

---

## 7. Alert Banner

The most prominent UI element — a full-width banner below the header:

### Alert State (red pulsing)
```html
<div class="alert-banner-danger">
  ⚠️  KAVACH ALERT — OBSTACLE DETECTED ON TRACK  ⚠️
</div>
```
- Background: dark red gradient `#3d0000 → #1a0000`
- Left border: bright red `#ff3333`
- Text: red `#ff6666` with red glow
- Animation: pulsing red box-shadow (1.5s cycle)

### Safe State (green static)
```html
<div class="alert-banner-safe">
  ✅  ALL CLEAR — TRACK IS SAFE
</div>
```
- Background: dark green gradient `#002a1a → #001a0f`
- Left border: bright green `#00cc66`
- Text: bright green `#00ff88`

The banner updates every 1 second with the latest alert state from the JSON.

---

## 8. Live Camera Feed

The left panel shows the processed camera frame (with segmentation overlay, bounding boxes, and annotations) streamed from `main.py` via the Base64 JPEG in the JSON:

```python
frame_img = decode_frame(state.get("frame_b64", ""))
if frame_img:
    st.image(frame_img, use_container_width=True)
```

If no frame is available yet, a placeholder is shown:
```
NO FEED AVAILABLE
```

The image refreshes every 1 second (dashboard polling rate). This means the dashboard feed lags by up to ~1.5 seconds behind real-time (0.5s JSON write interval + 1s dashboard poll).

---

## 9. Model Performance Bars

Three horizontal progress bars show inference performance:

```
BiSeNet Inference  ████████░░░░░░░  95.3 ms   (max 200ms)
YOLO11 Inference   █████████░░░░░░  112.7 ms  (max 200ms)
Total Latency      █████████████░░  208.0 ms  (max 400ms)
```

Colour changes based on percentage of max:
- **Blue** `#00d4ff` — below 50% of max (good performance)
- **Yellow** `#ffcc00` — 50–80% of max (acceptable)
- **Red** `#ff4444` — above 80% of max (overloaded)

---

## 10. System Info Panel

A compact info block shows:

```
CAMERA     Kreo Owl Lite FHD 1080p
DEVICE     MPS
SEGMODEL   BISENETV2
DETMODEL   YOLO11N
STARTED    2024-04-22 14:20:01
```

---

## 11. Obstacles By Class

If any obstacles have been detected during the session, a breakdown is shown:

```
▶ PERSON          5x
▶ DOG             2x
▶ CAR             1x
```

Sorted by count (most frequent first). Only appears when `detection_classes` dict is non-empty.

---

## 12. Detection Log

The last 10 detection events are shown, newest first:

```
14:25:03    PERSON    87.2%
14:22:41    DOG       43.0%
14:21:15    PERSON    91.5%
```

Each entry shows:
- **Time** — HH:MM:SS when the detection occurred (blue)
- **Class** — obstacle class in uppercase (red)
- **Confidence** — detection confidence percentage (yellow)

Only ON_TRACK and TOO_CLOSE events appear in the log. The log is capped at the last 50 entries in the JSON (`detection_log[-50:]`), and the dashboard shows the most recent 10.

---

## 13. Dashboard Update Loop

The dashboard runs an infinite `while True` loop:

```python
placeholder = st.empty()

while True:
    state = load_state()   # read session_data.json

    with placeholder.container():
        if state is None:
            # show waiting screen
            time.sleep(1)
            continue

        # render all UI elements ...

    time.sleep(1)   # poll every 1 second
```

Using `st.empty()` with `.container()` allows the entire UI to be replaced atomically each second without page flicker. Without this, Streamlit would append new content below existing content on each iteration.

---

## 14. Console Session Report (on Quit)

When `Q` is pressed in the OpenCV window, `main.py` prints a detailed session report. This is separate from the dashboard and gives a permanent record in the terminal:

### Report Sections

| Section | Content |
|---------|---------|
| 📅 SESSION INFO | Start/end times, duration, camera type, mode, resolution |
| 🤖 MODELS USED | BiSeNetV2 weights file, YOLO version, obstacle classes, device |
| 📊 FRAME STATISTICS | Total frames captured/processed, processing rate, average FPS |
| ⚡ PERFORMANCE METRICS | Average BiSeNet ms, average YOLO ms, combined latency |
| 🛤️ BISENET TRACK SEGMENTATION | Average track coverage %, segmentation stability (std dev) |
| 🎯 YOLO OBSTACLE DETECTION | Total detections, alert frames, alert rate %, confidence stats |
| 📈 ACCURACY EXPLANATION | Plain-English explanation of what each metric means |

The **Accuracy Explanation** section is intentionally written for non-technical readers:
```
Confidence Score: YOLO11m's certainty that a detected object
is what it says it is (0-100%). A score of 80% means the
model is 80% sure the detected object is correct.

Track Coverage: % of the camera frame identified as railway
track by BiSeNet. Stable coverage = consistent segmentation.

Alert Rate: % of processed frames where an obstacle was
found inside the track region.
```

---

## 15. Data Flow Summary

```
Camera/Image
    │
    │ raw frame (1920×1080 or image)
    ▼
webcam_reader / image_reader thread
    │
    │ latest_raw (via raw_lock)
    ▼
model_runner thread
    │
    ├── Every Nth frame:
    │   ├── BiSeNetV2 → mask + overlay
    │   ├── YOLO11m → detections
    │   ├── Risk classification → annotated frame
    │   └── Session stats updated
    │
    ├── latest_output (via out_lock) ──────────────▶ OpenCV display loop
    │                                                  │
    │                                                cv2.imshow()
    │                                                  │
    │                                               S → snapshot
    │                                               Q → quit
    │
    └── Every 0.5s: session_data.json ──────────────▶ Streamlit dashboard
                                                       │
                                                     Polls every 1s
                                                     Renders metrics,
                                                     feed, log, bars
```

---

## 16. Extending the Dashboard

Because the communication is pure JSON, the dashboard can be extended easily:

- **Add a new metric** in `main.py`'s `write_state()` → add a new key to the JSON → read it in `dashboard.py`
- **Historical charts** — store detection log to a CSV and use `st.line_chart()`
- **Multi-camera** — run multiple `main.py` instances writing to different JSON files; `dashboard.py` can read both
- **Alerts to phone** — add a `requests.post()` call in `write_state()` to trigger a webhook when `alert=True`

---