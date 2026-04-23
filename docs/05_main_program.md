# Drishti Kavach — Part 5: main.py Feature Deep-Dive

---

## 1. Overview

`main.py` is the **standard runtime** of Drishti Kavach. It ties together every component of the system — camera input, BiSeNetV2 segmentation, YOLO11m detection, risk classification, visual output, session tracking, and dashboard communication — into a single continuously running program.

Run it with:
```bash
python main.py
```

---

## 2. Configuration Block

At the top of `main.py`, all tunable parameters are defined as constants so they can be changed without digging through code:

```python
CAMERA_MODE      = "webcam"        # "webcam" or "image"
WEBCAM_INDEX     = 0               # camera device index
TEST_IMAGE_PATH  = "test_images/1.jpg"
WEIGHTS_PATH     = "weights/bisenet_railsem19.pth"
PROCESS_EVERY_N  = 3               # run models on 1 in every N frames
DISPLAY_SIZE     = (1280, 720)     # output window resolution (16:9)
STATE_FILE       = "static/session_data.json"
SAVE_OUTPUT      = False           # set True to record MP4
```

### Camera Modes

| Mode | Behaviour |
|------|-----------|
| `"webcam"` | Opens the Kreo Owl Lite FHD 1080p webcam at index 0, streams live |
| `"image"` | Prompts for a test image number at startup, loops that image as a static feed |

**Image mode prompt** — when `CAMERA_MODE = "image"`, the program pauses at startup and asks:
```
Enter test image number (e.g. 1 for 1.jpg):
```
This allows testing with different images without editing the source file. If the file doesn't exist, it re-prompts until a valid file is given.

---

## 3. Model Loading

Both models are loaded once at startup before any threads are started:

```python
# YOLO11m
yolo = YOLO("yolo11m.pt")

# BiSeNetV2
bisenet_config = BiSeNetV2Config()
segmentor = RailtrackSegmentationHandler(
    path_to_snapshot = WEIGHTS_PATH,
    model_config     = bisenet_config,
    overlay_alpha    = 0.5
)
```

The handler auto-selects the compute device:
- **Apple MPS** — if running on Apple Silicon Mac
- **CUDA** — if an NVIDIA GPU is present
- **CPU** — fallback for any machine

A startup summary is printed to console:
```
════════════════════════════════════════════════════
      Camera        : Kreo Owl Lite FHD 1080p
      Camera Mode   : WEBCAM
      Display Size  : (1280, 720)
      Process Rate  : Every 3 frames
      Save Output   : False
════════════════════════════════════════════════════
```

---

## 4. Thread Architecture

Three concurrent execution contexts run simultaneously:

```
┌─────────────────────────────────────────────────────────┐
│  Thread 1: webcam_reader / image_reader                 │
│  ─────────────────────────────────────────────────────  │
│  • Opens camera / loads image                           │
│  • Captures frames continuously                         │
│  • Writes latest frame to latest_raw (protected by      │
│    raw_lock)                                            │
│  • Increments total_frames_captured counter             │
└────────────────────────┬────────────────────────────────┘
                         │ latest_raw
┌────────────────────────▼────────────────────────────────┐
│  Thread 2: model_runner                                 │
│  ─────────────────────────────────────────────────────  │
│  • Reads latest_raw, sets it to None (consumes frame)  │
│  • Every Nth frame: runs BiSeNetV2 + YOLO11m           │
│  • Other frames: reuses last processed output           │
│  • Writes to latest_output (protected by out_lock)     │
│  • Every 0.5s: writes session_data.json                │
└────────────────────────┬────────────────────────────────┘
                         │ latest_output
┌────────────────────────▼────────────────────────────────┐
│  Main Thread: display loop                              │
│  ─────────────────────────────────────────────────────  │
│  • Reads latest_output                                  │
│  • Calls cv2.imshow()                                   │
│  • Handles S (snapshot) and Q (quit) keys              │
│  • Optionally writes video frames to VideoWriter        │
└─────────────────────────────────────────────────────────┘
```

### Why this architecture?

Without threading, the camera, model inference, and display would run sequentially — the display would freeze for every model inference cycle (~100–300ms). With threading, the camera always captures at full speed, and the display always updates smoothly, regardless of model speed.

### Thread Safety

```python
raw_lock = threading.Lock()   # protects latest_raw
out_lock = threading.Lock()   # protects latest_output

# Writer:
with raw_lock:
    latest_raw = frame

# Reader:
with raw_lock:
    frame = latest_raw
    latest_raw = None   # consume the frame
```

Using `None` as a sentinel: once `model_runner` reads `latest_raw`, it sets it to `None`. The next iteration checks `if frame is None: continue` to avoid processing the same frame twice.

---

## 5. Webcam Reader Thread

```python
def webcam_reader():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # 30-frame warmup to let camera auto-adjust
    for _ in range(30):
        cap.read()
```

**Key details:**
- Captures at native **1080p** resolution, then resizes to 1280×720 in the model runner — this preserves maximum image quality for detection
- **30-frame warmup** — discards the first 30 frames to allow the camera's auto-exposure and auto-focus to stabilise. Without this, early frames are often dark or blurry
- `AUTOFOCUS=1` and `AUTO_EXPOSURE=1` — let the camera handle focus and exposure automatically

---

## 6. Image Reader Thread

When `CAMERA_MODE = "image"`, this simpler thread runs instead:

```python
def image_reader():
    frame = cv2.imread(TEST_IMAGE_PATH)
    while running:
        session["total_frames_captured"] += 1
        with raw_lock:
            latest_raw = frame.copy()
        time.sleep(0.1)   # 10 FPS feed rate
```

It loops the same image at 10 FPS, simulating a live feed. This allows the full processing pipeline (segmentation, detection, risk classification) to run and be tuned without needing a live camera.

---

## 7. Model Runner Thread — The Processing Core

This is the most complex part of the system. It handles all AI inference and frame rate management:

```python
def model_runner():
    frame_count    = 0
    last_processed = None
    last_time      = time.time()
    last_write     = time.time()

    while running:
        # 1. Consume latest raw frame
        with raw_lock:
            frame = latest_raw
            latest_raw = None
        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        # 2. Calculate and record FPS
        now = time.time()
        fps = 1.0 / (now - last_time)
        session["fps_samples"].append(fps)
        last_time = now

        # 3. Resize to display resolution
        resized = cv2.resize(frame, DISPLAY_SIZE)

        # 4. Process every Nth frame
        if frame_count % PROCESS_EVERY_N == 0:
            session["total_frames_processed"] += 1
            overlay, mask, track_detected = segment_track(resized)
            output = detect_obstacles(overlay, mask, track_detected)
            last_processed = output.copy()
            with out_lock:
                latest_output = output

            # 5. Write dashboard JSON every 0.5s
            if now - last_write >= 0.5:
                write_state(output)
                last_write = now
        else:
            # Reuse last processed frame
            if last_processed is not None:
                with out_lock:
                    latest_output = last_processed.copy()
```

### Frame skipping logic

`PROCESS_EVERY_N = 3` means:
- Frame 1: skip (reuse last output)
- Frame 2: skip (reuse last output)
- Frame 3: **process** (run BiSeNetV2 + YOLO11m)
- Frame 4: skip
- ...

This gives the models 3× more time to complete inference before the next frame is due, effectively tripling the sustainable throughput. The display is still updated with a valid (slightly stale) output on skipped frames, so it appears smooth.

---

## 8. Segment Track Function

```python
def segment_track(frame):
    t_start = time.time()
    mask, overlay = segmentor.run(frame, only_mask=False)
    bisenet_time = (time.time() - t_start) * 1000
```

Steps inside this function:

1. **Run BiSeNetV2** via the handler → produces `mask` (2D class indices) and `overlay` (colour blended frame)
2. **Calculate track coverage** — percentage of frame pixels that are class 0 or 1
3. **Record timing** — BiSeNet inference ms logged to `session["bisenet_times"]`
4. **Resize mask** to match frame dimensions (BiSeNet runs at 512×1024 internally)
5. **Track detected flag** — True if coverage ≥ 1%
6. If track detected: **run connected components** to find and number individual track regions
7. **Draw "Track N" labels** on the overlay at the bottom of each track region

Returns: `(overlay, mask_resized, track_detected)`

---

## 9. Detect Obstacles Function

After segmentation, this function runs YOLO and performs risk classification:

### Step-by-step flow:

```
1. Run YOLO11m on the overlay frame (which already has segmentation colours)
2. Extract pre-computed track coordinates: np.where(mask_full == 1)
3. For each detection:
   a. Filter to obstacle classes only
   b. Clamp box coordinates to frame bounds
   c. Extract bottom half of bounding box
   d. Count track pixels in bottom half → overlap_pct
   e. Compare to per-class threshold → on_track flag
   f. If not on_track: compute foot-point distance to nearest track pixel
   g. Assign risk level (ON_TRACK / TOO_CLOSE / NEAR / FAR)
   h. Draw coloured bounding box + label
   i. Update session stats if alert level
4. Draw status banner
5. Draw timestamp (bottom-right)
6. Draw camera label (bottom-left)
```

### Status Banner

A full-width coloured banner is drawn at the top of the frame (y=0 to y=60):

| Condition | Banner Background | Text |
|-----------|------------------|------|
| No track detected | Dark blue-grey `(20, 80, 120)` | `NO TRACK DETECTED — MONITORING SURROUNDINGS` |
| ON_TRACK alert | Dark red `(0, 0, 180)` | `!! KAVACH ALERT: OBSTACLE ON TRACK !!` |
| TOO_CLOSE alert | Dark orange-red `(40, 40, 180)` | `SEMI-KAVACH ALERT: OBSTACLES TOO CLOSE TO TRACK!` |
| All clear | Dark green `(0, 120, 0)` | `TRACK CLEAR` |

The banner text is always **horizontally centred** using `cv2.getTextSize()` to measure text width before drawing.

### Frame Annotations Summary

Every processed frame has:
- **Top banner** — status at a glance
- **Coloured segmentation overlay** — track region shown in colour (50% opacity blend)
- **"Track N" labels** — at the base of each track region
- **Bounding boxes** — coloured by risk level
- **Object labels** — class name + confidence + risk prefix
- **Timestamp** — bottom-right corner (format: `YYYY-MM-DD HH:MM:SS`)
- **Camera label** — bottom-left corner (`Kreo Owl Lite FHD` or `Static Image`)

---

## 10. Session Statistics Tracking

A `session` dictionary accumulates statistics for the entire runtime:

```python
session = {
    "start_time"             : None,
    "end_time"               : None,
    "total_frames_captured"  : 0,
    "total_frames_processed" : 0,
    "total_alert_frames"     : 0,
    "total_detections"       : 0,
    "detection_confidences"  : [],   # list of floats
    "detection_classes"      : {},   # {"person": 5, "car": 2, ...}
    "track_coverages"        : [],   # list of floats (%)
    "bisenet_times"          : [],   # list of ms values
    "yolo_times"             : [],   # list of ms values
    "fps_samples"            : [],   # list of fps values
    "detection_log"          : [],   # list of detection events with timestamps
}
```

Only **ON_TRACK** and **TOO_CLOSE** detections are counted in `total_detections` and added to the detection log. FAR and NEAR are tracked but not escalated.

---

## 11. Dashboard JSON State (`write_state`)

Every 0.5 seconds, the model runner calls `write_state()` which serialises the current session state to `static/session_data.json`:

```json
{
  "running": true,
  "timestamp": "2024-04-22 14:30:05",
  "alert": false,
  "avg_fps": 12.4,
  "total_frames_captured": 1200,
  "total_frames_processed": 400,
  "total_detections": 3,
  "total_alert_frames": 2,
  "alert_rate": 0.5,
  "avg_confidence": 78.2,
  "avg_track_coverage": 18.5,
  "avg_bisenet_ms": 95.3,
  "avg_yolo_ms": 112.7,
  "detection_classes": {"person": 2, "dog": 1},
  "detection_log": [...],
  "frame_b64": "...",
  "device": "mps"
}
```

The `frame_b64` field contains the current processed frame encoded as a Base64 JPEG string (quality=70), which the Streamlit dashboard decodes and displays as the live feed image.

---

## 12. Display Loop & User Controls

The main thread runs the display loop:

```python
while True:
    with out_lock:
        display = latest_output.copy()

    cv2.imshow("Drishti Kavach", display)

    if SAVE_OUTPUT and video_writer:
        video_writer.write(display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):    # Save snapshot
        path = f"snapshots/snapshot_{time.strftime('%Y-%m-%d_%H%M')}.jpg"
        cv2.imwrite(path, display)

    if key == ord('q'):    # Quit
        running = False
        session["end_time"] = time.time()
        break
```

| Key | Action |
|-----|--------|
| `S` | Saves current display frame as `snapshots/snapshot_YYYY-MM-DD_HHMM.jpg` |
| `Q` | Sets `running = False` (stops all threads gracefully), records end time, breaks loop |

---

## 13. Video Recording

If `SAVE_OUTPUT = True`, an MP4 video is recorded:

```python
OUTPUT_PATH = f"captures/capture_{time.strftime('%Y-%m-%d_%H%M')}.mp4"
fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 10.0, DISPLAY_SIZE)
```

- Format: MP4 with `mp4v` codec
- Frame rate: 10 FPS (reflects the ~10 processed frames per second typical throughput)
- Resolution: 1280×720
- Saved to: `captures/` directory

---

## 14. Session Report (`print_report`)

When the user presses Q, after cleanup, a detailed session report is printed to the console:

```
════════════════════════════════════════════════════
         DRISHTI KAVACH — SESSION REPORT
════════════════════════════════════════════════════

📅  SESSION INFO
    Start Time          : 2024-04-22 14:20:01
    End Time            : 2024-04-22 14:30:05
    Total Duration      : 00:10:04
    Camera              : Kreo Owl Lite FHD 1080p
    Display Resolution  : 1280x720

🤖  MODELS USED
    Segmentation Model  : BiSeNetV2
    Detection Model     : YOLO11m (trained on COCO)
    Obstacle Classes    : person, cat, dog, ...

📊  FRAME STATISTICS
    Total Frames Captured   : 18000
    Total Frames Processed  : 6000
    Average FPS             : 29.8 fps

⚡  PERFORMANCE METRICS
    Avg BiSeNet Inference   : 95.3 ms/frame
    Avg YOLO11m Inference   : 112.7 ms/frame
    Avg Total Latency       : 208.0 ms/frame

🛤️   BISENET — TRACK SEGMENTATION
    Avg Track Coverage      : 18.50% of frame
    Segmentation Stability  : ±2.31% (std deviation)

🎯  YOLO11m — OBSTACLE DETECTION
    Total Detections        : 3
    Alert Rate              : 0.5% of processed frames
    Avg Detection Confidence: 78.2%
```

The report includes **explanations** of what each metric means, so non-technical operators can understand it.

---

## 15. Startup Sequence Summary

```
1. Parse CAMERA_MODE config
2. If image mode → prompt for image number
3. Load YOLO11m weights
4. Load BiSeNetV2 weights + handler
5. Print config summary
6. Create output directories (static/, snapshots/, captures/)
7. Optionally open VideoWriter
8. Start camera thread
9. Start model_runner thread
10. Block until first output frame is ready
11. Print "running" message with controls hint
12. Enter display loop
13. (User presses Q)
14. Set running=False → threads terminate
15. Release VideoWriter if active
16. Write final state to JSON
17. Print session report
18. Exit
```

---

*← [Part 4 — YOLO11m Architecture & Working](04_yolo11m.md)*
*→ Continue to [Part 6 — Hardware Pipeline](06_hardware_pipeline.md)*
