# Drishti Kavach: Project Overview & System Architecture

---

## 1. What is Drishti Kavach?

**Drishti Kavach** (Hindi: *Shield of Vision*) is a real-time AI-powered railway safety system designed to detect obstacles on or near railway tracks and alert operators before a collision can occur.

It combines two state-of-the-art deep learning models running in parallel:

- **BiSeNetV2** — performs pixel-level semantic segmentation to identify exactly where the railway track is in each camera frame.
- **YOLO11m** — performs real-time object detection to identify people, animals, and vehicles in the frame.

By combining these two models, the system can determine not just *that* an obstacle exists, but precisely *whether it is on the track* — and how close it is.

---

## 2. The Problem It Solves

Railway accidents caused by obstacles on tracks (animals, people, vehicles at level crossings) are a major safety concern globally. Traditional camera systems rely on human operators watching feeds manually, which is:

- **Slow** — human reaction time introduces dangerous delays
- **Unreliable** — attention fatigue leads to missed events
- **Unscalable** — one operator cannot watch dozens of track feeds simultaneously

Drishti Kavach automates this entirely. It:
1. Continuously watches the camera feed at up to ~30 FPS
2. Segments the track region in real time using BiSeNetV2
3. Detects obstacles using YOLO11m
4. Cross-references the two results to determine risk level
5. Displays alerts instantly on both an OpenCV window and a Streamlit dashboard

---

## 3. System Architecture

The system is built around a **producer-consumer multi-threaded pipeline**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DRISHTI KAVACH                           │
│                                                                 │
│   ┌────────────────┐         ┌────────────────────────────┐    │
│   │  Camera/Image  │──frame──▶   model_runner thread       │    │
│   │  Reader Thread │         │                            │    │
│   └────────────────┘         │  Every Nth frame:          │    │
│                               │  1. BiSeNetV2 → track mask │    │
│   Sources:                    │  2. YOLO11m  → detections  │    │
│   • Webcam (live)             │  3. Risk classification    │    │
│   • Static image              │  4. Overlay rendered       │    │
│                               │  5. JSON state written     │    │
│                               └──────────┬─────────────────┘   │
│                                          │ latest_output        │
│                               ┌──────────▼─────────────────┐   │
│                               │   Display Loop (main thread)│   │
│                               │   cv2.imshow()              │   │
│                               │   S = snapshot, Q = quit    │   │
│                               └──────────┬─────────────────┘   │
│                                          │ session_data.json    │
│                               ┌──────────▼─────────────────┐   │
│                               │  Streamlit Dashboard        │   │
│                               │  (separate process)         │   │
│                               │  Live metrics, feed, log    │   │
│                               └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Thread Roles

| Thread | Role |
|--------|------|
| `webcam_reader` / `image_reader` | Captures raw frames from camera or image file, writes to `latest_raw` |
| `model_runner` | Reads raw frames, runs BiSeNetV2 + YOLO11m, writes processed output to `latest_output` |
| Main thread | Reads `latest_output`, displays via OpenCV, handles keyboard input |
| Streamlit (separate process) | Reads `static/session_data.json` every second, renders dashboard |

Threads share data through `latest_raw` and `latest_output` variables protected by `threading.Lock()` objects (`raw_lock`, `out_lock`) to prevent race conditions.

---

## 4. The Two Modes of Operation

### Mode A — Standard (`main.py`)
Uses the **RailSem19-trained** BiSeNet weights (`bisenet_railsem19.pth`). Works on general railway track images and live webcam feeds from real train-mounted cameras.

### Mode B — Hardware (`main_hardware.py`)
Uses **custom fine-tuned** BiSeNet weights (`bisenet_hardware.pth`) trained specifically on images captured from the team's physical hardware prototype camera. This adapts the model to the specific camera angle, lighting conditions, and lens characteristics of the prototype device.

---

## 5. Risk Classification Logic

The heart of the safety system is a 4-level risk classifier that runs on every detected object:

```
For each obstacle detected by YOLO:

  Is there a track detected?
  │
  ├── YES
  │   ├── Does bottom-half of bounding box overlap track pixels?
  │   │   ├── YES (overlap ≥ class threshold) → ON_TRACK  🔴  FULL ALERT
  │   │   └── NO  → measure foot-point distance to nearest track pixel
  │   │             ├── dist < 80px   → TOO_CLOSE  🟠  SEMI ALERT
  │   │             ├── dist < 213px  → NEAR        🟡  Warning
  │   │             └── dist ≥ 213px  → FAR         🟢  Safe
  │
  └── NO TRACK DETECTED → NEAR (yellow) — monitor surroundings
```

### Per-class overlap thresholds

Each class has a tuned threshold — smaller/thinner objects need less overlap to trigger an alert, while large flat objects require more.

**`main.py` — standard deployment:**

| Category | Class(es) | Threshold |
|----------|-----------|-----------|
| People | person | 5% |
| Vehicles | bicycle, motorcycle | 3% |
| Vehicles | car, truck, bus | 10% |
| Street furniture | bench | 10% |
| Animals (medium) | cat, dog, horse, sheep, cow, zebra | 8% |
| Animals (large) | elephant, bear, giraffe | 5% |
| Carried items | backpack, handbag, suitcase | 8% |
| Carried items | umbrella | 5% |
| Indoor objects | chair, couch, bed | 10% |
| Indoor objects | laptop | 8% |

**`main_hardware.py` — hardware prototype deployment:**

| Category | Class(es) | Threshold |
|----------|-----------|-----------|
| People | person | 5% |
| Vehicles | bicycle, motorcycle | 3% |
| Vehicles | car, truck, bus | 10% |
| Animals | cat, dog | 8% |
| Carried items | backpack, handbag | 8% |
| Carried items | umbrella | 5% |
| Small objects | bottle, cup, bowl, book | 5% |
| Small objects | potted plant | 8% |
| Small objects | knife, spoon, mouse, remote, cell phone, scissors, toothbrush | 3% |

---

## 6. Key Design Decisions

| Decision | Reason |
|----------|--------|
| BiSeNetV2 for segmentation | Fastest real-time semantic segmentor with railway-specific pre-training on RailSem19 |
| YOLO11m for detection | Best speed/accuracy tradeoff for real-time multi-class obstacle detection |
| Process every Nth frame | Running both models on every frame is too expensive; N=3 triples throughput while output stays smooth |
| Streamlit dashboard via JSON file | Fully decouples the real-time inference process from the web UI; no shared memory needed |
| Apple MPS / CUDA / CPU auto-detect | Makes the system portable — works on MacBook (MPS), GPU server (CUDA), or any machine (CPU) |

---

## 7. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Computer Vision | OpenCV 4.x |
| Deep Learning | PyTorch 2.x |
| Segmentation Model | BiSeNetV2 |
| Detection Model | YOLO11m (Ultralytics) |
| Dashboard | Streamlit |
| Hardware Acceleration | Apple MPS / CUDA / CPU (auto-detected) |
| Data Augmentation | Albumentations |

---
