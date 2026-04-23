# Drishti Kavach — Part 1: Project Overview & System Architecture

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

## 5. Project File Map

```
drishti_kavach/
│
├── main.py                  ← Standard runtime (RailSem19 weights)
├── main_hardware.py         ← Hardware-tuned runtime
├── dashboard.py             ← Streamlit live dashboard
│
├── capture_hardware.py      ← Step 1: Capture images from prototype camera
├── label_hardware.py        ← Step 2: Paint pixel masks on captured images
├── train_hardware.py        ← Step 3: Fine-tune BiSeNet on labelled images
│
├── models/
│   └── rail_marking/        ← BiSeNetV2 model code (segmentation handler, config)
│
├── weights/
│   ├── bisenet_railsem19.pth    ← Pre-trained on RailSem19 dataset
│   ├── bisenet_hardware.pth     ← Fine-tuned on hardware captures (best val)
│   └── bisenet_hardware_last.pth← Fine-tuned (last epoch)
│
├── yolo11m.pt               ← YOLO11m weights (trained on COCO)
│
├── hardware_captures/       ← Raw photos from prototype camera
├── hardware_dataset/        ← Labelled images + masks (output of label_hardware.py)
├── test_images/             ← Static test images for image mode
├── snapshots/               ← Saved snapshots (S key during runtime)
├── captures/                ← Saved video recordings
├── static/
│   └── session_data.json    ← Live state written by main.py, read by dashboard.py
│
└── requirements.txt         ← All Python dependencies
```

---

## 6. Risk Classification Logic

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

| Class | Threshold |
|-------|-----------|
| person | 5% |
| bicycle, motorcycle | 3% |
| car, truck, bus | 10% |
| cat, dog, cow, horse | 8% |

---

## 7. Key Design Decisions

| Decision | Reason |
|----------|--------|
| BiSeNetV2 for segmentation | Fastest real-time semantic segmentor with railway-specific pre-training on RailSem19 |
| YOLO11m for detection | Best speed/accuracy tradeoff for real-time multi-class obstacle detection |
| Process every Nth frame | Running both models on every frame is too expensive; N=3 triples throughput while output stays smooth |
| Streamlit dashboard via JSON file | Fully decouples the real-time inference process from the web UI; no shared memory needed |
| Apple MPS / CUDA / CPU auto-detect | Makes the system portable — works on MacBook (MPS), GPU server (CUDA), or any machine (CPU) |

---

## 8. Technology Stack

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

*Continue to → [Part 2 — Datasets & Data Pipeline](02_datasets.md)*
