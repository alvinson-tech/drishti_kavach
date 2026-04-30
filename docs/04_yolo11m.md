# Drishti Kavach: YOLO11m Architecture & Working

---

## 1. What is Object Detection?

While BiSeNetV2 tells us *where the track is*, YOLO11m tells us *where the obstacles are*.

**Object detection** is the task of finding objects in an image and drawing a bounding box around each one, along with a class label and confidence score.

```
Input Image                    YOLO11m Output
┌─────────────────────┐        ┌─────────────────────┐
│                     │        │  ┌───┐               │
│   [railway scene]   │ ─────▶ │  │ P │  person 87%  │
│                     │        │  └───┘               │
│                     │        │       ┌──────┐        │
│                     │        │       │ car  │ 92%   │
└─────────────────────┘        │       └──────┘        │
                               └─────────────────────┘
```

Each detection contains:
- **Bounding box** — `(x1, y1, x2, y2)` pixel coordinates
- **Class label** — what the object is (e.g., "person")
- **Confidence score** — how certain the model is (0.0 to 1.0)

---

## 2. YOLO — You Only Look Once

YOLO is a family of real-time object detection models. The key insight that makes YOLO fast is that it performs detection in a **single forward pass** through the network — unlike older two-stage detectors (like Faster R-CNN) that first propose regions and then classify them.

```
Two-stage detector (slow):        YOLO (fast):
  Image → Region proposals          Image → Single CNN pass
        → Classify each region              → Direct detection output
        → NMS filtering                     → NMS filtering

  ~200ms per frame                  ~10–50ms per frame
```

---

## 3. YOLO11m Specifically

**YOLO11m** is the **medium** variant of the 11th generation YOLO model by Ultralytics (released 2024). The model family has size variants:

| Variant | Parameters | Speed | Accuracy |
|---------|-----------|-------|----------|
| YOLO11n (nano) | ~2.6M | Fastest | Lowest |
| YOLO11s (small) | ~9.4M | Fast | Low-medium |
| **YOLO11m (medium)** | **~20M** | **Balanced** | **Medium-high** |
| YOLO11l (large) | ~25M | Slower | High |
| YOLO11x (extra) | ~56M | Slowest | Highest |

YOLO11m was chosen for Drishti Kavach because it offers a good balance — accurate enough to reliably detect people and vehicles at railway distances, fast enough to run alongside BiSeNetV2 without causing unacceptable latency.

---

## 4. YOLO11m Architecture

YOLO11 follows the standard modern YOLO architecture with three main components:

```
Input Image (640 × 640)
        │
        ▼
┌───────────────────┐
│    BACKBONE       │   ← Feature extraction
│  (C3k2 blocks +  │
│   SPPF pooling)   │
└────────┬──────────┘
         │  Multi-scale features
         │  (small, medium, large objects)
         ▼
┌───────────────────┐
│      NECK         │   ← Feature aggregation
│  (C2PSA + FPN +  │
│   PAN pathway)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│      HEAD         │   ← Detection output
│  (Decoupled head) │
│  3 scale outputs  │
└────────┬──────────┘
         │
         ▼
  Boxes + Scores + Classes
```

### 4.1 Backbone

The backbone extracts hierarchical features from the input image:

- **C3k2 blocks** — cross-stage partial network blocks that split feature channels into two paths, process one, and concatenate — reducing computation while maintaining accuracy
- **SPPF (Spatial Pyramid Pooling - Fast)** — applies max pooling at multiple scales and concatenates results, giving the model a large effective receptive field to detect objects of very different sizes (a distant person vs. a nearby truck)

The backbone produces feature maps at three different scales:
- Large feature map → best for detecting **small objects** (distant people)
- Medium feature map → best for **medium objects** (nearby people, animals)
- Small feature map → best for **large objects** (vehicles, trucks)

### 4.2 Neck — Feature Pyramid Network (FPN) + Path Aggregation Network (PAN)

The neck combines features from different backbone stages so that detection can benefit from both high-level semantic information AND fine spatial details:

```
Backbone outputs:
  P3 (large, fine)   ──┐
  P4 (medium)        ──┤── FPN (top-down) → fuses semantic context downward
  P5 (small, coarse) ──┘

After FPN:
  N3 ──┐
  N4 ──┤── PAN (bottom-up) → fuses spatial detail upward
  N5 ──┘

Result: Each scale has both semantic + spatial information
```

**C2PSA (Cross-Stage Partial with Position-Sensitive Attention)** — a new addition in YOLO11 that adds a self-attention mechanism inside the neck, allowing the model to better focus on relevant parts of the feature map when objects are partially occluded or overlapping.

### 4.3 Head — Decoupled Detection Head

Unlike older YOLO versions that used a single head for both box regression and classification, YOLO11 uses **decoupled heads** — one branch for predicting the box coordinates, another for predicting the class probabilities. This improves both speed and accuracy.

For each of the 3 output scales, the head predicts:
- **Box**: `(cx, cy, w, h)` — centre x, centre y, width, height
- **Objectness**: confidence that an object is present
- **Class scores**: probability for each of the 80 COCO classes

---

## 5. COCO Training & What the Model Learned

YOLO11m was trained on the **COCO dataset** (Common Objects in Context):

| Stat | Value |
|------|-------|
| Training images | ~118,000 |
| Validation images | ~5,000 |
| Object categories | 80 |
| Total annotations | ~860,000 bounding boxes |

The model learns to detect all 80 COCO categories. For Drishti Kavach, only a curated subset is used — all others are filtered out at the Python level. The two deployment scripts use different subsets tuned to their context:

**`main.py` — 24 obstacle classes (standard railway deployment):**
```python
OBSTACLE_CLASSES = [
    # People
    "person",
    # Vehicles
    "bicycle", "car", "motorcycle", "bus", "truck",
    # Street furniture
    "bench",
    # Animals
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    # Carried items
    "backpack", "umbrella", "handbag", "suitcase",
    # Indoor furniture / objects
    "chair", "couch", "bed", "laptop",
]
```

**`main_hardware.py` — 24 obstacle classes (hardware prototype deployment):**
```python
OBSTACLE_CLASSES = [
    # People
    "person",
    # Vehicles
    "bicycle", "car", "motorcycle", "bus", "truck",
    # Animals
    "cat", "dog",
    # Carried items
    "backpack", "umbrella", "handbag",
    # Tabletop / small objects
    "bottle", "cup", "knife", "spoon", "bowl",
    "potted plant", "mouse", "remote", "cell phone",
    "book", "scissors", "toothbrush",
]
```

---

## 6. How Inference Works in Drishti Kavach

```python
results = yolo(frame, verbose=False, conf=0.15)[0]
```

Key parameters:
- **`conf=0.15`** — minimum confidence threshold. Any detection below 15% confidence is discarded. Set deliberately low so the system catches uncertain detections (a partially hidden person might score 20%) rather than missing them.
- **`verbose=False`** — suppresses YOLO's console output to keep the terminal clean
- **`[0]`** — takes the results for the first (and only) image in the batch

The results object contains all detections as `results.boxes`, which is iterated:

```python
for box in results.boxes:
    cls_name = yolo.names[int(box.cls)]   # class name string
    if cls_name not in OBSTACLE_CLASSES:
        continue                           # skip irrelevant classes

    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box corners
    conf = float(box.conf)                    # confidence score
```

---

## 7. Confidence Score — What It Means

The confidence score (0.0–1.0) represents how certain YOLO is that:
1. An object is present in that bounding box area
2. That object belongs to the predicted class

| Score | Interpretation |
|-------|----------------|
| < 15% | Discarded (below threshold) |
| 15–40% | Low confidence — displayed, but treat with caution |
| 40–70% | Moderate confidence — likely correct |
| 70–90% | High confidence — almost certainly correct |
| > 90% | Very high confidence — essentially certain |

In the Drishti Kavach display, confidence is shown as a percentage on each bounding box label:
```
ON TRACK: person (87%)
TOO CLOSE: dog (43%)
car (91%)
```

---

## 8. Non-Maximum Suppression (NMS)

YOLO can produce multiple overlapping bounding boxes for the same object (from different grid cells or scales all firing). **NMS** removes duplicates:

1. Sort all boxes by confidence score (highest first)
2. Keep the highest-scoring box
3. Remove any other box that overlaps with it by more than an IoU threshold (typically 0.45)
4. Repeat for remaining boxes

This is handled automatically by Ultralytics YOLO internally — the results already have NMS applied when they are returned.

---

## 9. Bounding Box to Risk — The Overlap Check

Once a bounding box is obtained, the system checks whether the object is on the track using the BiSeNetV2 mask:

```
Bounding box:
  ┌────────────────┐  ← (x1, y1)
  │   TOP HALF     │
  │ ─ ─ ─ ─ ─ ─ ─ │  ← mid_y = y1 + (y2-y1)//2
  │  BOTTOM HALF   │  ← only this region is checked
  │                │    because the feet/wheels are
  └────────────────┘    what actually touches the track
        (x2, y2)

overlap_pct = (track pixels in bottom half) / (total pixels in bottom half) × 100
on_track    = overlap_pct >= threshold[cls_name]
```

**Why only the bottom half?**
An object standing on the track will have its lower half overlapping the track mask. The upper body extends above the track into background. Using the full box would dilute the overlap percentage and miss detections.

---

## 10. Proximity Distance — Foot-Point to Nearest Track Pixel

For objects that are NOT on the track, the distance from the object's "foot point" (bottom-centre of bounding box) to the nearest track pixel is computed:

```python
foot_x = (x1 + x2) // 2   # horizontal centre
foot_y = y2                # bottom of box (where feet/wheels are)

# All track pixel coordinates (class 1 pixels)
track_coords = np.column_stack(np.where(mask_full == 1))

# Euclidean distance from foot to every track pixel, take minimum
diffs = track_coords - np.array([foot_y, foot_x])
dist  = np.hypot(diffs[:, 0], diffs[:, 1]).min()
```

This gives the exact pixel distance from the object's ground contact point to the nearest track edge — a much more meaningful safety metric than bounding box overlap alone.

| Distance | Risk Level | Colour |
|----------|-----------|--------|
| On track (overlap check) | ON_TRACK | 🔴 Red |
| < 80px | TOO_CLOSE | 🟠 Orange |
| 80–213px | NEAR | 🟡 Yellow |
| ≥ 213px | FAR | 🟢 Green |

The thresholds (80px, 213px) are calibrated for a **1280×720** display. At typical railway camera distances, 80px corresponds roughly to an object that is dangerously close to the track edge.

---

## 11. YOLO Performance Characteristics

| Metric | Typical Value (CPU) | Typical Value (MPS/GPU) |
|--------|--------------------|-----------------------|
| Inference time | 80–200ms | 15–50ms |
| mAP@0.5 on COCO | ~51% | same |
| Classes detected | 80 (filtered to 24 per script) | same |
| Input resolution | 640×640 (auto-resized) | same |

YOLO auto-resizes input frames to 640×640 internally. The bounding box coordinates are then automatically mapped back to the original frame dimensions in the results — no manual rescaling needed.

---