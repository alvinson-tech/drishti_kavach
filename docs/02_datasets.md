# Drishti Kavach: Datasets & Data Pipeline

---

## 1. Overview of Data Used

Drishti Kavach uses two separate datasets for training and fine-tuning:

| Dataset | Purpose | Source |
|---------|---------|--------|
| **RailSem19** | Pre-training BiSeNetV2 for railway track segmentation | Public academic dataset |
| **Hardware Captures** | Fine-tuning BiSeNetV2 for the custom prototype camera | Captured in-house using `capture_hardware.py` |
| **COCO** | Training YOLO11m for obstacle detection | Public academic dataset (used as-is, no re-training) |

---

## 2. RailSem19 — The Primary Segmentation Dataset

### What is RailSem19?

**RailSem19** is a large-scale, publicly available railway scene understanding dataset released in 2019. It is the **only major public dataset** specifically designed for semantic segmentation of railway environments.

- **8,500 images** taken from cameras mounted on the front of trains
- Images sourced from **diverse locations worldwide** (Europe, Asia, varied lighting)
- Pixel-level annotations for **19 semantic classes** including:
  - `rail-raised` — the actual steel rails
  - `rail-track` — the track bed between/around the rails
  - `background` — everything else (sky, vegetation, structures)
  - Plus road markings, fences, signals, persons, vehicles etc.

### How it was used in this project

The BiSeNetV2 model was **pre-trained on RailSem19** by the original research team. This pre-trained checkpoint (`bisenet_railsem19.pth`) is used directly in `main.py` for the standard operation mode.

For Drishti Kavach, only **3 semantic classes** are used (remapped from the 19 RailSem19 classes):

| Class Index | Name | What it represents |
|-------------|------|-------------------|
| 0 | `rail-lines` | The two parallel steel rail bars |
| 1 | `track-bed` | The surface area between/around the rails |
| 2 | `background` | Everything that is not railway track |

This 3-class simplification is intentional — for obstacle-on-track detection, all that matters is knowing where the track surface is (class 1 = `track-bed`), not fine-grained details like signals or fences.

### Where it is stored

```
downloaded_datasets/    ← RailSem19 images and annotations
```

---

## 3. COCO Dataset — For YOLO Obstacle Detection

### What is COCO?

**COCO** (Common Objects in Context) is the gold-standard benchmark dataset for object detection, containing:

- **330,000+ images**
- **1.5 million object instances**
- **80 object categories**

YOLO11m was trained on COCO by the Ultralytics team. The weights file `yolo11m.pt` is used directly — **no re-training was done** on COCO. The model is used purely for inference.

### Obstacle classes used from COCO

Both deployment scripts filter YOLO11m's output to a curated subset of COCO's 80 classes. Each script uses a different set tuned to its context:

**`main.py` — 24 classes (standard railway deployment):**

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

**`main_hardware.py` — 24 classes (hardware prototype deployment):**

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

All other COCO classes not in the respective list are silently ignored even if detected by YOLO11m.

---

## 4. Hardware Capture Pipeline — Building the Custom Dataset

Because the project involves a physical hardware prototype with its own camera, a completely separate dataset pipeline was built to adapt BiSeNetV2 to that specific camera's view. This pipeline has **three steps**:

```
Step 1: capture_hardware.py  →  Raw JPG photos saved to hardware_captures/
Step 2: label_hardware.py    →  PNG segmentation masks saved to hardware_dataset/
Step 3: train_hardware.py    →  Fine-tuned weights saved to weights/
```

### Step 1: Image Capture (`capture_hardware.py`)

This script opens the prototype camera and lets the user capture frames manually.

**How it works:**
- Opens the camera at index 0 (`cv2.VideoCapture(0)`)
- Sets resolution to 1280×720
- Shows a **live preview window** with an OpenCV HUD showing:
  - Top banner with title and timestamp
  - Bottom bar with key hints
  - A crosshair at frame center
  - Live capture counter (bottom-right)
- A **background console listener thread** waits for the user to press `ENTER`
- On `ENTER` press (from console OR OpenCV window), it:
  1. Saves the current frame as `hardware_captures/capture_XXXX.jpg` (zero-padded 4-digit index)
  2. Shows a **180ms white flash** on the preview as visual feedback
  3. Prints the saved filename to the console
- Press `Q` or `Esc` to stop

**Output format:**
```
hardware_captures/
  capture_0001.jpg
  capture_0002.jpg
  capture_0003.jpg
  ...
```

All images are sequential JPGs. The script auto-detects existing files so numbering always continues correctly even if the script is restarted.

---

### Step 2: Mask Labelling (`label_hardware.py`)

This is a full **custom pixel-level painting tool** built in OpenCV. It allows the user to draw segmentation masks directly on the captured hardware images.

#### Interface Layout

```
┌──────────────────────────────────────────────────┐
│  counter    filename (centered)       saved status│  ← Top HUD bar
│  [class swatch] class name                        │  ← Below top bar
├──────────────────────────────────────────────────┤
│                                                  │
│            Image with colour overlay             │
│                                                  │
│   ○  ← custom white circle cursor (brush size)   │
│                                                  │
├──────────────────────────────────────────────────┤
│  Class | Brush | Controls hint                   │  ← Bottom HUD bar
└──────────────────────────────────────────────────┘
```

#### Painting Classes

| Key | Class | Colour | Meaning |
|-----|-------|--------|---------|
| `1` | rail-lines (class 0) | Red overlay | The two steel rail bars |
| `2` | track-bed (class 1) | Blue/orange overlay | The area between/around rails |
| `3` | background (class 2) | Black overlay | Everything else |

#### Full Controls

| Control | Action |
|---------|--------|
| Left-click + drag | Paint selected class onto the mask |
| Right-click | Erase (reset to background class) |
| `[` / `]` | Decrease / Increase brush size (2–120px) |
| `1` / `2` / `3` | Switch active paint class |
| `S` | Save mask for current image |
| `N` | Next image |
| `P` | Previous image |
| Arrow keys (↑↓←→) | Shift the entire mask by `SHIFT_STEP` pixels (default: 2px) |
| `C` | Copy current mask to clipboard |
| `V` | Paste clipboard mask onto current image (useful for similar frames) |
| `R` | Reset mask (fill everything with background) |
| `Q` / `Esc` | Quit |

#### How the Cursor Works

The OS cursor is hidden inside the OpenCV window using `CoreGraphics.CGDisplayHideCursor()` (macOS). A **custom white circle** is drawn on the canvas at the mouse position each frame, with radius proportional to the current brush size scaled to window pixels. This gives precise visual feedback of exactly what area will be painted.

Cursor hide/show logic:
- Hidden: when the mouse moves inside the OpenCV window (detected via `mouse_cb` events)
- Restored: when the mouse hasn't moved for > 150ms (mouse left the window)
- Always restored: on quit

#### How Masks are Saved

When you press `S`, three files are written to `hardware_dataset/`:

1. **`capture_XXXX.jpg`** — a copy of the original image (if not already there)
2. **`capture_XXXX.png`** — the grayscale mask where each pixel value is its class index (0, 1, or 2)
3. **`capture_XXXX_preview.jpg`** — a colour-coded preview with bright colours so you can visually verify the mask looks correct

The mask PNG stores raw class indices (values 0, 1, 2) — not RGB colours. This is the format expected by the training script.

#### Auto-Resume

When you re-open the tool, it automatically skips to the **first image without a saved mask**, so you never have to track which images you've labelled.

#### Arrow Key Mask Shifting

The shift-mask feature is useful when the camera position between captures shifts slightly. Instead of repainting everything, you can nudge the entire mask:
- Uses `np.roll()` to shift the mask array
- The vacated edge is filled with `BG_CLASS` (not wrapped) — pixels don't bleed from one side to the other
- Each press shifts by `SHIFT_STEP = 2` pixels in image coordinates

---

### Step 3: Fine-Tuning (`train_hardware.py`)

This script fine-tunes the pre-trained BiSeNetV2 model on the hardware-captured and labelled dataset. Covered in detail in Part 3 (BiSeNetV2) and Part 6 (Hardware Pipeline).

---

## 5. Dataset Directory Structure After Labelling

```
hardware_dataset/
  capture_0001.jpg          ← copy of original image
  capture_0001.png          ← grayscale mask (values 0/1/2 = class index)
  capture_0001_preview.jpg  ← colour preview for visual verification
  capture_0002.jpg
  capture_0002.png
  capture_0002_preview.jpg
  ...
```

The `train_hardware.py` script looks for matching `.jpg` + `.png` pairs in this directory. Any image without a matching mask is excluded from training automatically.

---

## 6. Data Augmentation During Training

Since the hardware dataset is small (typically tens to low hundreds of images), data augmentation is applied during training to artificially expand variety:

| Augmentation | Setting | Effect |
|-------------|---------|--------|
| Horizontal flip | 50% probability | Doubles effective dataset size; simulates left/right track views |
| Random brightness | ±30% | Makes model robust to lighting changes |
| Random contrast | ±30% | Helps generalise across different times of day |

Augmentation is only applied during the **training** split — the validation split always uses clean, unmodified images to give an honest performance measure.

---

*← [Part 1 — Project Overview](01_project_overview.md)*
*→ Continue to [Part 3 — BiSeNetV2 Architecture & Working](03_bisenetv2.md)*
