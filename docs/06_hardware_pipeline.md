# Drishti Kavach — Part 6: Hardware Pipeline

---

## 1. Why a Separate Hardware Pipeline?

The standard `main.py` uses BiSeNetV2 weights pre-trained on **RailSem19** — images taken from real train-mounted cameras on actual railway lines. These images have specific visual characteristics:

- Camera mounted at train height (looking straight down the track)
- Professional wide-angle lenses
- Images from European and Asian railway systems

The hardware **prototype** camera is different:
- Different mounting angle and height
- Different lens (wider/narrower field of view, different distortion)
- Different colour balance and exposure characteristics
- Different lighting conditions in a lab/demo setting

A model trained only on RailSem19 will struggle to segment track reliably from this different viewpoint. The hardware pipeline solves this by **fine-tuning** the model on images captured from the actual prototype camera.

---

## 2. The Four-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                   HARDWARE ADAPTATION PIPELINE                      │
│                                                                     │
│  STEP 1                STEP 2               STEP 3        STEP 4   │
│  ───────               ───────              ───────        ───────  │
│  capture_             label_               train_         main_    │
│  hardware.py          hardware.py          hardware.py    hardware  │
│                                                           .py      │
│  Camera → JPG         JPG → PNG mask       Fine-tune      Run with │
│  saved to             painted in           BiSeNetV2      hardware  │
│  hardware_            OpenCV tool,         on labelled    weights   │
│  captures/            saved to             dataset        (deploy)  │
│                       hardware_dataset/                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step 1: Image Capture (`capture_hardware.py`)

### Purpose
Collect a representative set of images from the prototype camera covering all the angles, lighting conditions, and track configurations the deployed system will encounter.

### How to Run
```bash
python capture_hardware.py
```

### What Happens

1. Opens the camera at `CAMERA_INDEX = 0`
2. Sets resolution to 1280×720
3. Starts a **background console listener thread** that waits for `ENTER` key presses
4. Shows a live **OpenCV preview window** with a HUD:
   - Dark top bar with title (`DRISHTI KAVACH | Hardware Capture Mode`) and timestamp
   - Crosshair at frame centre for alignment reference
   - Dark bottom bar with key hint (`ENTER capture | Q quit`)
   - Capture counter in bottom-right (green)
5. **Capture trigger**: either press `ENTER` in the console or press `ENTER`/`SPACE` in the OpenCV window
6. On capture:
   - Determines the next sequential index by scanning `hardware_captures/` for existing files
   - Saves as `hardware_captures/capture_XXXX.jpg` (4-digit zero-padded)
   - Triggers a **180ms white flash effect** on the preview as visual confirmation
   - Prints the saved path to console

### Best Practices for Capture

For good fine-tuning results, capture images that cover:
- **Different distances** — close track, mid-range track, far track view
- **Different lighting** — bright, dim, shadows, artificial light
- **Different angles** — straight, slight left/right curves
- **Empty track only** — no obstacles during capture (obstacles are added by YOLO, not learned by BiSeNet)
- **At least 30–50 images** — more is better, up to a few hundred

### Output Structure
```
hardware_captures/
  capture_0001.jpg   ← raw camera frame
  capture_0002.jpg
  capture_0003.jpg
  ...
```

---

## 4. Step 2: Mask Labelling (`label_hardware.py`)

### Purpose
Manually draw pixel-level segmentation masks on each captured image, teaching the model what "rail-lines", "track-bed", and "background" look like from this specific camera.

### How to Run
```bash
python label_hardware.py
```

### Auto-Resume
On launch, the tool scans `hardware_dataset/` for existing `.png` masks. It automatically jumps to the **first unlabelled image**, so you never have to remember where you left off.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Drishti Kavach — Hardware Labeling Tool
  Found 45 images in 'hardware_captures/'
  Already labelled : 30 image(s)
  Resuming from    : capture_0031.jpg  (#31)
  Masks saved to   : 'hardware_dataset/'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### The Labelling Workflow

For each image:

1. **Look at the image** with the colour overlay showing what's already painted
2. **Select class 1** (track-bed) by pressing `2` — this is the most important class
3. **Paint the track area** by left-clicking and dragging. The white circle cursor shows the exact brush area
4. **Select class 0** (rail-lines) by pressing `1` — paint the two steel rail bars specifically
5. **Right-click to erase** any mistakes back to background (class 2)
6. **Adjust brush size** with `[` (smaller) and `]` (larger)
7. **Press S to save** — writes 3 files to `hardware_dataset/`
8. **Press N** to move to the next image

### Mask Coordinate System

The mask is stored in **original image resolution** (not window resolution). When you paint on the 1280×720 window, the mouse coordinates are mapped back:

```python
# In mouse_cb:
mx = int(x * state["img_w"] / state["win_w"])  # window X → image X
my = int(y * state["img_h"] / state["win_h"])  # window Y → image Y
cv2.circle(state["mask"], (mx, my), state["brush"], cls_to_paint, -1)
```

This ensures the mask is always at full image resolution regardless of window size.

### The Copy-Paste Workflow

For sequences of similar images (e.g., the camera barely moved between captures), use the clipboard:

1. Label one image carefully → press `S` to save → press `C` to copy mask to clipboard
2. Press `N` to go to the next image
3. Press `V` to paste the clipboard mask — it's instantly applied
4. Make small corrections (arrow keys to shift, brush to repaint)
5. Press `S` to save

This can dramatically speed up labelling when captures are similar.

### Arrow Key Mask Shifting

A common situation: the camera shifted slightly between two captures, so the track is in nearly the same position but offset by a few pixels. Rather than repainting:

- `↑` — shift entire mask up by 2px
- `↓` — shift entire mask down by 2px
- `←` — shift entire mask left by 2px
- `→` — shift entire mask right by 2px

The shift uses `np.roll()` with the vacated edges filled with background (class 2), not wrapped. Multiple presses accumulate — press 5 times for a 10px shift.

### Output Files Per Image

Pressing `S` on `capture_0031.jpg` produces:

```
hardware_dataset/
  capture_0031.jpg          ← copy of original (if not already present)
  capture_0031.png          ← grayscale mask: pixel values 0, 1, or 2
  capture_0031_preview.jpg  ← colour preview for visual verification
```

The **preview image** uses bright, visible colours (not the near-black 0/1/2 values) so you can open it in any image viewer and immediately confirm the mask looks correct:
- 🔴 Bright red → class 0 (rail-lines)
- 🔵 Bright blue/orange → class 1 (track-bed)
- ⬛ Dark grey → class 2 (background)

### Console Pixel Statistics on Save

On every save, per-class pixel counts are printed:
```
      class 0 [rail-lines  ] :    18432 px  ( 2.0%)  █
      class 1 [track-bed   ] :   147456 px  (16.0%)  ████████
      class 2 [background  ] :   754112 px  (82.0%)  █████████████████████████████████████████
  [✓] Mask saved   → hardware_dataset/capture_0031.png
  [✓] Preview saved → hardware_dataset/capture_0031_preview.jpg
```

This lets you quickly verify that your paint session produced reasonable class proportions.

---

## 5. Step 3: Fine-Tuning (`train_hardware.py`)

### Purpose
Start from the RailSem19-pretrained weights and continue training specifically on the hardware dataset, adapting the model to the prototype camera.

### How to Run
```bash
python train_hardware.py
```

### Dataset Loading

The script scans `hardware_dataset/` for matching `.jpg` + `.png` pairs:

```python
img_paths  = sorted(glob.glob("hardware_dataset/*.jpg"))
mask_paths = sorted(glob.glob("hardware_dataset/*.png"))

# Match by basename (excludes _preview.jpg automatically)
common = sorted(set(img_basenames) & set(mask_basenames))
```

The matching by basename ensures:
- Preview JPGs (`_preview.jpg`) are excluded automatically
- Any image without a mask is excluded
- Any mask without an image is excluded

### Train/Validation Split

85% of pairs go to training, 15% to validation:

```python
TRAIN_VAL_RATIO = 0.85
split = int(len(imgs) * 0.85)   # e.g., 38 train, 7 val for 45 total images
```

Pairs are shuffled before splitting so the split is random (seeded for reproducibility with `RANDOM_SEED = 42`).

### Data Augmentation (Training Only)

| Augmentation | Probability / Range | Purpose |
|-------------|---------------------|---------|
| Horizontal flip | 50% | Doubles effective data; simulates left/right curves |
| Brightness jitter | ±30% | Handles lighting variation |
| Contrast jitter | ±30% | Handles different exposure conditions |

Validation images are **never augmented** — they represent clean real-world conditions to give an honest loss measurement.

### Training Loop

```
For each epoch (1 → 60):
  ┌─ Training phase ─────────────────────────────┐
  │  For each batch of 2 images:                  │
  │  1. Forward pass → main + aux outputs         │
  │  2. OHEM loss on main output                  │
  │  3. 0.4× OHEM loss on each aux output         │
  │  4. Total loss.backward()                     │
  │  5. AdamW optimizer step                      │
  └──────────────────────────────────────────────┘
  ┌─ Validation phase ───────────────────────────┐
  │  For each validation image:                   │
  │  1. Forward pass (no_grad, eval mode)         │
  │  2. OHEM loss on main output only             │
  └──────────────────────────────────────────────┘
  
  Print: epoch | train_loss | val_loss | time/ep | ETA
  If val_loss < best_val_loss:
      Save bisenet_hardware.pth  ← best checkpoint
```

### Progress Output

During training, progress is printed every ~25% of each epoch:
```
    epoch 001 | iter 008/32 | loss 1.2341
    epoch 001 | iter 016/32 | loss 0.9823
    epoch 001 | iter 024/32 | loss 0.8512
    epoch 001 | iter 032/32 | loss 0.7234

  ── Epoch 001/060  train_loss=0.9476  val_loss=1.0123  (45s/ep  ETA 00:44:15)
  ★  New best val_loss = 1.0123
  ✓ Checkpoint saved → weights/bisenet_hardware.pth
```

The `★` line only appears when a new best validation loss is achieved.

### Output Weights

| File | When saved | Use |
|------|-----------|-----|
| `weights/bisenet_hardware.pth` | Every time val_loss improves | **Use this for deployment** |
| `weights/bisenet_hardware_last.pth` | Always after final epoch | Fallback / comparison |

### Device Auto-Selection

```python
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")   # Apple Silicon
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")   # Any machine
```

Training time varies significantly by device:
- **CPU only**: ~3–8 minutes per epoch (60 epochs = 3–8 hours)
- **Apple MPS**: ~1–3 minutes per epoch (60 epochs = 1–3 hours)
- **NVIDIA GPU**: ~10–30 seconds per epoch (60 epochs = 10–30 minutes)

---

## 6. Step 4: Hardware Deployment (`main_hardware.py`)

### Purpose
Run the full Drishti Kavach system using the fine-tuned hardware weights instead of the RailSem19 weights.

### How to Run
```bash
python main_hardware.py
```

It first checks that `weights/bisenet_hardware.pth` exists — if not, it exits with a clear error message directing you to run `train_hardware.py` first.

### Key Differences from `main.py`

| Feature | `main.py` | `main_hardware.py` |
|---------|-----------|-------------------|
| BiSeNet weights | `bisenet_railsem19.pth` | `bisenet_hardware.pth` |
| Default camera mode | `"webcam"` | `"image"` (prompts for hardware capture) |
| Image source directory | `test_images/` | `hardware_captures/` |
| State file | `static/session_data.json` | `static/session_data_hw.json` |
| Window title | `"Drishti Kavach"` | `"Drishti Kavach [Hardware Model]"` |

Everything else — threading model, YOLO detection, risk classification, session reporting, snapshot saving — is **identical** to `main.py`.

### Startup Check
```python
if not os.path.isfile(WEIGHTS_PATH):
    print(f"\n[ERROR] Hardware weights not found: {WEIGHTS_PATH}")
    print("        Run train_hardware.py first to generate the model.\n")
    sys.exit(1)
```

This prevents a confusing crash later and tells the user exactly what to do.

---

## 7. End-to-End Example Session

```bash
# Day 1: Capture images from prototype
python capture_hardware.py
# → Take 40 photos of the prototype camera view
# → hardware_captures/capture_0001.jpg ... capture_0040.jpg

# Day 1-2: Label the images
python label_hardware.py
# → Paint masks on all 40 images
# → hardware_dataset/capture_0001.jpg + .png + _preview.jpg ...

# Day 2: Train the model (run overnight if on CPU)
python train_hardware.py
# → 60 epochs, best checkpoint saved
# → weights/bisenet_hardware.pth

# Day 3+: Deploy and run
python main_hardware.py
# → Uses hardware weights
# → Accurate track segmentation for prototype camera
# → Full obstacle detection + alert system active
```

---

## 8. When to Re-Run the Pipeline

| Situation | Action |
|-----------|--------|
| Camera angle changed | Re-capture + re-label affected images, re-train |
| New lighting environment | Capture additional images in that lighting, add to dataset, re-train |
| Model performs poorly on new scenes | Add more diverse captures of those scenes, re-train |
| Want faster convergence | Increase `LR` slightly (e.g., 2e-4) or reduce `NUM_EPOCHS` |
| Overfitting (val_loss rises while train_loss falls) | Add more augmentation or reduce `NUM_EPOCHS` |

---

*← [Part 5 — main.py Feature Deep-Dive](05_main_program.md)*
*→ Continue to [Part 7 — Dashboard & Session Reporting](07_dashboard.md)*
