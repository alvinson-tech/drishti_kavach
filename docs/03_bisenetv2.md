# Drishti Kavach: BiSeNetV2 Architecture & Working

---

## 1. What is Semantic Segmentation?

Before diving into BiSeNetV2, it helps to understand what it is solving.

**Semantic segmentation** is the task of assigning a class label to **every single pixel** in an image. Unlike object detection (which draws bounding boxes), segmentation gives you an exact pixel-perfect map of where each class is.

```
Input Image (H × W × 3)          Output Mask (H × W)
┌─────────────────────┐          ┌─────────────────────┐
│                     │          │  2 2 2 2 2 2 2 2 2  │
│   [track photo]     │  ──────▶ │  2 2 1 1 1 1 2 2 2  │
│                     │          │  0 2 1 1 1 1 2 0 0  │
│                     │          │  0 2 1 1 1 1 2 0 0  │
└─────────────────────┘          └─────────────────────┘
                                  0=rail-lines
                                  1=track-bed
                                  2=background
```

This pixel-level precision is what allows Drishti Kavach to know *exactly* where the track surface is, enabling accurate obstacle-on-track overlap calculations.

---

## 2. The Challenge: Speed vs. Accuracy

Classic segmentation networks (like FCN, DeepLab) are very accurate but slow — they process images through a single high-resolution pathway, which is computationally expensive. This makes them unsuitable for real-time use.

BiSeNet solves this with a fundamental architectural insight: **you don't need to process spatial detail and semantic context at the same resolution**.

---

## 3. BiSeNetV2 Architecture

BiSeNetV2 (Bilateral Segmentation Network V2) uses two parallel processing paths that run simultaneously:

```
Input Frame
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
┌──────────────┐          ┌────────────────┐
│  Detail Path │          │ Semantic Path  │
│  (Spatial)   │          │ (Context)      │
├──────────────┤          ├────────────────┤
│ Shallow CNN  │          │ Deep CNN       │
│ High-res     │          │ Low-res        │
│ features     │          │ features       │
│              │          │                │
│ Preserves    │          │ Understands    │
│ fine edges,  │          │ "what" is in   │
│ boundaries   │          │ the scene      │
└──────┬───────┘          └───────┬────────┘
       │                          │
       └──────────┬───────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Aggregation   │
         │  Layer (BGA)   │  ← Bilateral Guided Aggregation
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │  Segmentation  │
         │  Head          │
         └────────┬───────┘
                  │
                  ▼
         Output Mask (H × W)
```

### 3.1 Detail Path (Spatial Path)

- Uses a **shallow, wide** convolutional network
- Operates at **high spatial resolution** (keeps more of the original image size)
- Only 3 convolutional layers — very fast
- Captures **fine spatial details**: exact pixel locations of edges, rail boundaries, transitions between classes
- Does NOT understand what it's looking at — just preserves where things are

### 3.2 Semantic Path (Context Path)

- Uses a **deep, narrow** backbone (lightweight network like MobileNet or a custom encoder)
- Operates at **low spatial resolution** (heavily downsampled)
- Has a large **receptive field** — each output pixel "sees" a large area of the input
- Understands **semantic context**: "this is a track, this is sky, this is a person"
- Uses **Average Pooling** at the end to capture the full global context of the scene

### 3.3 Bilateral Guided Aggregation (BGA)

This is the clever bridge between the two paths. Rather than simply concatenating or adding the two feature maps, BGA uses a **guided fusion** mechanism:

- The semantic path guides the spatial path to know *what* to emphasise
- The spatial path guides the semantic path to know *where* to place boundaries precisely
- This produces a combined feature map that is both semantically meaningful AND spatially precise

### 3.4 Segmentation Head

A final convolutional layer converts the aggregated features into a prediction of shape `(N_classes, H, W)`. A `softmax` or `argmax` over the class dimension gives the final per-pixel class assignment.

---

## 4. Auxiliary Heads (Training Only)

During training, BiSeNetV2 adds **auxiliary supervision heads** at intermediate points in the semantic path. These provide gradient signals deeper into the network, helping earlier layers learn meaningful features faster.

```python
# In train_hardware.py — handling auxiliary outputs:
outputs = model(imgs)
if isinstance(outputs, (tuple, list)):
    main_out = outputs[0]           # main segmentation output
    loss = criterion(main_out, masks)
    for aux in outputs[1:]:         # auxiliary outputs
        loss = loss + 0.4 * criterion(aux, masks)  # weighted at 0.4
```

The auxiliary loss weight of **0.4** means auxiliary heads contribute 40% of a full loss signal — enough to guide learning without dominating the main output's gradient.

**At inference time**, only the main output head is used. The auxiliary heads are discarded, making inference faster.

---

## 5. Input and Output Specifications

| Property | Value |
|----------|-------|
| Input size | 512 × 1024 pixels (H × W) |
| Input channels | 3 (BGR, normalised to [0, 1]) |
| Output | Per-pixel class index map (H × W) |
| Number of classes | 3 (rail-lines, track-bed, background) |
| Inference time | ~50–150ms per frame (CPU), ~15–40ms (GPU/MPS) |

The handler (`RailtrackSegmentationHandler`) automatically:
1. Resizes input frames to 512×1024
2. Normalises pixel values
3. Runs the model
4. Resizes output mask back to original frame dimensions
5. Generates a colour-blended overlay at `overlay_alpha = 0.5`

---

## 6. The Segmentation Handler

The model is not called directly — it is wrapped in `RailtrackSegmentationHandler`:

```python
segmentor = RailtrackSegmentationHandler(
    path_to_snapshot = "weights/bisenet_railsem19.pth",
    model_config     = BiSeNetV2Config(),
    overlay_alpha    = 0.5
)

mask, overlay = segmentor.run(frame, only_mask=False)
```

**`mask`** — a 2D numpy array (H × W) where each pixel is 0, 1, or 2  
**`overlay`** — the original frame blended with the colour-coded segmentation mask at 50% opacity

---

## 7. Track Coverage Calculation

After segmentation, the system calculates what percentage of the frame is track:

```python
total_pixels = mask.size
track_pixels = np.sum(mask > 0)        # class 0 + class 1 (both rail classes)
coverage_pct = (track_pixels / total_pixels) * 100

track_detected = coverage_pct >= 1.0  # track exists if > 1% of frame
```

A coverage below 1% is treated as "no track detected" — this handles edge cases like the camera pointing away from the track or being obscured.

---

## 8. Connected Component Track Numbering

Once the track mask is produced, the system uses **connected components analysis** to identify and number separate track regions:

```
Binary Track Mask          Connected Components         Labelled Output
┌─────────────────┐        ┌─────────────────┐         ┌─────────────────┐
│  ░░░░░░░░░░░░   │        │  ░░ label=1 ░░  │         │    [Track 1]    │
│  ░░░░░░░░░░░░   │  ──▶   │  ░░░░░░░░░░░░   │  ──▶   │                 │
│ ░░    ░░░░░░░   │        │  ░░░░░░░░░░░░   │         │    [Track 2]    │
│ ░░    ░░░░░░░   │        │label=2   label=1│         │                 │
└─────────────────┘        └─────────────────┘         └─────────────────┘
```

Steps:
1. Create a binary mask: `track_binary[mask == 0 or mask == 1] = 255`
2. Special case: if class 0 covers > 50% of the frame, it is likely being used as background — use only class 1 for labelling
3. Run `cv2.connectedComponentsWithStats()` with 8-connectivity
4. Skip any component smaller than 0.5% of the frame (noise filter)
5. For each valid component, find its bottom-most row and horizontal centre
6. Draw a black-background label box + `"Track N"` text in cyan at that position

This allows the operator to see which track region is which when multiple tracks are visible in the frame.

---

## 9. Pre-Training vs. Fine-Tuning

### Pre-Training on RailSem19
The weights `bisenet_railsem19.pth` were produced by training BiSeNetV2 from scratch on 8,500 RailSem19 images across 19 classes, then the output head was remapped to 3 classes. This gives the model a strong general understanding of what railway tracks look like.

### Fine-Tuning on Hardware Captures
`train_hardware.py` starts from `bisenet_railsem19.pth` and continues training on the small hardware dataset:

```
Pre-trained weights (RailSem19)
        ↓
Load into BiSeNetV2(n_classes=3)
        ↓
Fine-tune on hardware_dataset/
  • Low LR (1e-4) — gentle updates, preserve learned features
  • 60 epochs
  • OHEM loss
  • AdamW + Cosine LR schedule
        ↓
bisenet_hardware.pth  ← best validation checkpoint
bisenet_hardware_last.pth ← final epoch checkpoint
```

**Why fine-tune rather than train from scratch?**

Training from scratch on a small dataset almost always leads to overfitting. By starting from weights that already understand railway scenes, fine-tuning only needs to learn the specific visual characteristics of the prototype camera (lens distortion, mounting angle, colour balance) — a much simpler task.

---

## 10. Loss Function — OHEM Cross-Entropy

Standard cross-entropy loss treats all pixels equally. But in a railway scene, **most pixels are background** — the track region is a relatively small fraction. This class imbalance means a naive model can achieve low loss simply by predicting "background" everywhere.

**Online Hard Example Mining (OHEM)** fixes this:

```python
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7):
        self.thresh = thresh
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)   # per-pixel loss
        loss_flat = loss.view(-1)

        # Keep only pixels where loss > threshold (hard examples)
        keep = (loss_flat > self.thresh).sum()
        if keep == 0:
            keep = max(1, int(0.1 * loss_flat.numel()))

        loss_sorted, _ = torch.sort(loss_flat, descending=True)
        return loss_sorted[:keep].mean()   # average of hardest pixels only
```

**How it works:**
1. Compute per-pixel cross-entropy loss
2. Sort all pixel losses from highest to lowest
3. Keep only pixels where loss exceeds the threshold (0.7) — these are the "hard" pixels the model gets wrong
4. If no pixels exceed the threshold (model is doing well), fall back to the top 10%
5. Backpropagate only through these hard pixels

This forces the model to focus on difficult regions — typically the edges between track and background, where accurate boundaries matter most.

---

## 11. Optimizer & Learning Rate Schedule

| Setting | Value | Reason |
|---------|-------|--------|
| Optimizer | AdamW | Adaptive learning rates + weight decay decoupled from gradient updates |
| Initial LR | 1e-4 | Low enough to not destroy pre-trained features |
| Weight decay | 5e-4 | Regularisation to prevent overfitting on small dataset |
| LR schedule | Cosine Annealing | Smoothly reduces LR from 1e-4 to 1e-6 over 60 epochs |
| Batch size | 2 | Small to fit in Mac memory; increase on GPU |

The **Cosine Annealing** schedule reduces the learning rate following a cosine curve:

```
LR
1e-4 │╲
     │ ╲
     │  ╲___
     │      ╲___
     │           ╲_______
1e-6 │___________________╲
     └──────────────────────▶ Epoch
     0                      60
```

This prevents the model from overshooting the optimal weights as training progresses.

---