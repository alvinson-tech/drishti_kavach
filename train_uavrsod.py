"""
Drishti Kavach — BiSeNet Training on UAV-RSOD Dataset
=====================================================
Trains/fine-tunes BiSeNetV2 on the V1 UAV-RSOD_Dataset for Segmentation.

The dataset provides two separate binary masks per image:
  - Rail Inside  (track surface area between rails)
  - Rail Lines   (rail edge lines)

These are merged into a single 3-class mask:
  Class 0 = Rail Lines   (rail edges)
  Class 1 = Rail Inside  (track surface — used for obstacle detection)
  Class 2 = Background   (everything else)

Usage
-----
  python train_uavrsod.py

Output
------
  weights/bisenet_uavrsod.pth       ← best validation-loss checkpoint
  weights/bisenet_uavrsod_last.pth  ← last epoch checkpoint
"""

import sys
import os

# ── Path setup identical to main.py ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'rail_marking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'rail_marking', 'rail_marking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'rail_marking', 'cfg'))

import cv2
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from segmentation.models import BiSeNetV2
from bisenetv2_cfg import BiSeNetV2Config

# ─────────────────────────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────────────────────────
DATASET_ROOT    = "datasets/V1 UAV-RSOD_Dataset for Segmentation"
IMAGES_DIR      = os.path.join(DATASET_ROOT, "1 Images")
MASK_RAIL_INSIDE = os.path.join(DATASET_ROOT, "2 Annotations", "2.2 Masking", "Rail Inside")
MASK_RAIL_LINES  = os.path.join(DATASET_ROOT, "2 Annotations", "2.2 Masking", "Rail Lines")

# ─────────────────────────────────────────────────────────────────
# TRAINING HYPER-PARAMETERS  (edit freely)
# ─────────────────────────────────────────────────────────────────
PRETRAINED_W    = "weights/bisenet_railsem19.pth"
BEST_OUTPUT_W   = "weights/bisenet_uavrsod.pth"
LAST_OUTPUT_W   = "weights/bisenet_uavrsod_last.pth"

NUM_CLASSES     = 3              # Rail Lines, Rail Inside, Background
TRAIN_VAL_RATIO = 0.85           # 85% train, 15% val
IMG_H, IMG_W    = 512, 1024      # BiSeNet expected input size
BATCH_SIZE      = 4              # adjust based on GPU/MPS memory
NUM_EPOCHS      = 100            # training epochs
LR              = 5e-4           # learning rate (fine-tuning)
WEIGHT_DECAY    = 5e-4
NUM_WORKERS     = 0              # 0 is safest on macOS
RANDOM_SEED     = 42
MASK_THRESHOLD  = 128            # binarize JPEG masks above this value

# Augmentation settings
AUG_FLIP_P      = 0.5            # horizontal flip probability
AUG_BRIGHTNESS  = 0.3            # ±brightness fraction
AUG_CONTRAST    = 0.3            # ±contrast fraction
AUG_SCALE_MIN   = 0.75           # min random scale factor
AUG_SCALE_MAX   = 1.25           # max random scale factor
AUG_SCALE_P     = 0.3            # probability of random scaling

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Training on: {DEVICE}")

# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────
class UAVRSODDataset(Dataset):
    """
    Loads image + merged 3-class mask from the UAV-RSOD dataset.

    Merges two binary masks:
      - Rail Inside  → class 1 (track surface)
      - Rail Lines   → class 0 (rail edges)
      - Background   → class 2

    Priority: Rail Lines > Rail Inside > Background
    """

    def __init__(self, image_paths, phase="train"):
        self._images = image_paths
        self._phase  = phase

    def __len__(self):
        return len(self._images)

    def _load_and_merge_masks(self, basename, target_h, target_w):
        """Load both binary masks and merge into a single 3-class mask.
        
        Masks may be stored at different resolutions (some are thumbnails),
        so both are resized to (target_w, target_h) before merging.
        """
        ri_path = os.path.join(MASK_RAIL_INSIDE, basename)
        rl_path = os.path.join(MASK_RAIL_LINES, basename)

        # Load as grayscale
        ri_mask = cv2.imread(ri_path, cv2.IMREAD_GRAYSCALE)
        rl_mask = cv2.imread(rl_path, cv2.IMREAD_GRAYSCALE)

        if ri_mask is None:
            raise FileNotFoundError(f"Rail Inside mask not found: {ri_path}")
        if rl_mask is None:
            raise FileNotFoundError(f"Rail Lines mask not found: {rl_path}")

        # Resize masks to match image dimensions (some masks are thumbnails)
        if ri_mask.shape[:2] != (target_h, target_w):
            ri_mask = cv2.resize(ri_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        if rl_mask.shape[:2] != (target_h, target_w):
            rl_mask = cv2.resize(rl_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Binarize to clean JPEG compression artifacts
        ri_binary = (ri_mask > MASK_THRESHOLD).astype(np.uint8)  # 1 where rail inside
        rl_binary = (rl_mask > MASK_THRESHOLD).astype(np.uint8)  # 1 where rail lines

        # Build 3-class mask: start with background (class 2)
        merged = np.full((target_h, target_w), 2, dtype=np.uint8)

        # Class 1 = Rail Inside (track surface)
        merged[ri_binary == 1] = 1

        # Class 0 = Rail Lines (rail edges) — takes priority over Rail Inside
        merged[rl_binary == 1] = 0

        return merged

    def __getitem__(self, idx):
        img_path = self._images[idx]
        basename = os.path.basename(img_path)

        # Load image
        img = cv2.imread(img_path)  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load and merge masks (pass image dims for resizing mismatched masks)
        mask = self._load_and_merge_masks(basename, img.shape[0], img.shape[1])

        # Resize to BiSeNet input size
        img  = cv2.resize(img,  (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        # ── Augmentation (training only) ──────────────────────────
        if self._phase == "train":
            # Random scaling
            if random.random() < AUG_SCALE_P:
                scale = random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
                new_h = int(IMG_H * scale)
                new_w = int(IMG_W * scale)
                img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Crop or pad back to target size
                if scale > 1.0:
                    # Random crop
                    y_start = random.randint(0, new_h - IMG_H)
                    x_start = random.randint(0, new_w - IMG_W)
                    img  = img[y_start:y_start+IMG_H, x_start:x_start+IMG_W]
                    mask = mask[y_start:y_start+IMG_H, x_start:x_start+IMG_W]
                else:
                    # Pad with background (class 2 for mask, zeros for image)
                    pad_img  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                    pad_mask = np.full((IMG_H, IMG_W), 2, dtype=np.uint8)
                    y_off = (IMG_H - new_h) // 2
                    x_off = (IMG_W - new_w) // 2
                    pad_img[y_off:y_off+new_h, x_off:x_off+new_w]  = img
                    pad_mask[y_off:y_off+new_h, x_off:x_off+new_w] = mask
                    img  = pad_img
                    mask = pad_mask

            # Horizontal flip
            if random.random() < AUG_FLIP_P:
                img  = cv2.flip(img,  1)
                mask = cv2.flip(mask, 1)

            # Random brightness / contrast (image only)
            alpha = 1.0 + random.uniform(-AUG_CONTRAST, AUG_CONTRAST)
            beta  = random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS) * 255
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Normalise [0,1] and convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # Clamp mask to valid range
        mask = np.clip(mask.astype(np.int64), 0, NUM_CLASSES - 1)
        mask_tensor = torch.from_numpy(mask)  # (H, W) int64

        return img_tensor, mask_tensor


def build_loaders():
    """Discover image/mask pairs and create train/val data loaders."""
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

    if not img_paths:
        raise RuntimeError(f"No images found in '{IMAGES_DIR}/'.")

    # Verify that every image has both masks
    valid_paths = []
    for p in img_paths:
        bn = os.path.basename(p)
        ri = os.path.join(MASK_RAIL_INSIDE, bn)
        rl = os.path.join(MASK_RAIL_LINES, bn)
        if os.path.isfile(ri) and os.path.isfile(rl):
            valid_paths.append(p)
        else:
            print(f"  [WARN] Skipping {bn} — missing mask(s)")

    if not valid_paths:
        raise RuntimeError("No valid image+mask triplets found.")

    # Shuffle & split
    random.shuffle(valid_paths)
    split = max(1, int(len(valid_paths) * TRAIN_VAL_RATIO))
    train_paths = valid_paths[:split]
    val_paths   = valid_paths[split:]

    print(f"\n  Dataset: {len(valid_paths)} valid image+mask triplets")
    print(f"  Train  : {len(train_paths)}  |  Val: {len(val_paths)}")

    train_ds = UAVRSODDataset(train_paths, phase="train")
    val_ds   = UAVRSODDataset(val_paths,   phase="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1,
                              shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────
# LOSS — Online Hard Example Mining Cross-Entropy
# ─────────────────────────────────────────────────────────────────
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=255):
        super().__init__()
        self.thresh = thresh
        self.ce     = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)             # (N, H, W)
        loss_flat = loss.view(-1)
        keep      = (loss_flat > self.thresh).sum().item()
        if keep == 0:
            keep = max(1, int(0.1 * loss_flat.numel()))
        loss_sorted, _ = torch.sort(loss_flat, descending=True)
        return loss_sorted[:keep].mean()


# ─────────────────────────────────────────────────────────────────
# METRICS — Per-class IoU
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iou(pred, target, num_classes):
    """Compute per-class IoU between prediction and target masks."""
    ious = []
    for c in range(num_classes):
        pred_c   = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union        = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float('nan'))  # class not present
        else:
            ious.append(intersection / union)
    return ious


# ─────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Rail Lines", "Rail Inside", "Background"]


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        # BiSeNetV2 returns (main_logit, aux1, aux2, …) during training
        if isinstance(outputs, (tuple, list)):
            main_out = outputs[0]
            loss = criterion(main_out, masks)
            for aux in outputs[1:]:
                loss = loss + 0.4 * criterion(aux, masks)
        else:
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
            print(f"    epoch {epoch:03d} | iter {batch_idx+1:03d}/{len(loader)} "
                  f"| loss {loss.item():.4f}")

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_ious   = [[] for _ in range(NUM_CLASSES)]

    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        out   = model(imgs)
        if isinstance(out, (tuple, list)):
            out = out[0]

        loss = criterion(out, masks)
        total_loss += loss.item()

        # Compute per-class IoU
        pred = torch.argmax(out, dim=1)  # (N, H, W)
        for i in range(pred.shape[0]):
            ious = compute_iou(pred[i], masks[i], NUM_CLASSES)
            for c in range(NUM_CLASSES):
                if not np.isnan(ious[c]):
                    all_ious[c].append(ious[c])

    avg_loss = total_loss / max(1, len(loader))

    # Report per-class IoU
    print(f"\n    {'Class':<15} {'IoU':>8}")
    print(f"    {'─'*15} {'─'*8}")
    mean_ious = []
    for c in range(NUM_CLASSES):
        if all_ious[c]:
            ciou = np.mean(all_ious[c])
            mean_ious.append(ciou)
            print(f"    {CLASS_NAMES[c]:<15} {ciou:>7.1%}")
        else:
            print(f"    {CLASS_NAMES[c]:<15}     N/A")
    if mean_ious:
        miou = np.mean(mean_ious)
        print(f"    {'mIoU':<15} {miou:>7.1%}")

    return avg_loss


def save_checkpoint(model, path, meta=None):
    state = {"state_dict": model.state_dict()}
    if meta:
        state.update(meta)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("    DRISHTI KAVACH — BiSeNet Training (UAV-RSOD)")
    print("=" * 60)

    # ── Validate dataset ─────────────────────────────────────────
    if not os.path.isdir(DATASET_ROOT):
        print(f"\n[ERROR] Dataset not found at '{DATASET_ROOT}/'.")
        print("        Please ensure the V1 UAV-RSOD dataset is in the datasets/ folder.\n")
        sys.exit(1)

    train_loader, val_loader = build_loaders()

    # ── Load model ───────────────────────────────────────────────
    print(f"\n[1/3] Loading BiSeNetV2 architecture …")
    model = BiSeNetV2(n_classes=NUM_CLASSES).to(DEVICE)

    if os.path.isfile(PRETRAINED_W):
        ckpt = torch.load(PRETRAINED_W, map_location=DEVICE, weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)
        incompatible = model.load_state_dict(sd, strict=False)
        print(f"       Pre-trained weights loaded from: {PRETRAINED_W}")
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"       (Some keys skipped — normal for fine-tuning)")
    else:
        print(f"       [WARN] Pre-trained weights not found at '{PRETRAINED_W}'.")
        print(f"       Training from scratch — this will take much longer.")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params : {total_params:,}")
    print(f"       Trainable    : {train_params:,}")

    # ── Optimizer & scheduler ────────────────────────────────────
    print(f"\n[2/3] Setting up optimizer …")
    criterion = OHEMCrossEntropyLoss(thresh=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    os.makedirs("weights", exist_ok=True)

    # ── Training loop ────────────────────────────────────────────
    print(f"\n[3/3] Training for {NUM_EPOCHS} epoch(s) …")
    print(f"      Batch size : {BATCH_SIZE}  |  LR : {LR}  |  Device : {DEVICE}")
    print(f"      Classes    : {', '.join(CLASS_NAMES)}")
    print(f"      Pre-trained: {PRETRAINED_W if os.path.isfile(PRETRAINED_W) else 'None (scratch)'}\n")
    print("=" * 60)

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss   = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t_ep
        eta     = (time.time() - t_start) / epoch * (NUM_EPOCHS - epoch)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        lr_now  = optimizer.param_groups[0]['lr']

        print(f"\n  ── Epoch {epoch:03d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={lr_now:.2e}  ({elapsed:.0f}s/ep  ETA {eta_str})")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, BEST_OUTPUT_W,
                            meta={"epoch": epoch, "val_loss": val_loss,
                                  "classes": CLASS_NAMES})
            print(f"  ★  New best val_loss = {best_val_loss:.4f}")

    # Always save last epoch
    save_checkpoint(model, LAST_OUTPUT_W,
                    meta={"epoch": NUM_EPOCHS, "val_loss": val_loss,
                          "classes": CLASS_NAMES})

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))
    print("\n" + "=" * 60)
    print("    TRAINING COMPLETE")
    print(f"    Total time      : {total_time}")
    print(f"    Best val loss   : {best_val_loss:.4f}")
    print(f"    Best weights    : {BEST_OUTPUT_W}")
    print(f"    Last weights    : {LAST_OUTPUT_W}")
    print("=" * 60)
    print(f"\n  Next step: Update WEIGHTS_PATH in main.py to '{BEST_OUTPUT_W}'")
    print(f"             Then run: python main.py\n")


if __name__ == "__main__":
    main()
