"""
Drishti Kavach — BiSeNet Fine-Tuning Script
============================================
Fine-tunes the pre-trained bisenet_railsem19.pth on YOUR hardware prototype
images that you labelled with label_hardware.py.

Usage
-----
  python train_hardware.py

Output
------
  weights/bisenet_hardware.pth   ← best validation-loss checkpoint
  weights/bisenet_hardware_last.pth  ← last epoch checkpoint

Dataset expected layout (produced by label_hardware.py):
  hardware_dataset/
    capture_XXXX.jpg    ← original image
    capture_XXXX.png    ← grayscale mask  (0=rail-raised, 1=rail-track, 2=bg)
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
# TRAINING HYPER-PARAMETERS  (edit freely)
# ─────────────────────────────────────────────────────────────────
DATASET_DIR     = "hardware_dataset"
PRETRAINED_W    = "weights/bisenet_railsem19.pth"
BEST_OUTPUT_W   = "weights/bisenet_hardware.pth"
LAST_OUTPUT_W   = "weights/bisenet_hardware_last.pth"

NUM_CLASSES     = 3
TRAIN_VAL_RATIO = 0.85      # 85% train, 15% val
IMG_H, IMG_W    = 512, 1024  # BiSeNet expected input size
BATCH_SIZE      = 2          # keep small for Mac/CPU; increase for GPU
NUM_EPOCHS      = 60         # fine-tuning epochs (fewer than scratch training)
LR              = 1e-4       # lower LR for fine-tuning
WEIGHT_DECAY    = 5e-4
NUM_WORKERS     = 0          # 0 is safest on macOS
RANDOM_SEED     = 42

# Augmentation probability
AUG_FLIP_P      = 0.5
AUG_BRIGHTNESS  = 0.3        # ±fraction
AUG_CONTRAST    = 0.3

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
class HardwareDataset(Dataset):
    """
    Loads image+mask pairs from hardware_dataset/.
    Images: *.jpg   Masks: *.png (grayscale, pixel = class index)
    """

    def __init__(self, image_paths, mask_paths, phase="train"):
        self._images = image_paths
        self._masks  = mask_paths
        self._phase  = phase

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img  = cv2.imread(self._images[idx])            # BGR uint8
        mask = cv2.imread(self._masks[idx], cv2.IMREAD_GRAYSCALE)

        # Resize to BiSeNet input size
        img  = cv2.resize(img,  (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        # ── Augmentation (training only) ──────────────────────────
        if self._phase == "train":
            # Horizontal flip
            if random.random() < AUG_FLIP_P:
                img  = cv2.flip(img,  1)
                mask = cv2.flip(mask, 1)

            # Random brightness / contrast (image only)
            alpha = 1.0 + random.uniform(-AUG_CONTRAST,   AUG_CONTRAST)
            beta  = random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS) * 255
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Normalise [0,1] and convert to tensor  (C, H, W)
        img_tensor  = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).permute(2, 0, 1)          # (3, H, W)

        # Clamp mask to valid range just in case
        mask = np.clip(mask.astype(np.int64), 0, NUM_CLASSES - 1)
        mask_tensor = torch.from_numpy(mask)   # (H, W)  int64

        return img_tensor, mask_tensor


def build_loaders():
    img_paths  = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))

    if not img_paths:
        raise RuntimeError(f"No images found in '{DATASET_DIR}/'. "
                           "Run label_hardware.py first to create masks.")

    # Match by basename
    img_base  = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
    mask_base = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
    common    = sorted(set(img_base) & set(mask_base))

    if not common:
        raise RuntimeError("No matching image+mask pairs found. "
                           "Make sure label_hardware.py saved masks for the same filenames.")

    imgs  = [img_base[k]  for k in common]
    masks = [mask_base[k] for k in common]

    # Shuffle & split
    paired = list(zip(imgs, masks))
    random.shuffle(paired)
    imgs, masks = zip(*paired)

    split = max(1, int(len(imgs) * TRAIN_VAL_RATIO))
    train_imgs, val_imgs   = imgs[:split],  imgs[split:]
    train_masks, val_masks = masks[:split], masks[split:]

    print(f"\n  Dataset: {len(imgs)} total pairs")
    print(f"  Train  : {len(train_imgs)}  |  Val: {len(val_imgs)}")

    train_ds = HardwareDataset(list(train_imgs), list(train_masks), phase="train")
    val_ds   = HardwareDataset(list(val_imgs),   list(val_masks),   phase="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────
# LOSS  — Online Hard Example Mining Cross-Entropy
# ─────────────────────────────────────────────────────────────────
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=255):
        super().__init__()
        self.thresh  = thresh
        self.ce      = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)                 # (N, H, W)
        loss_flat = loss.view(-1)
        keep      = (loss_flat > self.thresh).sum().item()
        if keep == 0:
            keep = max(1, int(0.1 * loss_flat.numel()))
        loss_sorted, _ = torch.sort(loss_flat, descending=True)
        return loss_sorted[:keep].mean()


# ─────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────────
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
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        out   = model(imgs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = criterion(out, masks)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


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
    print("    DRISHTI KAVACH — BiSeNet Fine-Tuning")
    print("=" * 60)

    # ── Validate dataset ─────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        print(f"\n[ERROR] '{DATASET_DIR}/' not found.")
        print("        Run label_hardware.py first to label your images.\n")
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
        print(f"       Training from scratch — this may take much longer.")

    # ── Optimizer & scheduler ────────────────────────────────────
    print(f"\n[2/3] Setting up optimizer …")
    criterion = OHEMCrossEntropyLoss(thresh=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    os.makedirs("weights", exist_ok=True)

    # ── Training loop ────────────────────────────────────────────
    print(f"\n[3/3] Training for {NUM_EPOCHS} epoch(s) …")
    print(f"      Batch size : {BATCH_SIZE}  |  LR : {LR}  |  Device : {DEVICE}\n")
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

        print(f"\n  ── Epoch {epoch:03d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({elapsed:.0f}s/ep  ETA {eta_str})")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, BEST_OUTPUT_W,
                            meta={"epoch": epoch, "val_loss": val_loss})
            print(f"  ★  New best val_loss = {best_val_loss:.4f}")

    # Always save last epoch
    save_checkpoint(model, LAST_OUTPUT_W,
                    meta={"epoch": NUM_EPOCHS, "val_loss": val_loss})

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))
    print("\n" + "=" * 60)
    print("    TRAINING COMPLETE")
    print(f"    Total time   : {total_time}")
    print(f"    Best val loss: {best_val_loss:.4f}")
    print(f"    Best weights : {BEST_OUTPUT_W}")
    print(f"    Last weights : {LAST_OUTPUT_W}")
    print("=" * 60)
    print("\n  Next step: python main_hardware.py\n")


if __name__ == "__main__":
    main()
