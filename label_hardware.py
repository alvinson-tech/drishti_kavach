"""
Drishti Kavach — Hardware Dataset Labeling Tool
================================================
Paints pixel-level masks for hardware_captures images so they can be used
to fine-tune the BiSeNet track segmentation model.

Controls
--------
  Left-click + drag  →  Paint selected class
  Right-click        →  Erase (back to class 2 = background)
  [ / ]              →  Decrease / Increase brush size
  1                  →  Select class 0  "rail-lines"   (RED   overlay)  ← the two parallel rail bars
  2                  →  Select class 1  "track-bed"    (BLUE  overlay)  ← inside area between / around rails
  3                  →  Select class 2  "background"   (black overlay)
  S                  →  Save mask for current image → hardware_dataset/
  N                  →  Next image
  P                  →  Previous image
  Arrow keys         →  Shift the entire mask Up / Down / Left / Right by SHIFT_STEP pixels
  R                  →  Reset mask to all-background
  Q / Esc            →  Quit

Output
------
  hardware_dataset/
    capture_XXXX.jpg   ← copy of original image
    capture_XXXX.png   ← 8-bit grayscale mask  (pixel value = class index)
"""

import cv2
import numpy as np
import os
import shutil
import glob
import sys
import time
import ctypes
import ctypes.util

# ── macOS cursor hide/show via CoreGraphics (no extra packages needed) ────────
if sys.platform == 'darwin':
    try:
        _cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library('CoreGraphics'))
        _cursor_hidden = False

        def _hide_cursor():
            global _cursor_hidden
            if not _cursor_hidden:
                _cg.CGDisplayHideCursor(0)
                _cursor_hidden = True

        def _show_cursor():
            global _cursor_hidden
            if _cursor_hidden:
                _cg.CGDisplayShowCursor(0)
                _cursor_hidden = False
    except Exception:
        def _hide_cursor(): pass
        def _show_cursor(): pass
else:
    def _hide_cursor(): pass
    def _show_cursor(): pass

_last_mouse_t = 0.0   # timestamp of the last mouse-move event inside the window

# ── Config ────────────────────────────────────────────────────────
SOURCE_DIR  = "hardware_captures"
DATASET_DIR = "hardware_dataset"
WIN_W, WIN_H = 1280, 720      # display window size
SHIFT_STEP  = 2              # pixels to shift mask per arrow-key press (in original image coords)

# Arrow-key codes across platforms (cv2.waitKeyEx returns full values)
# macOS: 63232/63233/63234/63235  |  Linux: 65362/65364/65361/65363  |  Windows: 2490368/2621440/2424832/2555904
KEY_UP    = {63232, 65362, 2490368,  82}   # 82  = old Linux masked value
KEY_DOWN  = {63233, 65364, 2621440,  84}
KEY_LEFT  = {63234, 65361, 2424832,  81}
KEY_RIGHT = {63235, 65363, 2555904,  83}

# Class definitions (must match Rs19DatasetConfig)
# class 0 = rail-lines  : the two parallel steel rails
# class 1 = track-bed   : the area between / around the rails
# class 2 = background  : everything else
CLASSES = ["rail-lines", "track-bed", "background"]
COLORS  = [
    (  0,   0, 255),   # RED  → rail-lines  (class 0)  ← paint on the rail bars
    (255,  80,   0),   # BLUE → track-bed   (class 1)  ← paint inside the tracks
    (  0,   0,   0),   # black → background (class 2)
]
BG_CLASS = 2           # default class index for unpainted pixels

os.makedirs(DATASET_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────
def load_image_list():
    paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "capture_*.jpg")))
    return paths


def mask_path_for(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(DATASET_DIR, base + ".png")


def dst_img_path_for(img_path):
    return os.path.join(DATASET_DIR, os.path.basename(img_path))


def load_or_create_mask(img_path, h, w):
    mp = mask_path_for(img_path)
    if os.path.isfile(mp):
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape[:2] == (h, w):
            return mask
    return np.full((h, w), BG_CLASS, dtype=np.uint8)


def save_mask(img_path, mask):
    # ── Save raw label PNG (values 0/1/2 = class indices) ──
    mp = mask_path_for(img_path)
    cv2.imwrite(mp, mask)

    # ── Copy original image if not already there ──
    dp = dst_img_path_for(img_path)
    if not os.path.isfile(dp):
        shutil.copy2(img_path, dp)

    # ── Save a colour-coded preview so you can visually verify ──
    # Uses bright colours (not the near-black class indices)
    PREVIEW_COLORS = [
        (  0,   0, 255),   # class 0  rail-lines  → bright red  (matches overlay)
        (255, 100,   0),   # class 1  track-bed   → bright blue (matches overlay)
        ( 40,  40,  40),   # class 2  background  → dark grey
    ]
    preview_color_map = np.array(PREVIEW_COLORS, dtype=np.uint8)
    preview = preview_color_map[mask]          # (H, W, 3)  fully visible colours
    preview_path = mp.replace(".png", "_preview.jpg")
    cv2.imwrite(preview_path, preview)

    # ── Print per-class pixel counts so you can confirm in the console ──
    total = mask.size
    for cls_idx, cls_name in enumerate(CLASSES):
        px = int(np.sum(mask == cls_idx))
        pct = px / total * 100
        bar = '█' * int(pct / 2)
        print(f"      class {cls_idx} [{cls_name:<12}] : {px:>8} px  ({pct:5.1f}%)  {bar}")
    print(f"  [✓] Mask saved   → {mp}")
    print(f"  [✓] Preview saved → {preview_path}  (open this to visually verify)")


def make_overlay(image_resized, mask):
    """Blend class colours onto the image for visual feedback."""
    color_map = np.array(COLORS, dtype=np.uint8)   # (3, 3)
    color_mask = color_map[mask]                    # (H, W, 3)
    overlay = cv2.addWeighted(image_resized, 0.6, color_mask, 0.4, 0)
    return overlay


def draw_hud(canvas, idx, total, cls_idx, brush_size, saved, img_path):
    h, w = canvas.shape[:2]

    # ── Top-right: filename + saved status ──────────────────────
    filename = os.path.basename(img_path)
    saved_txt = "✓ SAVED" if saved else "● unsaved"
    saved_color = (80, 220, 80) if saved else (80, 180, 255)

    # Dark top bar
    cv2.rectangle(canvas, (0, 0), (w, 38), (20, 20, 20), -1)

    # Filename — centred in top bar
    (fw, _), _ = cv2.getTextSize(filename, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
    cv2.putText(canvas, filename, ((w - fw) // 2, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)

    # Saved status — top-right
    (sw, _), _ = cv2.getTextSize(saved_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, saved_txt, (w - sw - 12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, saved_color, 1, cv2.LINE_AA)

    # Image counter — top-left
    cv2.putText(canvas, f"{idx+1} / {total}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Bottom bar: controls ─────────────────────────────────────
    cv2.rectangle(canvas, (0, h - 38), (w, h), (20, 20, 20), -1)
    info = (f"Class: {cls_idx} [{CLASSES[cls_idx]}]  |  "
            f"Brush: {brush_size}px  |  "
            f"1/2/3=class  [/]=brush  C=copy  V=paste  S=save  N/P=nav  Arrows=shift-mask  R=reset  Q=quit")
    cv2.putText(canvas, info, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Class colour swatch (below top bar, left side) ──────────
    swatch_color = COLORS[cls_idx]
    cv2.rectangle(canvas, (8, 46), (28, 66), swatch_color, -1)
    cv2.rectangle(canvas, (8, 46), (28, 66), (200, 200, 200), 1)
    cv2.putText(canvas, CLASSES[cls_idx], (32, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Clipboard indicator (top bar, left of saved status) ───
    if clipboard["mask"] is not None:
        clip_txt = f"⧉ copied: {clipboard['source']}"
        cv2.putText(canvas, clip_txt, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 220, 255), 1, cv2.LINE_AA)

    return canvas


# ── Mouse callback state ──────────────────────────────────────────
state = {
    "drawing": False,
    "erase"  : False,
    "x": 0, "y": 0,
    "cls"   : 1,     # default: rail-track
    "brush" : 18,
    "mask"  : None,
    "img_w" : 1, "img_h": 1,  # original image dimensions
    "win_w" : WIN_W, "win_h": WIN_H,
    "dirty" : False,
    "saved" : False,
    "mouse_x": 0, "mouse_y": 0,   # current cursor position in window coords
}

# Clipboard — holds a copied mask (None if empty)
clipboard = {"mask": None, "source": None}   # source = filename string


def mouse_cb(event, x, y, flags, param):
    global _last_mouse_t
    s = state

    # Always track mouse position so the cursor circle follows the pointer
    s["mouse_x"] = x
    s["mouse_y"] = y

    # Hide the OS cursor while inside the OpenCV window
    _hide_cursor()
    _last_mouse_t = time.monotonic()

    if event == cv2.EVENT_LBUTTONDOWN:
        s["drawing"] = True
        s["erase"]   = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        s["drawing"] = True
        s["erase"]   = True
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        s["drawing"] = False
        s["erase"]   = False

    if s["drawing"]:
        # Map window coords → mask coords
        mx = int(x * s["img_w"] / s["win_w"])
        my = int(y * s["img_h"] / s["win_h"])
        cls_to_paint = BG_CLASS if s["erase"] else s["cls"]
        cv2.circle(s["mask"], (mx, my), s["brush"], int(cls_to_paint), -1)
        s["dirty"] = True
        s["saved"] = False


def _shift_mask(mask, dx=0, dy=0):
    """Translate *mask* by (dx, dy) pixels.

    Pixels that roll off one edge are NOT wrapped — the vacated strip
    is filled with BG_CLASS so the mask stays consistent.
    """
    h, w = mask.shape[:2]
    shifted = mask.copy()

    if dy != 0:
        shifted = np.roll(shifted, dy, axis=0)
        if dy > 0:
            shifted[:dy, :] = BG_CLASS   # zero top strip
        else:
            shifted[dy:, :] = BG_CLASS   # zero bottom strip

    if dx != 0:
        shifted = np.roll(shifted, dx, axis=1)
        if dx > 0:
            shifted[:, :dx] = BG_CLASS   # zero left strip
        else:
            shifted[:, dx:] = BG_CLASS   # zero right strip

    return shifted


# ── Main ──────────────────────────────────────────────────────────
def main():
    image_paths = load_image_list()
    if not image_paths:
        print(f"[ERROR] No images found in '{SOURCE_DIR}'. "
              "Run capture_hardware.py first.")
        return

    # ── Auto-resume: find first image without a saved mask ───────
    already_done = 0
    start_idx = 0
    for i, p in enumerate(image_paths):
        if os.path.isfile(mask_path_for(p)):
            already_done += 1
        else:
            start_idx = i
            break
    else:
        # All images already labelled — start from beginning
        start_idx = 0

    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Drishti Kavach — Hardware Labeling Tool")
    print(f"  Found {len(image_paths)} images in '{SOURCE_DIR}/'")
    print(f"  Already labelled : {already_done} image(s)")
    print(f"  Resuming from    : {os.path.basename(image_paths[start_idx])}  (#{start_idx+1})")
    print(f"  Masks saved to   : '{DATASET_DIR}/'")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    WINDOW = "Drishti Kavach — Label Tool  (S=save  N/P=nav  Q=quit)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIN_W, WIN_H + 52)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    idx = start_idx

    def load_current():
        img = cv2.imread(image_paths[idx])
        h, w = img.shape[:2]
        mask = load_or_create_mask(image_paths[idx], h, w)
        state["mask"]  = mask
        state["img_w"] = w
        state["img_h"] = h
        state["dirty"] = True
        state["saved"] = os.path.isfile(mask_path_for(image_paths[idx]))
        return img, mask

    img, mask = load_current()

    while True:
        # Always redraw every tick so the brush-cursor circle tracks the mouse
        if state["dirty"]:
            resized     = cv2.resize(img, (WIN_W, WIN_H))
            base_canvas = make_overlay(resized, cv2.resize(
                state["mask"], (WIN_W, WIN_H), interpolation=cv2.INTER_NEAREST))
            base_canvas = draw_hud(base_canvas, idx, len(image_paths),
                                   state["cls"], state["brush"], state["saved"],
                                   image_paths[idx])
            state["_base_canvas"] = base_canvas
            state["dirty"] = False

        # Draw brush-cursor circle on a fresh copy each frame
        canvas = state.get("_base_canvas")
        if canvas is not None:
            display = canvas.copy()
            cx, cy = state["mouse_x"], state["mouse_y"]
            # Scale brush radius from image coords → window coords
            scale  = WIN_W / max(state["img_w"], 1)
            r = max(2, int(state["brush"] * scale))
            # Thin black shadow for contrast on any background
            cv2.circle(display, (cx, cy), r,     (0, 0, 0),       2, cv2.LINE_AA)
            # White outline circle
            cv2.circle(display, (cx, cy), r,     (255, 255, 255), 1, cv2.LINE_AA)
            # Tiny centre dot
            cv2.circle(display, (cx, cy), 2,     (255, 255, 255), -1, cv2.LINE_AA)
            cv2.imshow(WINDOW, display)

        # If no mouse event for >150 ms the pointer has left the window — restore cursor
        if time.monotonic() - _last_mouse_t > 0.15:
            _show_cursor()

        key = cv2.waitKeyEx(16)
        if key == -1:
            continue

        if key in (ord('q'), 27):     # Q / Esc — quit
            break

        elif key == ord('s'):         # S — save
            save_mask(image_paths[idx], state["mask"])
            state["saved"] = True
            state["dirty"] = True

        elif key == ord('r'):         # R — reset mask
            h, w = state["mask"].shape[:2]
            state["mask"] = np.full((h, w), BG_CLASS, dtype=np.uint8)
            state["dirty"] = True

        elif key == ord('n'):         # N — next image
            if idx < len(image_paths) - 1:
                idx += 1
                img, mask = load_current()

        elif key == ord('p'):         # P — previous image
            if idx > 0:
                idx -= 1
                img, mask = load_current()

        # ── Arrow keys: shift the entire mask ────────────────────
        elif key in KEY_UP:
            state["mask"] = _shift_mask(state["mask"], dy=-SHIFT_STEP)
            state["dirty"] = True
            state["saved"] = False
        elif key in KEY_DOWN:
            state["mask"] = _shift_mask(state["mask"], dy=SHIFT_STEP)
            state["dirty"] = True
            state["saved"] = False
        elif key in KEY_LEFT:
            state["mask"] = _shift_mask(state["mask"], dx=-SHIFT_STEP)
            state["dirty"] = True
            state["saved"] = False
        elif key in KEY_RIGHT:
            state["mask"] = _shift_mask(state["mask"], dx=SHIFT_STEP)
            state["dirty"] = True
            state["saved"] = False

        elif key == ord('1'):
            state["cls"] = 0; state["dirty"] = True
        elif key == ord('2'):
            state["cls"] = 1; state["dirty"] = True
        elif key == ord('3'):
            state["cls"] = 2; state["dirty"] = True

        elif key == ord('['):         # Decrease brush
            state["brush"] = max(2, state["brush"] - 4)
            state["dirty"] = True
        elif key == ord(']'):         # Increase brush
            state["brush"] = min(120, state["brush"] + 4)
            state["dirty"] = True

        elif key == ord('c'):         # C — copy current mask to clipboard
            clipboard["mask"]   = state["mask"].copy()
            clipboard["source"] = os.path.basename(image_paths[idx])
            state["dirty"] = True
            print(f"  [⧉] Copied mask from  {clipboard['source']}")

        elif key == ord('v'):         # V — paste clipboard mask onto current image
            if clipboard["mask"] is None:
                print("  [!] Clipboard is empty — press C on an image first.")
            else:
                # Resize clipboard mask to match current image if sizes differ
                ch, cw = clipboard["mask"].shape[:2]
                ih, iw = state["mask"].shape[:2]
                if (ch, cw) == (ih, iw):
                    pasted = clipboard["mask"].copy()
                else:
                    pasted = cv2.resize(
                        clipboard["mask"], (iw, ih),
                        interpolation=cv2.INTER_NEAREST
                    )
                state["mask"]  = pasted
                state["saved"] = False
                state["dirty"] = True
                print(f"  [V] Pasted mask from {clipboard['source']} → {os.path.basename(image_paths[idx])}"
                      f"  (now paint your corrections, then press S)")

        # Force redraw on every mouse drag (dirty set by callback)
        if state["drawing"]:
            state["dirty"] = True

    _show_cursor()   # always restore OS cursor on exit
    cv2.destroyAllWindows()
    total_saved = len(glob.glob(os.path.join(DATASET_DIR, "*.png")))
    print(f"\n[Done] {total_saved} mask(s) saved in '{DATASET_DIR}/'")
    print(f"       Now run:  python train_hardware.py")


if __name__ == "__main__":
    main()
