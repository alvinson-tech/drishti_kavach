"""
Hardware Capture Utility — Drishti Kavach
Press ENTER in the console to capture a photo from the connected camera.
Photos are saved sequentially in the 'hardware_captures/' folder.
Press Ctrl+C to exit.
"""

import cv2
import os
import sys
import threading
import time
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAVE_DIR = "hardware_captures"
CAMERA_INDEX = 0          # Change if your camera is on a different index
PREVIEW_WINDOW = "Drishti Kavach — Hardware Capture  |  Press ENTER to capture  |  Q to quit"

# ──────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)


def get_next_index() -> int:
    """Return the next sequential capture index based on existing files."""
    existing = [
        f for f in os.listdir(SAVE_DIR)
        if f.startswith("capture_") and f.endswith(".jpg")
    ]
    indices = []
    for name in existing:
        try:
            idx = int(name.replace("capture_", "").replace(".jpg", ""))
            indices.append(idx)
        except ValueError:
            pass
    return max(indices, default=0) + 1


def draw_overlay(frame, capture_count: int, flash: bool = False):
    """Draw HUD overlay on the live preview frame."""
    h, w = frame.shape[:2]

    # Flash effect on capture
    if flash:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.putText(frame, "DRISHTI KAVACH  |  Hardware Capture Mode",
                (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 200, 255), 1, cv2.LINE_AA)

    # Timestamp top-right
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, ts, (w - tw - 14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Bottom banner
    cv2.rectangle(frame, (0, h - 44), (w, h), (15, 15, 15), -1)
    hint = "ENTER  capture    Q  quit"
    cv2.putText(frame, hint, (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 120, 120), 1, cv2.LINE_AA)

    # Capture counter (bottom-right)
    counter_text = f"Captured: {capture_count}"
    (cw, _), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
    cv2.putText(frame, counter_text, (w - cw - 14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 120), 1, cv2.LINE_AA)

    # Centre crosshair
    cx, cy = w // 2, h // 2
    size, thickness, color = 22, 1, (0, 200, 255)
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}). "
              "Check connection or change CAMERA_INDEX in the script.")
        sys.exit(1)

    # Try to set a good resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    capture_count = 0
    capture_flag = threading.Event()
    quit_flag = threading.Event()
    flash_until = [0.0]   # mutable so the input thread can reset it

    def console_listener():
        """Background thread: waits for ENTER key presses."""
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  Drishti Kavach — Hardware Capture Utility")
        print(f"  Saving to  : ./{SAVE_DIR}/")
        print("  Press ENTER to capture  |  Ctrl+C to quit")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        while not quit_flag.is_set():
            try:
                input()           # blocks until ENTER
                capture_flag.set()
            except EOFError:
                break

    listener = threading.Thread(target=console_listener, daemon=True)
    listener.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame — retrying…")
                time.sleep(0.05)
                continue

            # ── Capture triggered ──
            if capture_flag.is_set():
                capture_flag.clear()
                idx = get_next_index()
                filename = os.path.join(SAVE_DIR, f"capture_{idx:04d}.jpg")
                cv2.imwrite(filename, frame)
                capture_count += 1
                flash_until[0] = time.time() + 0.18   # 180 ms white flash
                print(f"  [✓] Saved  →  {filename}  (total: {capture_count})")

            # ── Draw overlay ──
            flash_active = time.time() < flash_until[0]
            display = draw_overlay(frame.copy(), capture_count, flash=flash_active)

            cv2.imshow(PREVIEW_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:   # Q or ESC
                break
            elif key == 13:   # ENTER via OpenCV window
                capture_flag.set()

    except KeyboardInterrupt:
        pass
    finally:
        quit_flag.set()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[Done] {capture_count} photo(s) saved to './{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
