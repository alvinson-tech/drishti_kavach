import sys
import os

# ── Path setup for BiSeNet ─────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'rail_marking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'rail_marking', 'rail_marking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'rail_marking', 'cfg'))

import cv2
import torch
import numpy as np
import threading
import time
import json
import base64
from ultralytics import YOLO
from segmentation.deploy.railtrack_segmentation_handler import RailtrackSegmentationHandler
from bisenetv2_cfg import BiSeNetV2Config

# ─── CONFIG ───────────────────────────────────────────────
# Camera mode: "webcam" or "image"
CAMERA_MODE     = "webcam"

# Webcam settings (Kreo Owl Lite FHD 1080p)
WEBCAM_INDEX    = 0

TEST_IMAGE_PATH = "test_images/1.jpg"  # fallback / default

# Model settings
WEIGHTS_PATH    = "weights/bisenet_railsem19.pth"
PROCESS_EVERY_N = 3
DISPLAY_SIZE    = (1280, 720)
STATE_FILE      = "static/session_data.json"

# Output recording
SAVE_OUTPUT     = False

# Obstacle classes (from COCO dataset)
OBSTACLE_CLASSES = [
    "person", "cat", "dog", "cow", "horse",
    "bicycle", "motorcycle", "car", "truck", "bus"
]

# Per-class overlap thresholds (%)
# How much of bottom half of bounding box must overlap track to trigger alert
OVERLAP_THRESHOLDS = {
    "person"    : 5,
    "bicycle"   : 3,
    "motorcycle": 3,
    "car"       : 10,
    "truck"     : 10,
    "bus"       : 10,
    "cat"       : 8,
    "dog"       : 8,
    "cow"       : 8,
    "horse"     : 8,
}

# ─── PROXIMITY RISK THRESHOLDS (pixels at DISPLAY_SIZE resolution) ─────────
# Distance from obstacle foot-point to nearest track pixel
PROXIMITY_WARN_PX = 80   # closer than this  → TOO CLOSE  (semi-alert, orange box)
PROXIMITY_SAFE_PX = 213  # closer than this  → NEAR       (yellow box, track clear)
                          # farther or equal  → FAR        (green box,  track clear)

# ─── SESSION STATS ────────────────────────────────────────
session = {
    "start_time"             : None,
    "end_time"               : None,
    "total_frames_captured"  : 0,
    "total_frames_processed" : 0,
    "total_alert_frames"     : 0,
    "total_detections"       : 0,
    "detection_confidences"  : [],
    "detection_classes"      : {},
    "track_coverages"        : [],
    "bisenet_times"          : [],
    "yolo_times"             : [],
    "fps_samples"            : [],
    "detection_log"          : [],
}

os.makedirs("static", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
os.makedirs("captures", exist_ok=True)

# ─── VIDEO WRITER ─────────────────────────────────────────
video_writer = None
OUTPUT_PATH   = ""
if SAVE_OUTPUT:
    OUTPUT_PATH = f"captures/capture_{time.strftime('%Y-%m-%d_%H%M')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        OUTPUT_PATH, fourcc, 10.0,
        (DISPLAY_SIZE[0], DISPLAY_SIZE[1])
    )
    print(f"Recording output to: {OUTPUT_PATH}")

# ─── IMAGE SELECTION (image mode only) ───────────────────────────────
if CAMERA_MODE == "image":
    print("=" * 60)
    print("         DRISHTI KAVACH — Initializing System")
    print("=" * 60)
    while True:
        try:
            img_num = input("\n  Enter test image number (e.g. 1 for 1.jpg): ").strip()
            candidate = f"test_images/{img_num}.jpg"
            if not os.path.isfile(candidate):
                print(f"  ✗ File not found: {candidate}  —  try again.")
                continue
            TEST_IMAGE_PATH = candidate
            print(f"  ✓ Image set to: {TEST_IMAGE_PATH}")
            break
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)
else:
    print("=" * 60)
    print("         DRISHTI KAVACH — Initializing System")
    print("=" * 60)

# ─── LOAD MODELS ──────────────────────────────────────────
print("\n[1/2] Loading YOLO11m...")
yolo = YOLO("yolo11m.pt")
print("      YOLO11m loaded! ✓")

print("\n[2/2] Loading BiSeNet...")
bisenet_config = BiSeNetV2Config()
segmentor = RailtrackSegmentationHandler(
    path_to_snapshot=WEIGHTS_PATH,
    model_config=bisenet_config,
    overlay_alpha=0.5
)
print("      BiSeNet loaded! ✓")
print("\n" + "=" * 60)
print(f"      Camera        : {'Kreo Owl Lite FHD 1080p' if CAMERA_MODE == 'webcam' else 'Static Image'}")
print(f"      Camera Mode   : {CAMERA_MODE.upper()}")
print(f"      Display Size  : {DISPLAY_SIZE}")
print(f"      Process Rate  : Every {PROCESS_EVERY_N} frames")
print(f"      Save Output   : {SAVE_OUTPUT}")
print("=" * 60 + "\n")

# ─── SHARED STATE ─────────────────────────────────────────
latest_raw        = None
latest_output     = None
latest_alert      = False
latest_detections = []
raw_lock          = threading.Lock()
out_lock          = threading.Lock()
running           = True

# ─── WRITE STATE TO JSON (for Streamlit dashboard) ────────
def write_state(frame=None):
    avg_fps      = float(np.mean(session["fps_samples"][-30:])) if session["fps_samples"] else 0.0
    avg_conf     = float(np.mean(session["detection_confidences"]) * 100) if session["detection_confidences"] else 0.0
    avg_coverage = float(np.mean(session["track_coverages"])) if session["track_coverages"] else 0.0
    avg_bisenet  = float(np.mean(session["bisenet_times"])) if session["bisenet_times"] else 0.0
    avg_yolo     = float(np.mean(session["yolo_times"])) if session["yolo_times"] else 0.0
    alert_rate   = float(session["total_alert_frames"] / session["total_frames_processed"] * 100) if session["total_frames_processed"] > 0 else 0.0
    duration     = time.time() - session["start_time"] if session["start_time"] else 0

    frame_b64 = ""
    if frame is not None:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf).decode('utf-8')

    state = {
        "running"                : running,
        "timestamp"              : time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time"             : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session["start_time"])) if session["start_time"] else "",
        "duration_seconds"       : round(duration, 1),
        "camera_mode"            : "Kreo Owl Lite FHD 1080p" if CAMERA_MODE == "webcam" else "Static Image",
        "alert"                  : latest_alert,
        "avg_fps"                : round(avg_fps, 1),
        "total_frames_captured"  : session["total_frames_captured"],
        "total_frames_processed" : session["total_frames_processed"],
        "total_detections"       : session["total_detections"],
        "total_alert_frames"     : session["total_alert_frames"],
        "alert_rate"             : round(alert_rate, 1),
        "avg_confidence"         : round(avg_conf, 1),
        "avg_track_coverage"     : round(avg_coverage, 2),
        "avg_bisenet_ms"         : round(avg_bisenet, 1),
        "avg_yolo_ms"            : round(avg_yolo, 1),
        "detection_classes"      : session["detection_classes"],
        "detection_log"          : session["detection_log"][-50:],
        "frame_b64"              : frame_b64,
        "device"                 : str(segmentor._device),
    }

    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

# ─── SEGMENT TRACK ────────────────────────────────────────
def segment_track(frame):
    t_start = time.time()
    mask, overlay = segmentor.run(frame, only_mask=False)
    bisenet_time = (time.time() - t_start) * 1000

    total_pixels = mask.size
    track_pixels = np.sum(mask > 0)
    coverage_pct = (track_pixels / total_pixels) * 100

    session["bisenet_times"].append(bisenet_time)
    session["track_coverages"].append(coverage_pct)

    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Track detected only if it covers at least 1% of the frame
    track_detected = coverage_pct >= 1.0

    # ── Number each separate track region ──
    if track_detected:
        # Binary mask: class 0 = rail-raised, class 1 = rail-track, class 2 = unidentified
        # Use classes 0 and 1 (both rail classes) for track numbering
        track_binary = np.zeros_like(mask_resized, dtype=np.uint8)
        track_binary[(mask_resized == 0) | (mask_resized == 1)] = 255
        # Exclude tiny noise by only keeping regions in the non-"unidentified" area
        # Since class 2 is background, anything that is NOT class 2 is a track region
        # However, if class 0 dominates as background, use only class 1:
        # Let's check if class 0 covers most of the frame (meaning it's actually background)
        class0_ratio = np.sum(mask_resized == 0) / mask_resized.size
        if class0_ratio > 0.5:
            # Class 0 is likely background, only use class 1 for track numbering
            track_binary = np.zeros_like(mask_resized, dtype=np.uint8)
            track_binary[mask_resized == 1] = 255

        # Find connected components to identify separate track regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            track_binary, connectivity=8
        )

        # Label each track (skip label 0 = background)
        track_num = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Only label tracks larger than 0.5% of frame to avoid noise
            if area < (frame.shape[0] * frame.shape[1] * 0.005):
                continue
            track_num += 1

            # Find all rows of this track region
            track_region = (labels == i)
            rows = np.where(track_region.any(axis=1))[0]
            if len(rows) == 0:
                continue
            bottom_row = rows[-1]

            # Anchor near the bottom of the track region,
            # clamped well above the bottom footer row (timestamp / camera label)
            fps_row = frame.shape[0] - 50  # clear both footer text lines
            anchor_row = min(int(bottom_row), fps_row)

            # Horizontal center at the anchor row
            cols_anchor = np.where(track_region[min(anchor_row, frame.shape[0] - 1)])[0]
            cx = int(np.mean(cols_anchor)) if len(cols_anchor) > 0 else int(centroids[i][0])

            label_text = f"Track {track_num}"

            # Draw label — LINE_AA gives smooth anti-aliased strokes (approx 1.5 weight)
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            # Clamp x within frame
            lx = max(4, min(cx - tw // 2, frame.shape[1] - tw - 8))
            # Place text above the anchor row; clamp within frame
            ly = max(th + 6, anchor_row - 6)
            ly = min(ly, frame.shape[0] - 4)
            cv2.rectangle(
                overlay,
                (lx - 4, ly - th - 4),
                (lx + tw + 4, ly + baseline + 2),
                (0, 0, 0), -1
            )
            cv2.putText(
                overlay, label_text,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                lineType=cv2.LINE_AA
            )

    return overlay, mask_resized, track_detected

# ─── DETECT OBSTACLES ─────────────────────────────────────
def detect_obstacles(frame, mask, track_detected):
    global latest_alert, latest_detections

    t_start = time.time()
    results = yolo(frame, verbose=False, conf=0.15)[0]
    yolo_time = (time.time() - t_start) * 1000
    session["yolo_times"].append(yolo_time)

    alert      = False   # full KAVACH alert (on track)
    semi_alert = False   # semi alert (too close)
    detections = []

    # ── Resize mask to frame dimensions ──────────────────────
    mask_full = cv2.resize(
        mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # ── Pre-compute track pixel coordinates for distance queries ──
    # Track = class 1 pixels (rail-track surface)
    track_coords = np.column_stack(np.where(mask_full == 1))  # (row, col)

    def nearest_track_dist(foot_x, foot_y):
        """Euclidean pixel distance from (foot_x, foot_y) to nearest track pixel."""
        if track_coords.size == 0:
            return float('inf')
        # track_coords columns are (row, col) → compare as (y, x)
        diffs = track_coords - np.array([foot_y, foot_x])
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        return float(dists.min())

    for box in results.boxes:
        cls_name = yolo.names[int(box.cls)]
        if cls_name not in OBSTACLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)

        # Clamp to frame bounds
        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))

        # ── on-track check (existing overlap logic) ──────────
        mid_y        = y1 + (y2 - y1) // 2
        bottom_half  = mask_full[
            max(0, mid_y):min(frame.shape[0], y2),
            max(0, x1):min(frame.shape[1], x2)
        ]
        total_pixels = bottom_half.size
        track_pixels = np.sum(bottom_half == 1)
        overlap_pct  = (track_pixels / total_pixels * 100) if total_pixels > 0 else 0
        threshold    = OVERLAP_THRESHOLDS.get(cls_name, 5)
        on_track     = overlap_pct >= threshold

        # ── foot-point (bottom-center of box) ────────────────
        foot_x = max(0, min((x1 + x2) // 2, frame.shape[1] - 1))
        foot_y = max(0, min(y2,              frame.shape[0] - 1))

        # ── classify risk ─────────────────────────────────────
        if track_detected and on_track:
            risk = "ON_TRACK"
        elif track_detected and track_coords.size > 0:
            dist = nearest_track_dist(foot_x, foot_y)
            if dist < PROXIMITY_WARN_PX:
                risk = "TOO_CLOSE"
            elif dist < PROXIMITY_SAFE_PX:
                risk = "NEAR"
            else:
                risk = "FAR"
        else:
            risk = "NEAR"   # no track info → treat as near (yellow)

        # ── draw bounding box + label by risk ─────────────────
        if risk == "ON_TRACK":
            color = (0, 0, 255)            # Red
            label = f"ON TRACK: {cls_name} ({conf:.0%})"
        elif risk == "TOO_CLOSE":
            color = (0, 100, 255)          # Dark orange (BGR)
            label = f"TOO CLOSE: {cls_name} ({conf:.0%})"
        elif risk == "NEAR":
            color = (0, 255, 255)          # Yellow
            label = f"{cls_name} ({conf:.0%})"
        else:  # FAR
            color = (0, 200, 0)            # Green
            label = f"{cls_name} ({conf:.0%})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                    lineType=cv2.LINE_AA)

        # ── update alert flags & session stats ────────────────
        if risk == "ON_TRACK":
            alert = True
            detections.append({"class": cls_name, "confidence": conf, "risk": "ON_TRACK"})
            session["detection_classes"][cls_name] = session["detection_classes"].get(cls_name, 0) + 1
            session["detection_confidences"].append(conf)
            session["total_detections"] += 1
            session["detection_log"].append({
                "time"      : time.strftime("%H:%M:%S"),
                "class"     : cls_name,
                "confidence": round(conf * 100, 1),
                "status"    : "ON TRACK"
            })
            print(f"  [KAVACH ALERT] ON TRACK: {cls_name} ({conf:.0%})")

        elif risk == "TOO_CLOSE":
            semi_alert = True
            detections.append({"class": cls_name, "confidence": conf, "risk": "TOO_CLOSE"})
            session["detection_classes"][cls_name] = session["detection_classes"].get(cls_name, 0) + 1
            session["detection_confidences"].append(conf)
            session["total_detections"] += 1
            session["detection_log"].append({
                "time"      : time.strftime("%H:%M:%S"),
                "class"     : cls_name,
                "confidence": round(conf * 100, 1),
                "status"    : "TOO CLOSE"
            })
            print(f"  [SEMI-ALERT] TOO CLOSE: {cls_name} ({conf:.0%})")

        else:
            detections.append({"class": cls_name, "confidence": conf, "risk": risk})

    # ── Status Banner ─────────────────────────────────────────
    def _banner_text(img, text, font_scale, color, thickness=2):
        """Draw text centered horizontally in the status banner (y=40)."""
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = max(8, (DISPLAY_SIZE[0] - tw) // 2)
        cv2.putText(img, text, (tx, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if not track_detected:
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (20, 80, 120), -1)
        _banner_text(frame, "NO TRACK DETECTED  —  MONITORING SURROUNDINGS", 0.70, (255, 255, 255))
    elif alert:
        session["total_alert_frames"] += 1
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (0, 0, 180), -1)
        _banner_text(frame, "!! KAVACH ALERT: OBSTACLE ON TRACK !!", 0.85, (255, 255, 255))
    elif semi_alert:
        session["total_alert_frames"] += 1
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (40, 40, 180), -1)  # light red bg
        _banner_text(frame, "SEMI-KAVACH ALERT: OBSTACLES TOO CLOSE TO TRACK!", 0.72, (255, 255, 255))
    else:
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (0, 120, 0), -1)
        _banner_text(frame, "TRACK CLEAR", 0.90, (255, 255, 255))

    # Timestamp — bottom-right corner
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (ts_w, ts_h), _ = cv2.getTextSize(
        timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    ts_x = DISPLAY_SIZE[0] - ts_w - 15
    ts_y = DISPLAY_SIZE[1] - 12
    cv2.putText(frame, timestamp,
                (ts_x, ts_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Camera label — bottom-left (no FPS)
    cam_label = "Kreo Owl Lite FHD" if CAMERA_MODE == "webcam" else "Static Image"
    cv2.putText(frame, cam_label,
                (20, DISPLAY_SIZE[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    latest_alert = alert or semi_alert
    latest_detections = detections
    return frame

# ─── SESSION REPORT ───────────────────────────────────────
def print_report():
    s            = session
    duration     = s["end_time"] - s["start_time"]
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))

    avg_bisenet  = np.mean(s["bisenet_times"])        if s["bisenet_times"]        else 0
    avg_yolo     = np.mean(s["yolo_times"])           if s["yolo_times"]           else 0
    avg_fps      = np.mean(s["fps_samples"])          if s["fps_samples"]          else 0
    avg_coverage = np.mean(s["track_coverages"])      if s["track_coverages"]      else 0
    std_coverage = np.std(s["track_coverages"])       if s["track_coverages"]      else 0
    avg_conf     = np.mean(s["detection_confidences"]) * 100 if s["detection_confidences"] else 0
    alert_rate   = (s["total_alert_frames"] / s["total_frames_processed"] * 100) if s["total_frames_processed"] > 0 else 0

    print("\n")
    print("=" * 60)
    print("         DRISHTI KAVACH — SESSION REPORT")
    print("=" * 60)

    print("\n📅  SESSION INFO")
    print(f"    Start Time          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s['start_time']))}")
    print(f"    End Time            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s['end_time']))}")
    print(f"    Total Duration      : {duration_str}")
    print(f"    Camera              : {'Kreo Owl Lite FHD 1080p' if CAMERA_MODE == 'webcam' else 'Static Image'}")
    print(f"    Camera Mode         : {CAMERA_MODE.upper()}")
    print(f"    Display Resolution  : {DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]}")

    print("\n🤖  MODELS USED")
    print(f"    Segmentation Model  : BiSeNetV2")
    print(f"    Segmentation Weights: bisenet_railsem19.pth (trained on RailSem19)")
    print(f"    Detection Model     : YOLO11m (trained on COCO dataset)")
    print(f"    Obstacle Classes    : {', '.join(OBSTACLE_CLASSES)}")
    print(f"    Running On          : {segmentor._device}")

    print("\n📊  FRAME STATISTICS")
    print(f"    Total Frames Captured   : {s['total_frames_captured']}")
    print(f"    Total Frames Processed  : {s['total_frames_processed']}")
    print(f"    Processing Rate         : Every {PROCESS_EVERY_N} frames")
    print(f"    Average FPS             : {avg_fps:.1f} fps")

    print("\n⚡  PERFORMANCE METRICS")
    print(f"    Avg BiSeNet Inference   : {avg_bisenet:.1f} ms/frame")
    print(f"    Avg YOLO11m Inference   : {avg_yolo:.1f} ms/frame")
    print(f"    Avg Total Latency       : {avg_bisenet + avg_yolo:.1f} ms/frame")

    print("\n🛤️   BISENET — TRACK SEGMENTATION")
    print(f"    Avg Track Coverage      : {avg_coverage:.2f}% of frame")
    print(f"    Segmentation Stability  : ±{std_coverage:.2f}% (std deviation)")
    print(f"    Frames Segmented        : {s['total_frames_processed']}")

    print("\n🎯  YOLO11m — OBSTACLE DETECTION")
    print(f"    Total Detections        : {s['total_detections']}")
    print(f"    Total Alert Frames      : {s['total_alert_frames']}")
    print(f"    Alert Rate              : {alert_rate:.1f}% of processed frames")
    if s["detection_confidences"]:
        print(f"    Avg Detection Confidence: {avg_conf:.1f}%")
        print(f"    Min Confidence          : {min(s['detection_confidences'])*100:.1f}%")
        print(f"    Max Confidence          : {max(s['detection_confidences'])*100:.1f}%")
    else:
        print(f"    Avg Detection Confidence: N/A (no detections this session)")

    if s["detection_classes"]:
        print(f"\n    Obstacles Detected By Class:")
        for cls, count in sorted(s["detection_classes"].items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"      → {cls:<15} : {count} time(s)")

    print("\n📈  ACCURACY EXPLANATION")
    print("    Confidence Score: YOLO11m's certainty that a detected object")
    print("    is what it says it is (0-100%). A score of 80% means the")
    print("    model is 80% sure the detected object is correct.")
    print("")
    print("    Track Coverage: % of the camera frame identified as railway")
    print("    track by BiSeNet. Stable coverage = consistent segmentation.")
    print("")
    print("    Alert Rate: % of processed frames where an obstacle was")
    print("    found inside the track region.")
    print("")
    print("    Note: Full precision/recall metrics require a labelled ground")
    print("    truth dataset. These metrics reflect live system performance.")

    print("\n" + "=" * 60)
    print("              END OF SESSION REPORT")
    print("=" * 60 + "\n")

# ─── WEBCAM READER (Kreo Owl Lite FHD 1080p) ──────────────
def webcam_reader():
    global latest_raw, running
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # Warmup — let camera adjust to lighting conditions
    print("Warming up Kreo Owl Lite...")
    for _ in range(30):
        cap.read()
    print("Kreo Owl Lite connected! ✓")

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        session["total_frames_captured"] += 1
        with raw_lock:
            latest_raw = frame

    cap.release()

# ─── STATIC IMAGE READER ──────────────────────────────────
def image_reader():
    global latest_raw, running
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print(f"ERROR: Could not load image at {TEST_IMAGE_PATH}")
        running = False
        return

    print(f"Image loaded: {TEST_IMAGE_PATH} ✓")
    while running:
        session["total_frames_captured"] += 1
        with raw_lock:
            latest_raw = frame.copy()
        time.sleep(0.1)

# ─── MODEL RUNNER ─────────────────────────────────────────
def model_runner():
    global latest_raw, latest_output, running
    frame_count    = 0
    last_processed = None
    last_time      = time.time()
    last_write     = time.time()

    while running:
        with raw_lock:
            frame = latest_raw
            latest_raw = None

        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        now = time.time()
        fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
        session["fps_samples"].append(fps)
        last_time = now

        resized = cv2.resize(frame, DISPLAY_SIZE)

        if frame_count % PROCESS_EVERY_N == 0:
            session["total_frames_processed"] += 1
            overlay, mask, track_detected = segment_track(resized)
            output         = detect_obstacles(overlay, mask, track_detected)
            last_processed = output.copy()
            with out_lock:
                latest_output = output

            # Write state to dashboard JSON every 0.5s
            if now - last_write >= 0.5:
                write_state(output)
                last_write = now
        else:
            if last_processed is not None:
                with out_lock:
                    latest_output = last_processed.copy()
            else:
                output = resized.copy()
                cv2.rectangle(output, (0, 0), (DISPLAY_SIZE[0], 60), (0, 120, 0), -1)
                (tw, _), _ = cv2.getTextSize("TRACK CLEAR", cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)
                tx = max(8, (DISPLAY_SIZE[0] - tw) // 2)
                cv2.putText(output, "TRACK CLEAR",
                            (tx, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.90, (255, 255, 255), 2)
                with out_lock:
                    latest_output = output

# ─── START THREADS ────────────────────────────────────────
if CAMERA_MODE == "webcam":
    t_cam = threading.Thread(target=webcam_reader, daemon=True)
elif CAMERA_MODE == "image":
    t_cam = threading.Thread(target=image_reader, daemon=True)
else:
    print(f"Unknown CAMERA_MODE: '{CAMERA_MODE}'. Use 'webcam' or 'image'.")
    sys.exit(1)

t_model = threading.Thread(target=model_runner, daemon=True)

session["start_time"] = time.time()
t_cam.start()
t_model.start()

print("Waiting for first frame...")
while True:
    with out_lock:
        if latest_output is not None:
            break
    time.sleep(0.05)

print("Drishti Kavach is running!")
print("→ OpenCV window is open")
print("→ Run dashboard: streamlit run dashboard.py")
print("→ Press S to save a snapshot")
print("→ Press Q to quit\n")

# ─── DISPLAY LOOP ─────────────────────────────────────────
while True:
    with out_lock:
        if latest_output is not None:
            display = latest_output.copy()
        else:
            continue

    cv2.imshow("Drishti Kavach", display)

    # Save output recording
    if SAVE_OUTPUT and video_writer is not None:
        video_writer.write(display)

    key = cv2.waitKey(1) & 0xFF

    # S key — save snapshot
    if key == ord('s'):
        snapshot_path = f"snapshots/snapshot_{time.strftime('%Y-%m-%d_%H%M')}.jpg"
        cv2.imwrite(snapshot_path, display)
        print(f"  Snapshot saved: {snapshot_path}")

    # Q key — quit
    if key == ord('q'):
        print("\nShutting down Drishti Kavach...")
        running = False
        session["end_time"] = time.time()
        break

cv2.destroyAllWindows()

if SAVE_OUTPUT and video_writer is not None:
    video_writer.release()
    print(f"Recording saved to: {OUTPUT_PATH}")

write_state()
print_report()