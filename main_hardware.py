import sys
import os

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

# ─── CONFIG ───────────────────────────────────────────────────────
CAMERA_MODE   = "webcam"
WEBCAM_INDEX  = 0
TEST_IMAGE_PATH = "hardware_captures/capture_0024.jpg"

# ← Hardware-trained weights (produced by train_hardware.py)
WEIGHTS_PATH  = "weights/bisenet_hardware.pth"

PROCESS_EVERY_N = 3
DISPLAY_SIZE    = (1280, 720)
STATE_FILE      = "static/session_data_hw.json"

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
OVERLAP_THRESHOLDS = {
    # People
    "person"      : 5,
    # Vehicles
    "bicycle"     : 3,
    "motorcycle"  : 3,
    "car"         : 10,
    "truck"       : 10,
    "bus"         : 10,
    # Animals
    "cat"         : 8,
    "dog"         : 8,
    # Carried items
    "backpack"    : 8,
    "umbrella"    : 5,
    "handbag"     : 8,
    # Tabletop / small objects
    "bottle"      : 5,
    "cup"         : 5,
    "knife"       : 3,
    "spoon"       : 3,
    "bowl"        : 5,
    "potted plant": 8,
    "mouse"       : 3,
    "remote"      : 3,
    "cell phone"  : 3,
    "book"        : 5,
    "scissors"    : 3,
    "toothbrush"  : 3,
}
PROXIMITY_WARN_PX = 80
PROXIMITY_SAFE_PX = 213

session = {
    "start_time": None, "end_time": None,
    "total_frames_captured": 0, "total_frames_processed": 0,
    "total_alert_frames": 0, "total_detections": 0,
    "detection_confidences": [], "detection_classes": {},
    "track_coverages": [], "bisenet_times": [], "yolo_times": [],
    "fps_samples": [], "detection_log": [],
}

os.makedirs("static",          exist_ok=True)
os.makedirs("snapshots",       exist_ok=True)
os.makedirs("session_reports", exist_ok=True)

if CAMERA_MODE == "image":
    print("=" * 60)
    print("  DRISHTI KAVACH [Hardware Model] — Initializing")
    print("=" * 60)
    while True:
        try:
            img_num   = input("\n  Enter test image number (e.g. 1 for 1.jpg): ").strip()
            candidate = f"hardware_captures/{img_num}.jpg"
            if not os.path.isfile(candidate):
                print(f"  ✗ Not found: {candidate}")
                continue
            TEST_IMAGE_PATH = candidate
            print(f"  ✓ Image: {TEST_IMAGE_PATH}")
            break
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
else:
    print("=" * 60)
    print("  DRISHTI KAVACH [Hardware Model] — Initializing")
    print("=" * 60)

# ─── LOAD MODELS ──────────────────────────────────────────────────
if not os.path.isfile(WEIGHTS_PATH):
    print(f"\n[ERROR] Hardware weights not found: {WEIGHTS_PATH}")
    print("        Run train_hardware.py first to generate the model.\n")
    sys.exit(1)

print("\n[1/2] Loading YOLO11m...")
yolo = YOLO("yolo11m.pt")
print("      YOLO11m loaded! ✓")

print("\n[2/2] Loading BiSeNet (hardware-trained)...")
bisenet_config = BiSeNetV2Config()
segmentor = RailtrackSegmentationHandler(
    path_to_snapshot=WEIGHTS_PATH,
    model_config=bisenet_config,
    overlay_alpha=0.5
)
print("      BiSeNet [hardware] loaded! ✓")
print("\n" + "=" * 60)
print(f"      Weights       : {WEIGHTS_PATH}")
print(f"      Camera Mode   : {CAMERA_MODE.upper()}")
print(f"      Display Size  : {DISPLAY_SIZE}")
print("=" * 60 + "\n")

# ─── SHARED STATE ─────────────────────────────────────────────────
latest_raw        = None
latest_output     = None
latest_alert      = False
latest_detections = []
raw_lock          = threading.Lock()
out_lock          = threading.Lock()
running           = True

# ─── WRITE STATE ──────────────────────────────────────────────────
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
        "running": running,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session["start_time"])) if session["start_time"] else "",
        "duration_seconds": round(duration, 1),
        "camera_mode": "Hardware Model Webcam" if CAMERA_MODE == "webcam" else "Static Image",
        "alert": latest_alert,
        "avg_fps": round(avg_fps, 1),
        "total_frames_captured": session["total_frames_captured"],
        "total_frames_processed": session["total_frames_processed"],
        "total_detections": session["total_detections"],
        "total_alert_frames": session["total_alert_frames"],
        "alert_rate": round(alert_rate, 1),
        "avg_confidence": round(avg_conf, 1),
        "avg_track_coverage": round(avg_coverage, 2),
        "avg_bisenet_ms": round(avg_bisenet, 1),
        "avg_yolo_ms": round(avg_yolo, 1),
        "detection_classes": session["detection_classes"],
        "detection_log": session["detection_log"][-50:],
        "frame_b64": frame_b64,
        "device": str(segmentor._device),
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception:
        pass

# ─── SEGMENT TRACK ────────────────────────────────────────────────
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
        mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    track_detected = coverage_pct >= 1.0

    if track_detected:
        track_binary = np.zeros_like(mask_resized, dtype=np.uint8)
        class0_ratio = np.sum(mask_resized == 0) / mask_resized.size
        if class0_ratio > 0.5:
            track_binary[mask_resized == 1] = 255
        else:
            track_binary[(mask_resized == 0) | (mask_resized == 1)] = 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            track_binary, connectivity=8
        )
        track_num = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < (frame.shape[0] * frame.shape[1] * 0.005):
                continue
            track_num += 1
            track_region = (labels == i)
            rows = np.where(track_region.any(axis=1))[0]
            if len(rows) == 0:
                continue
            bottom_row  = rows[-1]
            fps_row     = frame.shape[0] - 50
            anchor_row  = min(int(bottom_row), fps_row)
            cols_anchor = np.where(track_region[min(anchor_row, frame.shape[0] - 1)])[0]
            cx = int(np.mean(cols_anchor)) if len(cols_anchor) > 0 else int(centroids[i][0])
            label_text = f"Track {track_num}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx = max(4, min(cx - tw // 2, frame.shape[1] - tw - 8))
            ly = max(th + 6, anchor_row - 6)
            ly = min(ly, frame.shape[0] - 4)
            cv2.rectangle(overlay, (lx - 4, ly - th - 4), (lx + tw + 4, ly + baseline + 2), (0, 0, 0), -1)
            cv2.putText(overlay, label_text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    return overlay, mask_resized, track_detected

# ─── DETECT OBSTACLES ─────────────────────────────────────────────
def detect_obstacles(frame, mask, track_detected):
    global latest_alert, latest_detections

    t_start = time.time()
    results  = yolo(frame, verbose=False, conf=0.15)[0]
    yolo_time = (time.time() - t_start) * 1000
    session["yolo_times"].append(yolo_time)

    alert = False
    semi_alert = False
    detections = []

    mask_full = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    track_coords = np.column_stack(np.where(mask_full == 1))

    def nearest_track_dist(fx, fy):
        if track_coords.size == 0:
            return float('inf')
        diffs = track_coords - np.array([fy, fx])
        return float(np.hypot(diffs[:, 0], diffs[:, 1]).min())

    for box in results.boxes:
        cls_name = yolo.names[int(box.cls)]
        if cls_name not in OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))

        mid_y       = y1 + (y2 - y1) // 2
        bottom_half = mask_full[max(0, mid_y):min(frame.shape[0], y2),
                                max(0, x1):min(frame.shape[1], x2)]
        total_pix   = bottom_half.size
        track_pix   = np.sum(bottom_half == 1)
        overlap_pct = (track_pix / total_pix * 100) if total_pix > 0 else 0
        threshold   = OVERLAP_THRESHOLDS.get(cls_name, 5)
        on_track    = overlap_pct >= threshold

        foot_x = max(0, min((x1 + x2) // 2, frame.shape[1] - 1))
        foot_y = max(0, min(y2,              frame.shape[0] - 1))

        if track_detected and on_track:
            risk = "ON_TRACK"
        elif track_detected and track_coords.size > 0:
            dist = nearest_track_dist(foot_x, foot_y)
            risk = "TOO_CLOSE" if dist < PROXIMITY_WARN_PX else ("NEAR" if dist < PROXIMITY_SAFE_PX else "FAR")
        else:
            risk = "NEAR"

        color_map = {"ON_TRACK": (0, 0, 255), "TOO_CLOSE": (0, 100, 255),
                     "NEAR": (0, 255, 255), "FAR": (0, 200, 0)}
        color = color_map[risk]
        prefix = {"ON_TRACK": "ON TRACK: ", "TOO_CLOSE": "TOO CLOSE: ",
                  "NEAR": "", "FAR": ""}[risk]
        label = f"{prefix}{cls_name} ({conf:.0%})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        if risk == "ON_TRACK":
            alert = True
            detections.append({"class": cls_name, "confidence": conf, "risk": risk})
            session["detection_classes"][cls_name] = session["detection_classes"].get(cls_name, 0) + 1
            session["detection_confidences"].append(conf)
            session["total_detections"] += 1
            session["detection_log"].append({
                "time": time.strftime("%H:%M:%S"), "class": cls_name,
                "confidence": round(conf * 100, 1), "status": "ON TRACK"
            })
            print(f"  [KAVACH ALERT] ON TRACK: {cls_name} ({conf:.0%})")
        elif risk == "TOO_CLOSE":
            semi_alert = True
            detections.append({"class": cls_name, "confidence": conf, "risk": risk})
            session["detection_classes"][cls_name] = session["detection_classes"].get(cls_name, 0) + 1
            session["detection_confidences"].append(conf)
            session["total_detections"] += 1
            session["detection_log"].append({
                "time": time.strftime("%H:%M:%S"), "class": cls_name,
                "confidence": round(conf * 100, 1), "status": "TOO CLOSE"
            })
            print(f"  [SEMI-ALERT] TOO CLOSE: {cls_name} ({conf:.0%})")
        else:
            detections.append({"class": cls_name, "confidence": conf, "risk": risk})

    def _banner(img, text, font_scale, color, thickness=2):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = max(8, (DISPLAY_SIZE[0] - tw) // 2)
        cv2.putText(img, text, (tx, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    if not track_detected:
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (20, 80, 120), -1)
        _banner(frame, "NO TRACK DETECTED  —  MONITORING SURROUNDINGS", 0.70, (255, 255, 255))
    elif alert:
        session["total_alert_frames"] += 1
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (0, 0, 180), -1)
        _banner(frame, "!! KAVACH ALERT: OBSTACLE ON TRACK !!", 0.85, (255, 255, 255))
    elif semi_alert:
        session["total_alert_frames"] += 1
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (40, 40, 180), -1)
        _banner(frame, "SEMI-KAVACH ALERT: OBSTACLES TOO CLOSE TO TRACK!", 0.72, (255, 255, 255))
    else:
        cv2.rectangle(frame, (0, 0), (DISPLAY_SIZE[0], 60), (0, 120, 0), -1)
        _banner(frame, "TRACK CLEAR", 0.90, (255, 255, 255))

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (ts_w, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, timestamp, (DISPLAY_SIZE[0] - ts_w - 15, DISPLAY_SIZE[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cam_label = "Hardware Model Webcam" if CAMERA_MODE == "webcam" else "Static Image"
    cv2.putText(frame, cam_label, (20, DISPLAY_SIZE[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    latest_alert      = alert or semi_alert
    latest_detections = detections
    return frame

# ─── SESSION REPORT ───────────────────────────────────────────────
def build_report_lines():
    """Build and return the session report as a list of strings."""
    s            = session
    duration     = s["end_time"] - s["start_time"]
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
    avg_bisenet  = np.mean(s["bisenet_times"])         if s["bisenet_times"]         else 0
    avg_yolo     = np.mean(s["yolo_times"])            if s["yolo_times"]            else 0
    avg_fps      = np.mean(s["fps_samples"])           if s["fps_samples"]           else 0
    avg_coverage = np.mean(s["track_coverages"])       if s["track_coverages"]       else 0
    avg_conf     = np.mean(s["detection_confidences"]) * 100 if s["detection_confidences"] else 0
    alert_rate   = (s["total_alert_frames"] / s["total_frames_processed"] * 100) if s["total_frames_processed"] > 0 else 0

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("    DRISHTI KAVACH [Hardware Model] — SESSION REPORT")
    lines.append("=" * 60)
    lines.append(f"\n    Duration      : {duration_str}")
    lines.append(f"    Camera Mode   : {CAMERA_MODE.upper()}")
    lines.append(f"    BiSeNet Weights: {WEIGHTS_PATH}")
    lines.append(f"    Device        : {segmentor._device}")
    lines.append(f"\n    Frames Captured  : {s['total_frames_captured']}")
    lines.append(f"    Frames Processed : {s['total_frames_processed']}")
    lines.append(f"    Avg FPS          : {avg_fps:.1f}")
    lines.append(f"\n    Avg BiSeNet ms   : {avg_bisenet:.1f}")
    lines.append(f"    Avg YOLO ms      : {avg_yolo:.1f}")
    lines.append(f"    Avg Track Cover  : {avg_coverage:.2f}%")
    lines.append(f"\n    Total Detections : {s['total_detections']}")
    lines.append(f"    Alert Frames     : {s['total_alert_frames']}")
    lines.append(f"    Alert Rate       : {alert_rate:.1f}%")
    if s["detection_confidences"]:
        lines.append(f"    Avg Confidence   : {avg_conf:.1f}%")
    if s["detection_classes"]:
        lines.append("\n    Obstacles by Class:")
        for cls, cnt in sorted(s["detection_classes"].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"      → {cls:<15}: {cnt}")
    lines.append("\n" + "=" * 60 + "\n")
    return lines

def print_report():
    lines = build_report_lines()
    for line in lines:
        print(line)

def save_report():
    """Save the session report to a timestamped txt file in session_reports/."""
    report_time = time.strftime('%Y-%m-%d_%H%M%S')
    report_path = f"session_reports/hw_report_{report_time}.txt"
    lines = build_report_lines()
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Session report saved: {report_path}")

# ─── CAMERA THREADS ───────────────────────────────────────────────
def webcam_reader():
    global latest_raw, running
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    print("Warming up webcam...")
    for _ in range(30):
        cap.read()
    print("Webcam connected! ✓")
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        session["total_frames_captured"] += 1
        with raw_lock:
            latest_raw = frame
    cap.release()

def image_reader():
    global latest_raw, running
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print(f"ERROR: Could not load {TEST_IMAGE_PATH}")
        running = False
        return
    print(f"Image loaded: {TEST_IMAGE_PATH} ✓")
    while running:
        session["total_frames_captured"] += 1
        with raw_lock:
            latest_raw = frame.copy()
        time.sleep(0.1)

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
            output = detect_obstacles(overlay, mask, track_detected)
            last_processed = output.copy()
            with out_lock:
                latest_output = output
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
                cv2.putText(output, "TRACK CLEAR", (tx, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
                with out_lock:
                    latest_output = output

# ─── START ────────────────────────────────────────────────────────
if CAMERA_MODE == "webcam":
    t_cam = threading.Thread(target=webcam_reader, daemon=True)
elif CAMERA_MODE == "image":
    t_cam = threading.Thread(target=image_reader, daemon=True)
else:
    print(f"Unknown CAMERA_MODE: '{CAMERA_MODE}'")
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

print("Drishti Kavach [Hardware Model] is running!")
print("→ Press S to save a snapshot")
print("→ Press Q to quit\n")

# ─── DISPLAY LOOP ─────────────────────────────────────────────────
while True:
    with out_lock:
        if latest_output is not None:
            display = latest_output.copy()
        else:
            continue

    cv2.imshow("Drishti Kavach [Hardware Model]", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        snap = f"snapshots/hw_snapshot_{time.strftime('%Y-%m-%d_%H%M%S')}.jpg"
        cv2.imwrite(snap, display)
        print(f"  Snapshot saved: {snap}")

    if key == ord('q'):
        print("\nShutting down Drishti Kavach [Hardware Model]...")
        running = False
        session["end_time"] = time.time()
        break

cv2.destroyAllWindows()

write_state()
print_report()
save_report()
