"""
Final integrated script — Real-time camera pipeline:
- YOLOv8 (Ultralytics) for object detection
- MiDaS (Intel-ISL) for monocular depth estimation
- Simple anomaly (pothole/texture) detector via OpenCV as a backup
- Annotates each detected object with approximate distance (meters, relative)
- Runs on a single front-facing camera (or a video file)

USAGE:
    python drive_cv_pipeline.py --source 0
    python drive_cv_pipeline.py --source path/to/video.mp4

DEPENDENCIES (install with pip):
    pip install opencv-python ultralytics torch torchvision timm matplotlib numpy

NOTES:
- The script assumes the environment can download models on first run
  (YOLO weights and MiDaS model) via their respective libraries.
- Distance values from monocular MiDaS are relative. To convert to absolute
  meters you must calibrate against known distances / use stereo or LiDAR.
- Tweak thresholds (confidence, anomaly params) depending on camera/resolution.

Author: (your name)
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

# YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("Ultralytics YOLO is required. pip install ultralytics") from e

# MiDaS depth helpers (we'll try to load models via torch.hub)
# We'll build a small wrapper to load MiDaS and its transforms robustly.
def load_midas(device):
    """
    Load MiDaS (DPT_Large if available). Returns model and transform function.
    """
    # Try recommended DPT models (better) then fallback to MiDaS small
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        print("[MiDaS] Loaded DPT_Large model.")
    except Exception:
        # fallback
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        print("[MiDaS] Loaded MiDaS_small model (fallback).")

    midas.to(device)
    midas.eval()
    return midas, transform


def estimate_depth(midas, transform, device, frame_bgr):
    """
    Returns a depth map (numpy float32) scaled to 0..1 (relative depth)
    """
    # MiDaS expects RGB
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    # Normalize to 0..1 (relative)
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)
    return depth_norm.astype(np.float32)


def median_depth_in_box(depth_map, box):
    """
    box: [x1, y1, x2, y2] in pixel coords
    returns median depth value in that region
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_map[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def pothole_anomaly_detector(gray, ksize=5, lap_thresh=400):
    """
    Simple anomaly detector (very heuristic):
    - compute Laplacian variance in blocks to find texture discontinuities
    - returns list of bounding boxes for candidate anomalies
    """
    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap_abs = np.abs(lap).astype(np.uint8)

    # Threshold to get edges / sharp texture changes
    _, th = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # skip small noise - tune as needed
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Evaluate variance in Laplacian inside the rect as anomaly score
        roi = lap[y : y + h, x : x + w]
        var = np.var(roi)
        if var > lap_thresh:
            boxes.append((x, y, x + w, y + h, var))
    return boxes


def draw_annotations(frame, detections, depth_map, anomaly_boxes, fps):
    """
    detections: list of dicts: {'box': [x1,y1,x2,y2], 'label': str, 'conf': float, 'depth': float}
    anomaly_boxes: list of (x1,y1,x2,y2,score)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["label"]
        conf = det["conf"]
        depth = det.get("depth", None)
        color = (0, 200, 0)  # green for objects

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        if depth is not None:
            # depth is relative 0..1 — show a pseudo-meter by mapping to meters (requires calibration)
            pseudo_m = 0.5 + (1.0 - depth) * 50  # heuristic mapping: nearer -> smaller depth_norm -> larger pseudo_m
            # This mapping is arbitrary; replace with calibrated mapping if available.
            text += f" | ~{pseudo_m:.1f}m"
        cv2.putText(out, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw anomalies
    for (x1, y1, x2, y2, score) in anomaly_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange
        cv2.putText(out, f"anomaly {score:.0f}", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    # Depth overlay (mini)
    dm = (depth_map * 255).astype(np.uint8)
    dm_color = cv2.applyColorMap(cv2.resize(dm, (int(w * 0.25), int(h * 0.25))), cv2.COLORMAP_INFERNO)
    dh, dw = dm_color.shape[:2]
    out[0:dh, 0:dw] = dm_color

    # FPS
    cv2.putText(out, f"FPS: {fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return out


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load YOLOv8 model (small and fast by default; replace with larger model if needed)
    model = YOLO(args.yolo_weights)  # e.g., "yolov8n.pt"
    # Optionally set model conf threshold globally; we'll filter manually too
    model.fuse()

    # Load MiDaS
    midas, midas_transform = load_midas(device)

    # Video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture("Donia.mp4")   # put the path of your video file
#*********************************************************************************
    #if not cap.isOpened():
        #raise RuntimeError(f"Could not open source: {args.source}")
    # Warm-up runs
    print("[INFO] Warming up models...")
    _, sample = cap.read()
    if sample is None:
        raise RuntimeError("No frames in source.")
    # Warm-up YOLO a bit
    _ = model(sample, conf=0.25)
    # Warm-up MiDaS
    _ = estimate_depth(midas, midas_transform, device, sample)

    # For smoothing depth estimates and distances over frames
    depth_history = deque(maxlen=3)
    t0 = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or can't retrieve frame.")
                break
            frame_count += 1
            start = time.time()

            # Resize for speed (but keep original for display)
            # YOLO accepts native frames; ultralytics will handle resizing internally.
            # Run YOLO (returns Results object)
            results = model(frame, conf=args.conf_thres, imgsz=args.imgsz)
            res = results[0]

            detections = []
            # Extract boxes. results.boxes.xyxy is a tensor
            if hasattr(res, "boxes") and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                names = model.names

                for (box, score, cls) in zip(boxes_xyxy, scores, classes):
                    if score < args.conf_thres:
                        continue
                    label = names.get(cls, str(cls))
                    detections.append({"box": box.tolist(), "label": label, "conf": float(score)})

            # Depth estimation (MiDaS)
            depth_map = estimate_depth(midas, midas_transform, device, frame)
            depth_history.append(depth_map)
            # Smooth depth by median over recent frames
            depth_stack = np.stack(depth_history, axis=0)
            depth_map_smoothed = np.median(depth_stack, axis=0)

            # Compute median depth for each detection
            for det in detections:
                m = median_depth_in_box(depth_map_smoothed, det["box"])
                det["depth"] = m  # relative 0..1 (higher = farther in our normalization)

            # Anomaly detection (potholes) on grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            anomaly_boxes = pothole_anomaly_detector(gray, ksize=5, lap_thresh=args.anom_thresh)

            # Draw annotations
            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            annotated = draw_annotations(frame, detections, depth_map_smoothed, anomaly_boxes, fps)

            # Show
            cv2.imshow("Drive CV Pipeline", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # save snapshot
                fname = f"snap_{int(time.time())}.png"
                cv2.imwrite(fname, annotated)
                print(f"[INFO] Saved {fname}")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drive CV Pipeline - YOLOv8 + MiDaS + anomaly detection")
    parser.add_argument("--source", type=str, default="0", help="Camera source (0) or video file path")
    parser.add_argument("--yolo-weights", dest="yolo_weights", type=str, default="yolov8n.pt", help="YOLOv8 weights (ultralytics will download if missing)")
    parser.add_argument("--conf-thres", dest="conf_thres", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--imgsz", dest="imgsz", type=int, default=640, help="Inference image size for detection")
    parser.add_argument("--anom-thresh", dest="anom_thresh", type=float, default=450.0, help="Anomaly Laplacian variance threshold (tune)")
    args = parser.parse_args()
    main(args)
