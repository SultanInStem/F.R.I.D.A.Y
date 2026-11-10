"""
test_model.py - Live test of the YOLOv8 ONNX model on the RealSense D435i.

No kinematics, no IK, no socket to the Pi. It only runs detection, draws
boxes, reads depth at each box center, and reports timing and per-class
counts so you can judge the model before putting it on the arm.

Usage:
    python test_model.py                       # RealSense (default)
    python test_model.py --source webcam       # laptop webcam, no depth
    python test_model.py --source path/to/video.mp4
    python test_model.py --source path/to/image.jpg

Keys in the video window:
    q      quit
    s      save the current annotated frame
    + / -  raise / lower the confidence threshold by 0.05
    d      toggle the depth overlay (RealSense only)
"""

import argparse
import os
import time
from collections import Counter

import cv2
import numpy as np

# ─────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "./AI_model/yolo/best.onnx"
NAMES_PATH  = "./AI_model/yolo/best.names"   # optional
# Fallback if no .names file and no ONNX metadata. Order MUST match data.yaml.
DEFAULT_NAMES = ["peduncle", "ripe", "unripe"]
INPUT_SIZE  = (640, 640)      # must match the imgsz used at export
CONF_THRESH = 0.50
NMS_THRESH  = 0.45
DEPTH_PATCH = 5               # median over a DEPTH_PATCH x DEPTH_PATCH window

# One BGR colour per class (cycles if there are more classes)
PALETTE = [(60, 200, 60), (40, 40, 230), (0, 200, 255),
           (255, 160, 0), (200, 0, 200), (255, 255, 0)]


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    net = cv2.dnn.readNetFromONNX(model_path)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA backend")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("CUDA not found, using CPU")
    return net


def load_classes(names_path, model_path):
    """.names file -> Ultralytics ONNX metadata -> DEFAULT_NAMES."""
    if names_path and os.path.exists(names_path):
        with open(names_path) as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"Class names from {names_path}")
        return names
    try:
        import ast
        import onnx
        meta = {p.key: p.value for p in onnx.load(model_path).metadata_props}
        if "names" in meta:
            names_dict = ast.literal_eval(meta["names"])   # {0: 'peduncle', ...}
            print("Class names from ONNX metadata")
            return [names_dict[i] for i in sorted(names_dict)]
    except Exception:
        pass
    print("Class names from DEFAULT_NAMES - check the order against data.yaml!")
    return DEFAULT_NAMES


# ─────────────────────────────────────────────
# DETECTION (same pipeline as the picking script)
# ─────────────────────────────────────────────
def letterbox(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (tw - new_w) // 2, (th - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def detect(net, frame, class_names, conf_thresh):
    letterboxed, scale, pad_x, pad_y = letterbox(frame, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, INPUT_SIZE,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    output = np.squeeze(net.forward()).T       # (8400, 4 + nc)

    expected = 4 + len(class_names)
    if output.shape[1] != expected:
        raise ValueError(
            f"Model output has {output.shape[1]} channels, expected {expected} "
            f"(4 box + {len(class_names)} classes). Class list doesn't match the model.")

    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []
    for det in output:
        scores = det[4:]
        cid = int(np.argmax(scores))
        conf = float(scores[cid])
        if conf < conf_thresh:
            continue
        cx = (det[0] - pad_x) / scale
        cy = (det[1] - pad_y) / scale
        bw, bh = det[2] / scale, det[3] / scale
        boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
        confidences.append(conf)
        class_ids.append(cid)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, NMS_THRESH)
    results = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, bw, bh = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append((class_ids[i], confidences[i], x1, y1, x2, y2))
    return results


def median_depth(depth_frame, cx, cy, half=DEPTH_PATCH // 2):
    """Median of the valid depths in a small window - robust to specular dropouts."""
    w, h = depth_frame.get_width(), depth_frame.get_height()
    vals = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    vals.append(d)
    return float(np.median(vals)) if vals else 0.0


# ─────────────────────────────────────────────
# FRAME SOURCES
# ─────────────────────────────────────────────
class RealSenseSource:
    has_depth = True

    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth, color = frames.get_depth_frame(), frames.get_color_frame()
        if not depth or not color:
            return None, None
        return np.asanyarray(color.get_data()), depth

    def deproject(self, depth_frame, cx, cy, depth_m):
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        return self.rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_m)

    def colorize(self, depth_frame):
        return np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

    def release(self):
        self.pipeline.stop()


class OpenCVSource:
    has_depth = False

    def __init__(self, src):
        self.is_image = isinstance(src, str) and src.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp"))
        if self.is_image:
            self.image = cv2.imread(src)
            if self.image is None:
                raise FileNotFoundError(src)
        else:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open source: {src}")

    def read(self):
        if self.is_image:
            return self.image.copy(), None
        ok, frame = self.cap.read()
        return (frame if ok else None), None

    def release(self):
        if not self.is_image:
            self.cap.release()


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="realsense",
                        help="'realsense', 'webcam', a camera index, or a video/image path")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--names", default=NAMES_PATH)
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    args = parser.parse_args()

    if args.source == "realsense":
        source = RealSenseSource()
    elif args.source == "webcam":
        source = OpenCVSource(0)
    elif args.source.isdigit():
        source = OpenCVSource(int(args.source))
    else:
        source = OpenCVSource(args.source)

    net = load_model(args.model)
    class_names = load_classes(args.names, args.model)
    conf_thresh = args.conf
    print(f"Loaded {len(class_names)} classes: {class_names}")
    print("Keys: q quit | s save | +/- conf | d depth overlay")

    os.makedirs("test_captures", exist_ok=True)
    show_depth = False
    frame_count = 0
    infer_times = []
    class_counts = Counter()        # total detections per class across all frames
    frames_with = Counter()         # frames in which each class appeared
    t_start = time.time()
    fps_t, fps = time.time(), 0.0

    try:
        while True:
            color_image, depth_frame = source.read()
            if color_image is None:
                if getattr(source, "is_image", False):
                    continue
                print("End of stream.")
                break
            frame_count += 1

            t0 = time.perf_counter()
            detections = detect(net, color_image, class_names, conf_thresh)
            infer_ms = (time.perf_counter() - t0) * 1000
            infer_times.append(infer_ms)

            seen_this_frame = set()
            for (cid, conf, x1, y1, x2, y2) in detections:
                label = class_names[cid] if cid < len(class_names) else str(cid)
                color = PALETTE[cid % len(PALETTE)]
                class_counts[label] += 1
                seen_this_frame.add(label)

                text = f"{label} {conf:.2f}"
                if source.has_depth:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    depth_m = median_depth(depth_frame, cx, cy)
                    if depth_m > 0:
                        X, Y, Z = source.deproject(depth_frame, cx, cy, depth_m)
                        text += f" | cam {X*1000:.0f},{Y*1000:.0f},{Z*1000:.0f} mm"
                    else:
                        text += " | no depth"
                    cv2.circle(color_image, (cx, cy), 3, color, -1)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, text, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            frames_with.update(seen_this_frame)

            # FPS (smoothed, updated twice a second)
            if time.time() - fps_t > 0.5:
                fps = frame_count / (time.time() - t_start)
                fps_t = time.time()

            counts_now = Counter(class_names[d[0]] for d in detections
                                 if d[0] < len(class_names))
            hud = (f"FPS {fps:.1f}  infer {infer_ms:.0f} ms  conf>={conf_thresh:.2f}  "
                   + "  ".join(f"{k}:{v}" for k, v in sorted(counts_now.items())))
            cv2.putText(color_image, hud, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            display = color_image
            if show_depth and source.has_depth:
                display = np.hstack([color_image, source.colorize(depth_frame)])
            cv2.imshow("Model test", display)

            key = cv2.waitKey(0 if getattr(source, "is_image", False) else 1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = f"test_captures/frame_{int(time.time())}.png"
                cv2.imwrite(path, display)
                print(f"Saved {path}")
            elif key in (ord("+"), ord("=")):
                conf_thresh = min(0.95, conf_thresh + 0.05)
                print(f"conf threshold -> {conf_thresh:.2f}")
            elif key == ord("-"):
                conf_thresh = max(0.05, conf_thresh - 0.05)
                print(f"conf threshold -> {conf_thresh:.2f}")
            elif key == ord("d"):
                show_depth = not show_depth

    finally:
        source.release()
        cv2.destroyAllWindows()
        if infer_times:
            t = np.array(infer_times[5:] or infer_times)   # skip warm-up frames
            print("\n=== Summary ===")
            print(f"Frames        : {frame_count}")
            print(f"Avg FPS       : {frame_count / (time.time() - t_start):.1f}")
            print(f"Inference ms  : mean {t.mean():.1f}  median {np.median(t):.1f}  "
                  f"p95 {np.percentile(t, 95):.1f}")
            for name in class_names:
                print(f"{name:12s}: {class_counts[name]:5d} detections, "
                      f"present in {frames_with[name] / frame_count * 100:5.1f}% of frames")


if __name__ == "__main__":
    main()