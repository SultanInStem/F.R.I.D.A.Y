import pyrealsense2 as rs
import numpy as np
import cv2
import warnings
import json
import time
import socket

# ─────────────────────────────────────────────
# KINEMATIC CHAIN
# ─────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ikpy.chain import Chain
    chain = Chain.from_urdf_file(
        "./mycobot_320pi.urdf",
        active_links_mask=[False, True, True, True, True, True, True, False]
    )

with open("cam2base.json") as f:
    T_CAM2BASE = np.array(json.load(f)["T_cam2base"])

GRIPPER_LENGTH = 0.13   # flange-to-tip, measured

# ─────────────────────────────────────────────
# PICK CONFIG
# ─────────────────────────────────────────────
PI_HOST       = "192.168.10.2"
PI_PORT       = 65432
PI_TIMEOUT    = 60.0     # s to wait for the Pi to finish the pick and ack
GRIPPER_VALUE = 20       # 0 = closed, 100 = open
GRIPPER_SPEED = 40
COOLDOWN_S    = 3.0      # dead time after a pick before re-arming

# ─────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "./AI_model/yolov8n_apples/my_model.onnx"
NAMES_PATH  = "./AI_model/yolov8n_apples/my_model.names"
INPUT_SIZE  = (640, 640)
CONF_THRESH = 0.60
NMS_THRESH  = 0.4
TARGET_CLASS = "apple"   # only pick this class; set to None to pick any

# ─────────────────────────────────────────────
# DETECTION / STABILITY CONFIG
# ─────────────────────────────────────────────
DETECTION_FRAMES     = 15      # consecutive stable frames required
STABILITY_TOL        = 0.010   # m - max SD across the buffer on every axis
CENTER_THRESHOLD     = 9999    # px
BRIGHTNESS_THRESHOLD = 10
DEPTH_PATCH          = 5       # median over a DEPTH_PATCH x DEPTH_PATCH window
frame_center_x       = 640 // 2
frame_center_y       = 480 // 2


# ─────────────────────────────────────────────
# IK
# ─────────────────────────────────────────────
def compute_angles(point_in_base_frame):
    """Returns the full ikpy angle vector (radians), or [] if unreachable."""
    angles = chain.inverse_kinematics(
        point_in_base_frame,
        [0, 0, -1],
        orientation_mode="Z",
        optimizer="least_squares",
        max_iter=1000,
    )
    achieved = chain.forward_kinematics(angles)[:3, 3]
    residual = np.linalg.norm(achieved - point_in_base_frame)
    if residual > 0.005:
        print(f"  IK residual {residual*1000:.1f} mm - point outside working radius")
        return []
    return angles


def angles_to_degrees(angles):
    """ikpy vector (8 links) -> the 6 joint angles in degrees for pymycobot."""
    return [round(float(np.degrees(a)), 3) for a in angles[1:7]]


# ─────────────────────────────────────────────
# SOCKET
# ─────────────────────────────────────────────
def send_to_pi(joint_deg, gripper_value, gripper_speed):
    """
    Sends one pick command and blocks until the Pi acks.
    Wire format out : "PICK,j1,j2,j3,j4,j5,j6,grip_value,grip_speed\n"
    Wire format back: "DONE\n" or "FAIL,<reason>\n"
    Returns (ok: bool, reply: str).
    """
    msg = "PICK," + ",".join(f"{a:.3f}" for a in joint_deg)
    msg += f",{gripper_value},{gripper_speed}\n"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(PI_TIMEOUT)
            s.connect((PI_HOST, PI_PORT))
            s.sendall(msg.encode())
            reply = s.recv(1024).decode().strip()
        return reply.startswith("DONE"), reply
    except socket.timeout:
        return False, "TIMEOUT"
    except socket.error as e:
        return False, f"SOCKET_ERROR:{e}"


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_model(model_path):
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


def load_classes(names_path):
    with open(names_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def get_output_layers(net):
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    if isinstance(unconnected[0], (list, np.ndarray)):
        return [layer_names[i[0] - 1] for i in unconnected]
    return [layer_names[i - 1] for i in unconnected]


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


def detect(net, frame, class_names):
    letterboxed, scale, pad_x, pad_y = letterbox(frame, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, INPUT_SIZE,
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(get_output_layers(net))
    output = np.squeeze(outputs[0]).T

    boxes, confidences, class_ids = [], [], []
    for det in output:
        scores = det[4:]
        cid = int(np.argmax(scores))
        conf = float(scores[cid])
        if conf < CONF_THRESH:
            continue
        cx = (det[0] - pad_x) / scale
        cy = (det[1] - pad_y) / scale
        bw, bh = det[2] / scale, det[3] / scale
        boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
        confidences.append(conf)
        class_ids.append(cid)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = (class_names[class_ids[i]]
                     if class_ids[i] < len(class_names) else str(class_ids[i]))
            results.append((label, confidences[i], x, y, x + bw, y + bh))
    return results


def median_depth(depth_frame, cx, cy, half=DEPTH_PATCH // 2):
    """Median of the valid depths in a small window - robust to specular dropouts."""
    vals = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            d = depth_frame.get_distance(cx + dx, cy + dy)
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0


# ─────────────────────────────────────────────
# REALSENSE
# ─────────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

net = load_model(MODEL_PATH)
class_names = load_classes(NAMES_PATH)
print(f"Loaded {len(class_names)} classes: {class_names}")
print(f"CONF_THRESH={CONF_THRESH}  target class={TARGET_CLASS}  "
      f"stability={STABILITY_TOL*1000:.0f} mm over {DETECTION_FRAMES} frames")
print("Picking continuously. Press q in the video window to stop.")

pick_count = 0
coords_buffer = []
conf_buffer = []
armed_at = 0.0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # intrinsics MUST come from the colour-aligned depth profile
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())

        detections = detect(net, color_image, class_names)

        # ---- pick ONE target per frame: highest-confidence valid detection ----
        best = None
        for (label, conf, x1, y1, x2, y2) in detections:
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(color_image.shape[1] - 1, x2)
            y2 = min(color_image.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if TARGET_CLASS is not None and label != TARGET_CLASS:
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue

            roi = cv2.cvtColor(color_image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if roi.mean() < BRIGHTNESS_THRESHOLD:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth_m = median_depth(depth_frame, cx, cy)
            if depth_m == 0:
                continue

            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
            p_base = (T_CAM2BASE @ np.array([*point_3d, 1.0]))[:3]
            p_base = [p_base[0], p_base[1], p_base[2] + GRIPPER_LENGTH]

            centered = (abs(cx - frame_center_x) < CENTER_THRESHOLD and
                        abs(cy - frame_center_y) < CENTER_THRESHOLD)

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image,
                        f"{label} {conf:.2f} | {p_base[0]*1000:.0f},"
                        f"{p_base[1]*1000:.0f},{p_base[2]*1000:.0f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if best is None or conf > best[1]:
                best = (label, conf, p_base, centered)

        # ---- accumulate ONE sample per frame ----
        in_cooldown = (time.time() - armed_at) < COOLDOWN_S
        if best is not None and best[3] and not in_cooldown:
            coords_buffer.append(best[2])
            conf_buffer.append(best[1])
        else:
            coords_buffer.clear()      # any dropped frame restarts the count
            conf_buffer.clear()

        cv2.putText(color_image,
                    f"picks {pick_count}  stable {len(coords_buffer)}/{DETECTION_FRAMES}"
                    + ("  [cooldown]" if in_cooldown else ""),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ─────────────────────────────────────
        # STABLE DETECTION -> PICK
        # ─────────────────────────────────────
        if len(coords_buffer) >= DETECTION_FRAMES:
            arr = np.array(coords_buffer)
            avg = arr.mean(axis=0)
            sd = arr.std(axis=0)

            if np.all(sd < STABILITY_TOL):
                t0 = time.perf_counter()
                pick_count += 1

                print(f"\n=== PICK {pick_count} ===")
                print(f"  target  : {avg[0]*1000:.1f}, {avg[1]*1000:.1f}, "
                      f"{avg[2]*1000:.1f} mm   (SD {sd[0]*1000:.1f}/"
                      f"{sd[1]*1000:.1f}/{sd[2]*1000:.1f})")
                print(f"  class   : {best[0]}  conf {np.mean(conf_buffer):.2f}")

                angles = compute_angles(list(avg))
                if angles is not None and len(angles) > 0:
                    joint_deg = angles_to_degrees(angles)
                    print(f"  angles  : {joint_deg}")
                    ok, pi_reply = send_to_pi(joint_deg, GRIPPER_VALUE, GRIPPER_SPEED)
                    print(f"  pi      : {pi_reply}")
                else:
                    print("  skipped : IK failed")

                print(f"  cycle   : {time.perf_counter() - t0:.2f} s")
                armed_at = time.time()

            coords_buffer.clear()
            conf_buffer.clear()

        cv2.imshow("Autonomous picking", color_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"Stopped after {pick_count} pick attempts.")